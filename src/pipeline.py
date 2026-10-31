"""The self-reflective RAG loop: route -> retrieve -> grade -> generate -> verify -> (rewrite & retry)."""
import time

from .config import Settings, settings
from .generator import generate, generate_direct
from .grader import grade_chunks
from .llm import LLM
from .prompts import REFUSAL_TEXT, is_refusal
from .rewriter import rewrite
from .router import route
from .schemas import Result
from .verifier import verify


class SelfRAGPipeline:
    def __init__(self, llm: LLM, retriever, cfg: Settings = settings):
        self.llm = llm
        self.retriever = retriever
        self.cfg = cfg

    # ------------------------------------------------------------------ baseline
    def baseline(self, question: str) -> Result:
        """Plain RAG (retrieve -> generate). Used as the comparison point in evaluation."""
        t0, c0 = time.time(), self.llm.calls
        chunks = self.retriever.retrieve(question, self.cfg.top_k)
        answer = generate(self.llm, question, chunks)
        return Result(
            question=question,
            answer=answer,
            mode="baseline",
            sources=[c.to_source() for c in chunks],
            contexts=[c.text for c in chunks],
            refused=is_refusal(answer),
            attempts=1,
            trace=[{"step": "retrieve", "query": question, "n": len(chunks)}],
            llm_calls=self.llm.calls - c0,
            latency_s=round(time.time() - t0, 2),
        )

    # ------------------------------------------------------------------ self-rag
    def answer(self, question: str) -> Result:
        t0, c0 = time.time(), self.llm.calls
        trace: list[dict] = []

        # Keep the best non-refusal candidate across retries.
        best_answer = None
        # best_sources = []
        # best_contexts = []
        best_confidence = -1.0
        best_answers_question = False

        # remember retrival results already seen during this request
        seen_retrieval_states = set()
        seen_queries = set()

        def finish(**kw) -> Result:
            return Result(
                question=question,
                mode="self_rag",
                trace=trace,
                llm_calls=self.llm.calls - c0,
                latency_s=round(time.time() - t0, 2),
                **kw,
            )

        # 1) Router
        if self.cfg.use_router:
            needs = route(self.llm, question)
            trace.append({"step": "route", "needs_retrieval": needs})
            if not needs:
                return finish(
                    answer=generate_direct(self.llm, question),
                    used_retrieval=False,
                    confidence=1.0,
                )

        # 2) Retrieve -> grade -> generate -> verify loop
        query = question
        for attempt in range(self.cfg.max_retries + 1):
            if query in seen_queries:
                trace.append({
                    "step": "stop",
                    "attempt": attempt,
                    "reason": "query_repeated",
                })
                break

            seen_queries.add(query)
            chunks = self.retriever.retrieve(query, self.cfg.top_k)
            trace.append(
                {
                    "step": "retrieve",
                    "attempt": attempt,
                    "query": query,
                    "n": len(chunks),
                    "top_score": round(chunks[0].score, 3) if chunks else None,
                }
            )

            grades = grade_chunks(self.llm, question, chunks)
            relevant = [c for c, g in zip(chunks, grades) if g.relevant]
            trace.append(
                {
                    "step": "grade",
                    "attempt": attempt,
                    "relevant": len(relevant),
                    "rejected": [
                        {"id": c.id, "reason": g.reason} for c, g in zip(chunks, grades) if not g.relevant
                    ],
                }
            )

            current_relevant_ids = tuple(sorted(c.id for c in relevant))
            retrieval_state = (query, current_relevant_ids)

            # Stop if this exact query + relevant passages were already seen.
            if retrieval_state in seen_retrieval_states:
                trace.append(
                    {
                        "step": "stop",
                        "attempt": attempt,
                        "reason": "retrieval_state_repeated",
                    }
                )
                break

            seen_retrieval_states.add(retrieval_state)

            failure_reason = ""

            if len(relevant) < self.cfg.min_relevant:
                failure_reason = f"only {len(relevant)} relevant passages retrieved"
            else:
                draft = generate(self.llm, question, relevant)
                if is_refusal(draft):
                    trace.append(
                        {
                            "step": "stop",
                            "attempt": attempt,
                            "reason": "generator_refused",
                        }
                    )

                    return finish(
                        answer=REFUSAL_TEXT,
                        sources=[c.to_source() for c in relevant],
                        contexts=[c.text for c in relevant],
                        refused=True,
                        attempts=attempt + 1,
                        confidence=0.0,
                    )
                else:
                    check = verify(self.llm, question, draft, relevant)
                trace.append(
                    {
                        "step": "verify",
                        "attempt": attempt,
                        "supported_ratio": round(check.supported_ratio, 2),
                        "answers_question": check.answers_question,
                        "unsupported_claims": check.unsupported_claims,
                    }
                )
                # Keep the strongest candidate seen so far.
                # A later retry should not automatically replace a better answer.
                if not is_refusal(draft):
                    candidate_score = check.supported_ratio

                    if (
                        best_answer is None
                        or candidate_score > best_confidence
                        or (
                            candidate_score == best_confidence
                            and check.answers_question
                            and not best_answers_question
                        )
                    ):
                        best_answer = draft
                        # best_sources = [c.to_source() for c in relevant]
                        # best_contexts = [c.text for c in relevant]
                        best_confidence = candidate_score
                        best_answers_question = check.answers_question

                if check.supported_ratio >= self.cfg.support_threshold and check.answers_question:
                    return finish(
                        answer=draft,
                        sources=[c.to_source() for c in relevant],
                        contexts=[c.text for c in relevant],
                        confidence=round(check.supported_ratio, 2),
                        attempts=attempt + 1,
                    )
                if not failure_reason:
                    failure_reason = "answer not fully supported: " + "; ".join(check.unsupported_claims[:3])


            # 3) Rewrite and retry (not after the last attempt)
            if attempt < self.cfg.max_retries:
                query = rewrite(self.llm, question, query, failure_reason)
                trace.append(
                    {
                        "step": "rewrite",
                        "attempt": attempt,
                        "new_query": query,
                        "reason": failure_reason,
                    }
                )

        # 4) No candidate passed verification: fail closed.
        return finish(
            answer=REFUSAL_TEXT,
            refused=True,
            attempts=self.cfg.max_retries + 1,
            confidence=0.0,
        )





def build_pipeline(cfg: Settings = settings) -> SelfRAGPipeline:
    from .retriever import Retriever

    return SelfRAGPipeline(LLM(cfg=cfg), Retriever(cfg), cfg)
