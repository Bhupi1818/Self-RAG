"""Pipeline logic tests with a scripted fake LLM and a fake retriever (no network, no models)."""
from src.config import Settings
from src.pipeline import SelfRAGPipeline
from src.prompts import REFUSAL_TEXT
from src.schemas import Chunk
from src.utils import extract_json


class FakeLLM:
    """Routes by system-prompt keyword and returns scripted responses."""

    def __init__(self, script):
        self.script = script
        self.calls = 0

    def complete(self, system, user, max_tokens=0, temperature=0.0):
        self.calls += 1
        for key, resp in self.script.items():
            if key in system:
                return resp.pop(0) if isinstance(resp, list) else resp
        raise AssertionError(f"unexpected prompt: {system[:40]}")

    def complete_json(self, system, user, max_tokens=0, retries=2):
        return extract_json(self.complete(system, user, max_tokens))


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=None):
        self.queries.append(query)
        return [Chunk(id=f"c{i}", text=f"text {i}", source="doc", page=1, score=0.9 - i / 10) for i in range(4)]


CFG = Settings(max_retries=1, min_relevant=2, support_threshold=0.8, use_router=True, top_k=4)
ALL_REL = '{"grades": [{"id": 1, "relevant": true}, {"id": 2, "relevant": true}, {"id": 3, "relevant": false}, {"id": 4, "relevant": false}]}'
NONE_REL = '{"grades": [{"id": 1, "relevant": false}, {"id": 2, "relevant": false}, {"id": 3, "relevant": false}, {"id": 4, "relevant": false}]}'
GOOD_VERIFY = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": true}], "answers_question": true}'
BAD_VERIFY = '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}], "answers_question": true}'


def test_happy_path_single_attempt():
    llm = FakeLLM({
        "decide whether": '{"needs_retrieval": true}',
        "relevance grader": ALL_REL,
        "ONLY the numbered context": "The answer is 42 [1].",
        "fact-checker": GOOD_VERIFY,
    })
    res = SelfRAGPipeline(llm, FakeRetriever(), CFG).answer("what is the answer?")
    assert not res.refused and res.attempts == 1 and res.confidence == 1.0
    assert len(res.sources) == 2


def test_router_skips_retrieval_for_smalltalk():
    llm = FakeLLM({"decide whether": '{"needs_retrieval": false}', "friendly assistant": "Hello!"})
    retriever = FakeRetriever()
    res = SelfRAGPipeline(llm, retriever, CFG).answer("hi there")
    assert res.answer == "Hello!" and not res.used_retrieval and retriever.queries == []


def test_rewrites_query_when_nothing_relevant_then_succeeds():
    llm = FakeLLM({
        "decide whether": '{"needs_retrieval": true}',
        "relevance grader": [NONE_REL, ALL_REL],
        "rewrite search queries": '{"query": "better query"}',
        "ONLY the numbered context": "Grounded answer [1].",
        "fact-checker": GOOD_VERIFY,
    })
    retriever = FakeRetriever()
    res = SelfRAGPipeline(llm, retriever, CFG).answer("vague question")
    assert retriever.queries == ["vague question", "better query"]
    assert not res.refused and res.attempts == 2
    assert any(s["step"] == "rewrite" for s in res.trace)


def test_refuses_after_exhausting_retries():
    llm = FakeLLM({
        "decide whether": '{"needs_retrieval": true}',
        "relevance grader": [NONE_REL, NONE_REL],
        "rewrite search queries": '{"query": "q2"}',
    })
    res = SelfRAGPipeline(llm, FakeRetriever(), CFG).answer("unanswerable")
    assert res.refused and res.answer == REFUSAL_TEXT and res.confidence == 0.0


def test_unsupported_claims_trigger_retry_not_acceptance():
    llm = FakeLLM({
        "decide whether": '{"needs_retrieval": true}',
        "relevance grader": [ALL_REL, ALL_REL],
        "ONLY the numbered context": ["Half-made-up answer.", "Corrected answer [1]."],
        "fact-checker": [BAD_VERIFY, GOOD_VERIFY],
        "rewrite search queries": '{"query": "refined"}',
    })
    res = SelfRAGPipeline(llm, FakeRetriever(), CFG).answer("tricky")
    assert res.answer == "Corrected answer [1]." and res.attempts == 2


def test_baseline_never_checks_or_rewrites():
    llm = FakeLLM({"ONLY the numbered context": "Baseline answer."})
    res = SelfRAGPipeline(llm, FakeRetriever(), CFG).baseline("q")
    assert res.mode == "baseline" and res.llm_calls == 1



def test_repeated_query_stops_without_regrading_same_state():
    llm = FakeLLM({
        "decide whether": '{"needs_retrieval": true}',
        "relevance grader": [NONE_REL],
        "rewrite search queries": '{"query": "vague question"}',
    })
    retriever = FakeRetriever()

    res = SelfRAGPipeline(llm, retriever, CFG).answer("vague question")

    assert retriever.queries == ["vague question"]
    assert res.refused
    assert any(
        step.get("reason") == "query_repeated"
        for step in res.trace
    )
