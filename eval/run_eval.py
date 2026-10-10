"""Compare baseline RAG vs self-reflective RAG on eval/testset.json.

    python eval/run_eval.py [--limit 20]

Metrics
  correctness     judge score vs reference (answerable questions)
  faithfulness    claim-level support of the answer by the context it used
  refusal_acc     unanswerable questions correctly refused
  false_refusal   answerable questions wrongly refused
  avg_llm_calls / avg_latency_s   cost of the extra reflection steps
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.llm import LLM
from src.pipeline import build_pipeline
from src.prompts import JUDGE_SYSTEM
from src.schemas import Chunk
from src.verifier import verify


def judge_correctness(judge: LLM, question: str, reference: str, answer: str) -> float:
    try:
        out = judge.complete_json(
            JUDGE_SYSTEM, f"Question: {question}\nReference: {reference}\nAnswer: {answer}", max_tokens=150
        )
        return float(out.get("score", 0.0))
    except (ValueError, TypeError):
        return 0.0


def judge_faithfulness(judge: LLM, question: str, answer: str, contexts: list[str]) -> float:
    chunks = [Chunk(id=str(i), text=t, source="ctx", page=0) for i, t in enumerate(contexts)]
    return verify(judge, question, answer, chunks).supported_ratio


def mean(xs):
    return round(st.mean(xs), 3) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = json.loads(Path("eval/testset.json").read_text(encoding="utf-8"))
    if args.limit:
        items = items[: args.limit]

    pipe = build_pipeline()
    judge = LLM(model=settings.judge_model)  # separate model/instance => separate call counter

    rows = {"baseline": [], "self_rag": []}
    for n, it in enumerate(items, start=1):
        print(f"[{n}/{len(items)}] {it['question'][:70]}")
        for mode in rows:
            res = pipe.baseline(it["question"]) if mode == "baseline" else pipe.answer(it["question"])
            row = {
                "question": it["question"],
                "answerable": it["answerable"],
                "answer": res.answer,
                "refused": res.refused,
                "llm_calls": res.llm_calls,
                "latency_s": res.latency_s,
            }
            if it["answerable"] and not res.refused:
                row["correctness"] = judge_correctness(judge, it["question"], it["reference_answer"], res.answer)
                row["faithfulness"] = judge_faithfulness(judge, it["question"], res.answer, res.contexts)
            elif it["answerable"]:
                row["correctness"] = 0.0
            rows[mode].append(row)

    summary = {}
    for mode, rs in rows.items():
        ans = [r for r in rs if r["answerable"]]
        unans = [r for r in rs if not r["answerable"]]
        summary[mode] = {
            "correctness": mean([r["correctness"] for r in ans if "correctness" in r]),
            "faithfulness": mean([r["faithfulness"] for r in ans if "faithfulness" in r]),
            "refusal_acc": mean([1.0 if r["refused"] else 0.0 for r in unans]),
            "false_refusal": mean([1.0 if r["refused"] else 0.0 for r in ans]),
            "avg_llm_calls": mean([r["llm_calls"] for r in rs]),
            "avg_latency_s": mean([r["latency_s"] for r in rs]),
        }

    Path("results").mkdir(exist_ok=True)
    Path("results/eval_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    metrics = list(summary["baseline"].keys())
    lines = ["| Metric | Baseline RAG | Self-reflective RAG |", "|---|---|---|"]
    lines += [f"| {m} | {summary['baseline'][m]} | {summary['self_rag'][m]} |" for m in metrics]
    table = "\n".join(lines)
    Path("results/eval_table.md").write_text(table + "\n", encoding="utf-8")
    print("\n" + table)


if __name__ == "__main__":
    main()
