"""Create eval/testset.json from your own indexed corpus.

Answerable questions are LLM-generated from random chunks; unanswerable ones come from a
list (extend it with questions that sound on-topic but are NOT in your documents).

    python eval/generate_testset.py --n 40
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.llm import LLM
from src.prompts import TESTGEN_SYSTEM

UNANSWERABLE = [
    "What was NVIDIA's closing stock price yesterday?",
    "Who won the most recent FIFA World Cup?",
    "What is the exact GPU memory required to run GPT-5 locally?",
    "What did the CEO of Anthropic say about this topic in 2026?",
    "Give the recipe for butter chicken.",
    "What is the population of Jaipur?",
    "How many parameters does the largest model released next year have?",
    "Which Python version introduced the walrus operator?",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of answerable questions")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import chromadb

    col = chromadb.PersistentClient(path=settings.chroma_dir).get_collection(settings.collection_name)
    total = col.count()
    random.seed(args.seed)
    offsets = random.sample(range(total), min(args.n * 2, total))  # oversample; some will be skipped

    llm = LLM(model=settings.judge_model)
    items = []
    for off in offsets:
        if len([i for i in items if i["answerable"]]) >= args.n:
            break
        got = col.get(limit=1, offset=off, include=["documents", "metadatas"])
        text, meta = got["documents"][0], got["metadatas"][0]
        if len(text.split()) < 80:
            continue
        try:
            out = llm.complete_json(TESTGEN_SYSTEM, f"Passage:\n{text}", max_tokens=300)
            items.append(
                {
                    "question": out["question"],
                    "reference_answer": out["reference_answer"],
                    "answerable": True,
                    "source": meta["source"],
                }
            )
        except (ValueError, KeyError):
            continue

    for q in UNANSWERABLE:
        items.append({"question": q, "reference_answer": "", "answerable": False, "source": ""})

    Path("eval/testset.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} questions ({sum(i['answerable'] for i in items)} answerable) to eval/testset.json")


if __name__ == "__main__":
    main()
