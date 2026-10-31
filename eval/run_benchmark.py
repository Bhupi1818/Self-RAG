import json
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

API_URL = "http://127.0.0.1:8000/ask"
DATASET = Path(__file__).parent / "benchmark.json"
OUTPUT = Path(__file__).parent / "benchmark_results.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_PASS_THRESHOLD = 0.55
MISLABEL_SIMILARITY_THRESHOLD = 0.65

print(f"Loading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)


def ask(question, mode):
    start = time.time()
    response = requests.post(
        API_URL,
        json={"question": question, "mode": mode},
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    return result, round(time.time() - start, 2)


def keyword_score(answer, keywords):
    if not keywords:
        return None
    answer = answer.lower()
    matched = [k for k in keywords if k.lower() in answer]
    return round(len(matched) / len(keywords), 2)


def semantic_similarity(answer, reference_answer):
    if not answer or not reference_answer:
        return 0.0
    embeddings = embedder.encode(
        [answer, reference_answer],
        normalize_embeddings=True,
    )
    return round(float(np.dot(embeddings[0], embeddings[1])), 3)


def evaluate_answer(answer, reference_answer, keywords, answerable, refused):
    kw = keyword_score(answer, keywords)
    sim = semantic_similarity(answer, reference_answer)

    if answerable:
        semantic_pass = sim >= SEMANTIC_PASS_THRESHOLD
        correct = semantic_pass
        label_issue = False
    else:
        semantic_pass = False
        correct = refused
        label_issue = (not refused and sim >= MISLABEL_SIMILARITY_THRESHOLD)

    return {
        "keyword_score": kw,
        "semantic_similarity": sim,
        "semantic_pass": semantic_pass,
        "correct": correct,
        "possible_label_issue": label_issue,
    }


def evaluate_case(case):
    print("\n" + "=" * 70)
    print(case["id"])
    print(case["question"])
    print("=" * 70)

    results = {}

    for mode in ["baseline", "self_rag"]:
        print(f"\nRunning {mode}...")
        result, wall_time = ask(case["question"], mode)

        answer = result.get("answer", "")
        refused = result.get("refused", False)

        evaluation = evaluate_answer(
            answer,
            case.get("reference_answer", ""),
            case.get("keywords", []),
            case["answerable"],
            refused,
        )

        results[mode] = {
            "answer": answer,
            "refused": refused,
            "attempts": result.get("attempts"),
            "llm_calls": result.get("llm_calls"),
            "latency_s": result.get("latency_s"),
            "wall_time_s": wall_time,
            "confidence": result.get("confidence"),
            "used_retrieval": result.get("used_retrieval"),
            **evaluation,
        }

        print("Answer:", answer)
        print("Refused:", refused)
        print("LLM calls:", result.get("llm_calls"))
        print("Latency:", result.get("latency_s"), "s")
        print("Keyword score:", evaluation["keyword_score"])
        print("Reference similarity:", evaluation["semantic_similarity"])
        print("Semantic pass:", evaluation["semantic_pass"])
        print("Correct:", evaluation["correct"])

        if evaluation["possible_label_issue"]:
            print("WARNING: possible dataset label issue")

    return {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "reference_answer": case.get("reference_answer", ""),
        "results": results,
    }


def summarize(results):
    print("\n\n" + "=" * 80)
    print("FINAL BENCHMARK")
    print("=" * 80)

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    for mode in ["baseline", "self_rag"]:
        ar = [r["results"][mode] for r in answerable]
        ur = [r["results"][mode] for r in unanswerable]
        allr = [r["results"][mode] for r in results]

        accuracy = sum(r["correct"] for r in ar) / len(ar)
        semantic_avg = sum(r["semantic_similarity"] for r in ar) / len(ar)
        kw_values = [r["keyword_score"] for r in ar if r["keyword_score"] is not None]
        kw_avg = sum(kw_values) / len(kw_values) if kw_values else 0.0
        refusal = sum(r["refused"] for r in ur) / len(ur)
        latency = sum(r["latency_s"] for r in allr) / len(allr)
        calls = sum(r["llm_calls"] for r in allr) / len(allr)

        issues = [
            r["id"] for r in results
            if r["results"][mode]["possible_label_issue"]
        ]

        print(f"\n{mode.upper()}")
        print(f"Answerable accuracy       : {accuracy:.2f}")
        print(f"Avg reference similarity  : {semantic_avg:.3f}")
        print(f"Avg keyword score         : {kw_avg:.2f}")
        print(f"Unanswerable refusal rate : {refusal:.2f}")
        print(f"Average latency           : {latency:.2f}s")
        print(f"Average LLM calls         : {calls:.2f}")

        if issues:
            print("Possible dataset label issues:", ", ".join(issues))


def main():
    with open(DATASET, "r") as f:
        dataset = json.load(f)

    print("Questions:", len(dataset))

    results = [evaluate_case(case) for case in dataset]

    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    summarize(results)
    print("\nResults saved to:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
