"""Download a ready-made corpus for the project.

Examples:
    python scripts/download_data.py --arxiv-curated --wikipedia
    python scripts/download_data.py --arxiv-query "retrieval augmented generation" --max 15
"""
import argparse
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
HEADERS = {"User-Agent": "self-rag-student-project/1.0 (educational use)"}

# Well-known papers: LLM / RAG / transformers. IDs are arXiv identifiers.
CURATED_ARXIV = {
    "1706.03762": "attention_is_all_you_need",
    "2005.11401": "rag_lewis",
    "2310.11511": "self_rag",
    "2401.15884": "corrective_rag",
    "2312.10997": "rag_survey_gao",
    "2004.04906": "dense_passage_retrieval",
    "2212.10496": "hyde",
    "2309.15217": "ragas",
    "1810.04805": "bert",
    "2005.14165": "gpt3_few_shot",
    "2302.13971": "llama",
    "2307.09288": "llama2",
    "2201.11903": "chain_of_thought",
    "2210.03629": "react",
    "2203.02155": "instructgpt",
    "2106.09685": "lora",
}

WIKI_TOPICS = [
    "Retrieval-augmented generation", "Large language model", "Transformer (deep learning architecture)",
    "Word embedding", "Vector database", "Machine learning", "Gradient descent", "Backpropagation",
    "Overfitting", "Support vector machine", "Random forest", "Convolutional neural network",
    "Recurrent neural network", "Reinforcement learning", "Natural language processing",
]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


def download_pdf(arxiv_id: str, name: str) -> None:
    out = RAW / f"{name}.pdf"
    if out.exists():
        return
    r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=HEADERS, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    print(f"[arxiv] {name}")
    time.sleep(3)  # be polite to arXiv


def arxiv_search(query: str, max_results: int) -> list[tuple[str, str]]:
    url = "http://export.arxiv.org/api/query"
    r = requests.get(url, params={"search_query": f"all:{query}", "max_results": max_results}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        title = " ".join(e.find("a:title", ns).text.split())
        out.append((aid, slug(title)))
    return out


def download_wikipedia(title: str) -> None:
    out = RAW / f"wiki_{slug(title)}.txt"
    if out.exists():
        return
    r = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "format": "json", "redirects": 1},
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    pages = r.json()["query"]["pages"]
    text = next(iter(pages.values())).get("extract", "")
    if text:
        out.write_text(text, encoding="utf-8")
        print(f"[wiki] {title}")
    time.sleep(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arxiv-curated", action="store_true")
    ap.add_argument("--arxiv-query", type=str)
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--wikipedia", action="store_true")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    if args.arxiv_curated:
        for aid, name in CURATED_ARXIV.items():
            download_pdf(aid, name)
    if args.arxiv_query:
        for aid, name in arxiv_search(args.arxiv_query, args.max):
            download_pdf(aid, name)
    if args.wikipedia:
        for t in WIKI_TOPICS:
            download_wikipedia(t)
    print(f"Done. Files in {RAW}: {len(list(RAW.glob('*')))}")


if __name__ == "__main__":
    main()
