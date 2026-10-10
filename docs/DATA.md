# Data

## Recommended corpus (domain: LLMs / RAG / machine learning)
Chosen because you can judge answer quality yourself, and the sources are free and legal to download.

| Source | What | How |
|---|---|---|
| **arXiv** | 16 curated papers (Attention, RAG, Self-RAG, CRAG, DPR, HyDE, RAGAS, BERT, GPT-3, LLaMA 1/2, CoT, ReAct, InstructGPT, LoRA) | `python scripts/download_data.py --arxiv-curated` |
| **arXiv search** | Any extra topic | `--arxiv-query "retrieval augmented generation" --max 15` |
| **Wikipedia** | 15 ML/NLP overview articles | `--wikipedia` (MediaWiki API) |
| **Your own notes** | DSA / Python / ML study notes (`.md`, `.txt`, `.pdf`) | Drop files into `data/raw/` |

Target: 30-100 documents (a few thousand chunks). More is not better for a one-week project.

## Alternative corpora
- **Library docs**: scikit-learn user guide, PyTorch docs, FastAPI docs (export/crawl to text or PDF).
- **Public-sector PDFs**: RBI, SEBI, NITI Aayog, or government scheme documents (good India-specific demo).
- **Project Gutenberg**: classic books as plain text (easy but less technical).
- **Hugging Face datasets** (for benchmarking rather than a demo corpus):
  - SQuAD 2.0 (contains *unanswerable* questions, ideal for refusal testing)
  - HotpotQA (multi-hop questions)
  - Natural Questions, MS MARCO (retrieval benchmarks)
  - RAGBench (RAG-specific evaluation sets)

## Licensing
arXiv papers are free to read and process for personal/educational use; do **not** commit the PDFs to GitHub (they are
git-ignored). Wikipedia text is CC BY-SA. Commit only the download script.

## Processing
1. PDFs are read page by page with `pypdf` (page number kept as metadata); TXT/MD are one unit.
2. `clean_text`: remove null bytes, repair hyphenated line breaks, collapse whitespace.
3. `chunk_words`: 250-word windows, 40-word overlap (~330 tokens). Pages under 30 words are skipped.
4. Chunk id = `<file>-p<page>-c<index>`; metadata = `{source, page}`.

## Known data issues
- PDF extraction breaks tables, equations and two-column layouts; answers about tables will be weaker.
- References/bibliography pages add noise; the grader usually filters them out.
