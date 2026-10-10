# Setup guide

## Requirements
Python 3.11+, ~2 GB disk (embedding model + index), an Anthropic API key **or** a local Ollama install.

## 1. Environment
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

## 2. Choose an LLM
**Anthropic (hosted):** set `ANTHROPIC_API_KEY` in `.env`. Default model is a small, cheap one; set `JUDGE_MODEL` to a
stronger model for evaluation.

**Ollama (free, local):**
```bash
ollama pull llama3.1:8b
```
then in `.env`: `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1:8b`. Small local models follow JSON instructions less
reliably, so expect more grader/verifier retries and consider `MAX_RETRIES=1`.

## 3. Get data and build the index
```bash
python scripts/download_data.py --arxiv-curated --wikipedia
python -m src.ingest --reset
```

## 4. Run
```bash
streamlit run app/streamlit_app.py     # http://localhost:8501
uvicorn api.main:app --reload          # http://localhost:8000/docs
```
Or `docker compose up --build`.

## 5. Test and lint
```bash
pytest -q
ruff check .
```
Tests use a fake LLM and fake retriever, so they need no API key or network.

## Configuration reference (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| LLM_PROVIDER | anthropic | `anthropic` or `ollama` |
| LLM_MODEL / JUDGE_MODEL | small model | answering model / evaluation judge |
| CHUNK_SIZE / CHUNK_OVERLAP | 250 / 40 | words per chunk / overlap |
| TOP_K | 8 | chunks retrieved per attempt |
| MAX_RETRIES | 2 | rewrite-and-retry budget |
| MIN_RELEVANT | 2 | relevant chunks required to attempt generation |
| SUPPORT_THRESHOLD | 0.8 | min supported-claim ratio to accept an answer |
| USE_ROUTER | true | skip retrieval for small talk |

## Troubleshooting
- `Collection ... does not exist`: run ingestion first.
- `No documents found`: run the download script or add files to `data/raw/`.
- `AuthenticationError`: check `ANTHROPIC_API_KEY`.
- Slow first run: the embedding model downloads once (~90 MB).
