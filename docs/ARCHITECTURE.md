# Architecture

## Components

| Component | File | Responsibility |
|---|---|---|
| Config | `src/config.py` | Env-driven settings (models, k, thresholds) |
| LLM wrapper | `src/llm.py` | Anthropic or Ollama calls, JSON parsing with retries, call counter |
| Ingestion | `src/ingest.py` | Load, clean, chunk, embed, upsert into ChromaDB |
| Retriever | `src/retriever.py` | Embed query, cosine search, return scored `Chunk`s |
| Router | `src/router.py` | Decide if retrieval is needed (fails safe to "yes") |
| Grader | `src/grader.py` | One batched LLM call grades all chunks (fail-open) |
| Generator | `src/generator.py` | Answer from relevant chunks only, with `[n]` citations |
| Verifier | `src/verifier.py` | Claim-level support check (fail-closed) |
| Rewriter | `src/rewriter.py` | Improve the retrieval query using the failure reason |
| Pipeline | `src/pipeline.py` | Orchestrates the loop; also the baseline path |
| API / UI | `api/main.py`, `app/streamlit_app.py` | Serving and demo |
| Eval | `eval/*` | Test-set generation and benchmark |

## Data flow

```
data/raw/*.pdf|txt|md ──ingest──► chunks + embeddings ──► ChromaDB (chroma_db/)
                                                              ▲
question ─► router ─► retriever ─► grader ─► generator ─► verifier ─► answer
                         ▲                         │            │
                         └──────── rewriter ◄──────┴────────────┘  (max MAX_RETRIES)
```

## Design principles
- **Dependency injection**: `SelfRAGPipeline(llm, retriever)` makes the loop unit-testable with fakes.
- **Lazy heavy imports**: embeddings/Chroma/Anthropic load on first use so tests and linting are instant.
- **Every step leaves a trace**: the API returns what was retrieved, rejected, rewritten and why.
- **Bounded loops**: `MAX_RETRIES` caps cost; exhaustion yields an explicit refusal, never a guess.

## Tech choices
See [DECISIONS.md](DECISIONS.md).
