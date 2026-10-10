# API reference

Base URL: `http://localhost:8000` · Interactive docs: `/docs` (Swagger)

## GET /health
```json
{"status": "ok", "chunks": 4210}
```
`status: "degraded"` with a `detail` message if the index is missing (run `python -m src.ingest --reset`).

## POST /ask
Request:
```json
{"question": "How does Self-RAG decide when to retrieve?", "mode": "self_rag"}
```
| Field | Type | Notes |
|---|---|---|
| question | string | 3-1000 chars |
| mode | `"self_rag"` \| `"baseline"` | default `self_rag` |

Response (abridged):
```json
{
  "answer": "Self-RAG trains the model to emit reflection tokens that decide ... [1][2]",
  "mode": "self_rag",
  "confidence": 1.0,
  "attempts": 1,
  "refused": false,
  "used_retrieval": true,
  "llm_calls": 4,
  "latency_s": 6.8,
  "sources": [{"id": "self_rag-p3-c1", "source": "self_rag", "page": 3, "score": 0.71, "snippet": "..."}],
  "trace": [
    {"step": "route", "needs_retrieval": true},
    {"step": "retrieve", "attempt": 0, "query": "...", "n": 8, "top_score": 0.71},
    {"step": "grade", "attempt": 0, "relevant": 5, "rejected": [{"id": "...", "reason": "..."}]},
    {"step": "verify", "attempt": 0, "supported_ratio": 1.0, "answers_question": true, "unsupported_claims": []}
  ]
}
```
Refusal example: `"refused": true`, `"confidence": 0.0`, `"sources": []`.

Errors: `500` with `detail` (missing index, missing API key, provider failure).
