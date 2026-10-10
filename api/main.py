from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline import build_pipeline

app = FastAPI(title="Self-Reflective RAG", version="1.0.0")


@lru_cache(maxsize=1)
def get_pipeline():
    return build_pipeline()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    mode: Literal["self_rag", "baseline"] = "self_rag"


@app.get("/health")
def health():
    try:
        return {"status": "ok", "chunks": get_pipeline().retriever.count()}
    except Exception as e:
        return {"status": "degraded", "detail": str(e)}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        pipe = get_pipeline()
        result = pipe.answer(req.question) if req.mode == "self_rag" else pipe.baseline(req.question)
        return result.to_dict()
    except Exception as e:  # surface config problems (missing index / API key) clearly
        raise HTTPException(status_code=500, detail=str(e))
