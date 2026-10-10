"""Central configuration. All values come from environment variables (.env supported)."""
import os
from dataclasses import dataclass

try:  # python-dotenv is optional at import time so tests run without it
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "anthropic")
    llm_model: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")
    judge_model: str = os.getenv("JUDGE_MODEL", os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"))
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")



    # Retrieval
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_dir: str = os.getenv("CHROMA_DIR", "chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "self_rag_docs")
    raw_dir: str = os.getenv("RAW_DIR", "data/raw")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "250"))  # in words
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "40"))  # in words
    top_k: int = int(os.getenv("TOP_K", "3"))

    # Self-reflection loop
    max_retries: int = int(os.getenv("MAX_RETRIES", "2"))
    min_relevant: int = int(os.getenv("MIN_RELEVANT", "2"))
    support_threshold: float = float(os.getenv("SUPPORT_THRESHOLD", "0.8"))
    use_router: bool = _bool("USE_ROUTER", True)


settings = Settings()
