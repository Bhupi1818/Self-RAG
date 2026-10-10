from .llm import LLM
from .prompts import ROUTER_SYSTEM


def route(llm: LLM, question: str) -> bool:
    """Return True if the question needs retrieval. Fails safe to True."""
    try:
        out = llm.complete_json(ROUTER_SYSTEM, f"Question: {question}", max_tokens=120)
        return bool(out.get("needs_retrieval", True))
    except ValueError:
        return True
