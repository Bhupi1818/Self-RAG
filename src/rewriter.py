from .llm import LLM
from .prompts import REWRITER_SYSTEM


def rewrite(llm: LLM, question: str, previous_query: str, reason: str) -> str:
    """Produce an improved retrieval query. Falls back to the original question."""
    user = (
        f"Original question: {question}\n"
        f"Previous query: {previous_query}\n"
        "Improve the query for retrieving passages that answer the original question. "
        "Preserve its technical meaning and terminology. "
        "Do not include internal pipeline errors or failure descriptions."
    )
    try:
        out = llm.complete_json(REWRITER_SYSTEM, user, max_tokens=120)
        q = str(out.get("query", "")).strip()
        return q or question
    except ValueError:
        return question
