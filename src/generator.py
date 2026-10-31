
from .llm import LLM
from .prompts import DIRECT_SYSTEM, GENERATOR_SYSTEM
from .schemas import Chunk


def format_context(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, page {c.page})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def generate(llm: LLM, question: str, chunks: list[Chunk]) -> str:
    user = f"Context:\n{format_context(chunks)}\n\nQuestion: {question}"
    return llm.complete(GENERATOR_SYSTEM, user, max_tokens=700).strip()


def generate_direct(llm: LLM, question: str) -> str:
    return llm.complete(DIRECT_SYSTEM, question, max_tokens=200).strip()
