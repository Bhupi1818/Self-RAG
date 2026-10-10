
from .generator import format_context
from .llm import LLM
from .prompts import VERIFIER_SYSTEM
from .schemas import Chunk, VerifyResult


def verify(
    llm: LLM,
    question: str,
    answer: str,
    chunks: list[Chunk],
) -> VerifyResult:
    """Check groundedness and whether the answer addresses the question."""

    user = (
        f"Question: {question}\n\n"
        f"Context:\n{format_context(chunks)}\n\n"
        f"Draft answer:\n{answer}"
    )

    try:
        out = llm.complete_json(
            VERIFIER_SYSTEM,
            user,
            max_tokens=900,
        )

        claims = out.get("claims")
        # Normalize a single-claim response only when its fields are valid.
        if claims is None and "claim" in out and "supported" in out:
            if (
                isinstance(out["claim"], str)
                and out["claim"].strip()
                and isinstance(out["supported"], bool)
                and isinstance(out.get("answers_question"), bool)
            ):
                claims = [
                    {
                        "claim": out["claim"],
                        "supported": out["supported"],
                    }
                ]

        if (
            not isinstance(claims, list)
            or not claims
            or not all(
                isinstance(c, dict)
                and isinstance(c.get("claim"), str)
                and c["claim"].strip()
                and isinstance(c.get("supported"), bool)
                for c in claims
            )
            or not isinstance(out.get("answers_question"), bool)
        ):
            return VerifyResult(0.0, False, ["verifier_failed"])

        supported = [
            c for c in claims
            if isinstance(c, dict) and c.get("supported") is True
        ]
        unsupported = [
            str(c.get("claim", ""))
            for c in claims
            if not isinstance(c, dict) or c.get("supported") is not True
        ]

        ratio = len(supported) / len(claims)

        return VerifyResult(
            ratio,
            bool(out.get("answers_question", False)),
            unsupported,
        )

    # except (ValueError, TypeError, AttributeError, KeyError):
    #     return VerifyResult(0.0, False, ["verifier_failed"])
    except (ValueError, TypeError, AttributeError, KeyError):
        return VerifyResult(0.0, False, ["verifier_failed"])
