import re

from .llm import LLM
from .prompts import GRADER_SYSTEM
from .schemas import Chunk, GradeResult


def fallback_grade(question, chunks):
    """Conservative keyword fallback if LLM grading fails."""
    stopwords = {
        "what", "is", "the", "of", "a", "an", "in", "on",
        "for", "to", "and", "or", "how", "why", "does",
        "do", "can", "could", "would", "should", "explain",
        "describe", "tell", "me", "about", "are", "was",
        "were", "be", "it", "as", "by", "with", "from",
        "this", "that", "current"
    }

    terms = {
        word for word in re.findall(r"[a-z0-9]+", question.lower())
        if len(word) > 2 and word not in stopwords
    }

    results = []

    for chunk in chunks:
        passage_terms = set(
            re.findall(r"[a-z0-9]+", chunk.text.lower())
        )
        overlap = terms & passage_terms

        # Require at least two matching informative terms.
        relevant = len(overlap) >= 2

        reason = (
            f"Fallback keyword overlap: {', '.join(sorted(overlap))}"
            if relevant
            else "Insufficient keyword overlap; conservatively rejected"
        )

        results.append(GradeResult(relevant, reason))

    return results



def grade_chunks(
    llm: LLM,
    question: str,
    chunks: list[Chunk],
) -> list[GradeResult]:
    """Grade all chunks in one LLM call; fail closed on invalid grades."""
    if not chunks:
        return []

    passages = "\n\n".join(
        f"[{i}] {chunk.text[:1200]}"
        for i, chunk in enumerate(chunks, start=1)
    )
    user = (
        f"Question: {question}\n"
        f"Number of passages: {len(chunks)}\n"
        f"Return exactly {len(chunks)} grade objects, with IDs 1 through {len(chunks)}. "
        "Do not create grades for any other IDs.\n\n"
        f"Passages:\n{passages}"
    )
    out = None

    try:
        out = llm.complete_json(
            GRADER_SYSTEM,
            user,
            max_tokens=400,
        )

        if not isinstance(out, dict):
            raise ValueError("Expected a JSON object")

        grades = out.get("grades")
        if not isinstance(grades, list):
            raise ValueError("Missing or invalid 'grades' list")

        by_id = {}
        expected_ids = set(range(1, len(chunks) + 1))

        for grade in grades:
            if not isinstance(grade, dict):
                raise ValueError("Grade entry is not an object")

            grade_id = grade.get("id")
            if (
                not isinstance(grade_id, int)
                or isinstance(grade_id, bool)
                or grade_id not in expected_ids
            ):
                raise ValueError(f"Invalid grade ID: {grade_id!r}")

            if grade_id in by_id:
                raise ValueError(f"Duplicate grade ID: {grade_id}")

            if not isinstance(grade.get("relevant"), bool):
                raise ValueError(
                    f"Invalid 'relevant' value for ID {grade_id}"
                )

            by_id[grade_id] = grade

        if set(by_id) != expected_ids:
            missing = sorted(expected_ids - set(by_id))
            raise ValueError(f"Incomplete grade IDs: {missing}")


    except Exception as exc:
        print(f"[GRADER DEBUG] {type(exc).__name__}: {exc}")
        print(f"[GRADER DEBUG] Parsed output: {out!r}")

        return fallback_grade(question, chunks)



    return [
        GradeResult(
            by_id[i]["relevant"],
            str(by_id[i].get("reason", "")),
        )
        for i in range(1, len(chunks) + 1)
    ]
