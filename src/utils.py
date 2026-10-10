"""Small pure helpers (no heavy imports) so they are easy to unit test."""
import json
import re


def extract_json(text: str) -> dict:
    """Parse the first JSON object found in an LLM response.

    Tolerates ```json fences and surrounding chatter.
    """
    if not text or not text.strip():
        raise ValueError("empty LLM response")
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    raise ValueError(f"no valid JSON object found in: {text[:120]!r}")


def clean_text(text: str) -> str:
    """Normalise whitespace and repair PDF line-break hyphenation."""
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_words(text: str, chunk_size: int = 250, overlap: int = 40) -> list[str]:
    """Split text into overlapping word windows."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")
    words = text.split()
    step = chunk_size - overlap
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_size]
        if start > 0 and len(piece) <= overlap:  # tail made only of overlap -> redundant
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
    return chunks
