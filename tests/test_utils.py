import pytest

from src.utils import chunk_words, clean_text, extract_json


def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced_with_chatter():
    raw = 'Sure! Here you go:\n```json\n{"needs_retrieval": true}\n```\nHope that helps.'
    assert extract_json(raw) == {"needs_retrieval": True}


def test_extract_json_nested_and_braces_in_strings():
    raw = 'prefix {"claims": [{"claim": "uses {x} syntax", "supported": true}]} suffix'
    assert extract_json(raw)["claims"][0]["supported"] is True


def test_extract_json_failure():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_clean_text_dehyphenates_and_collapses():
    assert clean_text("trans-\nformer   models\n\n\n\nrock") == "transformer models\n\nrock"


def test_chunk_words_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_words(text, chunk_size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert "w99" in chunks[-1].split()


def test_chunk_words_invalid_params():
    with pytest.raises(ValueError):
        chunk_words("a b c", chunk_size=5, overlap=5)
