"""Ingestion: load PDFs/TXT/MD -> clean -> chunk -> embed -> store in ChromaDB.

Usage:
    python -m src.ingest --reset
"""
import argparse
from pathlib import Path

from .config import settings
from .utils import chunk_words, clean_text


def load_documents(raw_dir: str):
    """Yield (source_name, page_number, text) for every supported file."""
    for path in sorted(Path(raw_dir).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            try:
                reader = PdfReader(str(path))
                for i, page in enumerate(reader.pages, start=1):
                    yield path.stem, i, page.extract_text() or ""
            except Exception as e:  # corrupt PDF: skip, keep going
                print(f"[skip] {path.name}: {e}")
        elif suffix in {".txt", ".md"}:
            yield path.stem, 0, path.read_text(encoding="utf-8", errors="ignore")


def build_chunks(raw_dir: str, chunk_size: int, overlap: int):
    ids, texts, metas = [], [], []
    for source, page, text in load_documents(raw_dir):
        text = clean_text(text)
        if len(text.split()) < 30:  # skip near-empty pages
            continue
        for j, piece in enumerate(chunk_words(text, chunk_size, overlap)):
            ids.append(f"{source}-p{page}-c{j}")
            texts.append(piece)
            metas.append({"source": source, "page": page})
    return ids, texts, metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true", help="drop the existing collection first")
    args = ap.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    ids, texts, metas = build_chunks(settings.raw_dir, settings.chunk_size, settings.chunk_overlap)
    if not texts:
        raise SystemExit(f"No documents found in {settings.raw_dir}. Run scripts/download_data.py first.")
    print(f"Prepared {len(texts)} chunks. Embedding with {settings.embedding_model} ...")

    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    client = chromadb.PersistentClient(path=settings.chroma_dir)
    if args.reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass
    col = client.get_or_create_collection(settings.collection_name, metadata={"hnsw:space": "cosine"})

    batch = 500
    for i in range(0, len(texts), batch):
        col.upsert(
            ids=ids[i : i + batch],
            documents=texts[i : i + batch],
            metadatas=metas[i : i + batch],
            embeddings=embeddings[i : i + batch].tolist(),
        )
    print(f"Done. Collection '{settings.collection_name}' now has {col.count()} chunks.")


if __name__ == "__main__":
    main()
