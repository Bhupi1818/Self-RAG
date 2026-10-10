"""Vector retriever on top of ChromaDB (heavy deps are imported lazily)."""
from .config import Settings, settings
from .schemas import Chunk


class Retriever:
    def __init__(self, cfg: Settings = settings):
        self.cfg = cfg
        self._model = None
        self._col = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.cfg.embedding_model)
        return self._model

    @property
    def collection(self):
        if self._col is None:
            import chromadb

            client = chromadb.PersistentClient(path=self.cfg.chroma_dir)
            self._col = client.get_collection(self.cfg.collection_name)
        return self._col

    def count(self) -> int:
        return self.collection.count()

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        k = k or self.cfg.top_k
        emb = self.model.encode([query], normalize_embeddings=True)[0].tolist()
        res = self.collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunks.append(
                Chunk(id=cid, text=doc, source=meta["source"], page=meta.get("page", 0), score=1.0 - dist)
            )
        return chunks
