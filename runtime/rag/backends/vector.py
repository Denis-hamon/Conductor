"""Vector RAG backend — hybrid dense + sparse retrieval with cross-encoder reranking.

Original RAG pipeline (layers 1-5) refactored into a pluggable backend.
Good for cross-document, multi-source retrieval at scale.
"""

import logging
from typing import Optional

from ..chunker import Chunker
from ..embeddings import EmbeddingModel
from ..retriever import HybridRetriever
from ..reranker import Reranker

logger = logging.getLogger("rag.vector")


class VectorBackend:
    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "flashrank",
        chunk_size: int = 512,
        search_alpha: float = 0.7,
        top_k: int = 20,
        rerank_top_k: int = 10,
        device: str = "cpu",
    ):
        self.chunker = Chunker(chunk_size=chunk_size)
        self.embed = EmbeddingModel(model_name=embed_model, device=device)
        self.retriever = HybridRetriever(self.embed, alpha=search_alpha, top_k=top_k)
        self.reranker = Reranker(model_name=reranker_model, device=device)
        self.rerank_top_k = rerank_top_k
        self._documents: list[dict] = []

    def ingest(self, documents: list[dict]):
        chunked = []
        for doc in documents:
            text = doc.get("content", doc.get("text", ""))
            meta = {k: v for k, v in doc.items() if k not in ("content", "text")}
            chunks = self.chunker.chunk(text, metadata=meta)
            chunked.extend(chunks)
        self._documents = chunked
        self.retriever.index(chunked)
        logger.info("Ingested %d docs → %d chunks", len(documents), len(chunked))

    def query(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        retrieved = self.retriever.search(query, top_k=top_k)
        results = [r.__dict__ for r in retrieved]
        reranked = self.reranker.rerank(query, results, top_k=self.rerank_top_k)
        return reranked
