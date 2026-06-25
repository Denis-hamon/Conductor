"""Layer 4 (part 1): Dense embedding model.

Default: all-MiniLM-L6-v2 (CPU-friendly, 384-dim, fast on CPU).
Upgrade: BGE-M3 (dense + sparse + ColBERT in one model, 100+ langs, 8192 tokens).

Reference: HydraDB/Cortex uses dense embeddings internally
  (exposes /embeddings/* endpoints despite claiming to "kill vector databases").
"""

import logging
from typing import Optional

logger = logging.getLogger("rag.embeddings")


class EmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self.dimension = 384

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s on %s", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        self._load()
        embeddings = self._model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return embeddings.tolist()

    def encode_query(self, query: str) -> list[float]:
        return self.encode([query])[0]

    def encode_documents(self, documents: list[str], batch_size: int = 32) -> list[list[float]]:
        return self.encode(documents, batch_size=batch_size)
