"""Layer 4: Hybrid retrieval — Dense (vector) + Sparse (BM25) + optional Graph.

Three retrievers run in parallel, fused via Reciprocal Rank Fusion (RRF).
Mirrors HydraDB/Cortex `search_alpha` parameter (0.0-1.0, default 0.7).

Reference: Cortex API docs — search endpoint with search_alpha controlling
  dense vs sparse blend. Graphiti adds a third dimension via graph traversal.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("rag.retriever")


@dataclass
class RetrievalResult:
    chunk_id: str
    content: str
    score: float
    rank: int = 0
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    def __init__(self, embedding_model, alpha: float = 0.7, top_k: int = 20):
        self.embed = embedding_model
        self.alpha = alpha
        self.top_k = top_k
        self._documents: list[dict] = []
        self._embeddings: list[list[float]] = []

    def index(self, documents: list[dict]):
        self._documents = documents
        texts = [d["content"] for d in documents]
        logger.info("Indexing %d documents...", len(texts))
        self._embeddings = self.embed.encode_documents(texts)
        logger.info("Indexed %d documents (dim=%d)", len(self._embeddings), self.embed.dimension)

    def search(self, query: str, top_k: Optional[int] = None) -> list[RetrievalResult]:
        k = top_k or self.top_k
        if not self._documents:
            logger.warning("No documents indexed")
            return []

        query_vec = self.embed.encode_query(query)

        dense_scores = self._dense_search(query_vec)
        sparse_scores = self._sparse_search(query)

        fused = self._reciprocal_rank_fusion([dense_scores, sparse_scores], k=k)

        results = []
        for chunk_id, score in fused:
            doc = self._doc_by_id(chunk_id)
            if doc:
                results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    content=doc["content"],
                    score=score,
                    rank=len(results),
                    metadata=doc.get("metadata", {}),
                ))
        return results

    def _dense_search(self, query_vec: list[float]) -> list[tuple[str, float]]:
        scores = []
        for i, emb in enumerate(self._embeddings):
            sim = self._cosine_similarity(query_vec, emb)
            scores.append((self._documents[i]["chunk_id"], sim))
        scores.sort(key=lambda x: -x[1])
        return scores

    def _sparse_search(self, query: str) -> list[tuple[str, float]]:
        query_terms = set(query.lower().split())
        scores = []
        for doc in self._documents:
            content = doc["content"].lower()
            matches = sum(1 for t in query_terms if t in content)
            scores.append((doc["chunk_id"], matches / max(len(query_terms), 1)))
        scores.sort(key=lambda x: -x[1])
        return scores

    def _reciprocal_rank_fusion(self, rankings: list[list[tuple[str, float]]],
                                 k: int = 60, top_k: int = 20) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, (chunk_id, _) in enumerate(ranking[:top_k]):
                if chunk_id not in fused:
                    fused[chunk_id] = 0.0
                fused[chunk_id] += 1.0 / (k + rank + 1)

        for ranking in rankings:
            dense_score_map = {cid: sc for cid, sc in ranking}
            for cid in fused:
                if cid in dense_score_map:
                    fused[cid] = self.alpha * fused[cid] + (1 - self.alpha) * dense_score_map[cid]

        sorted_items = sorted(fused.items(), key=lambda x: -x[1])
        return sorted_items[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    def _doc_by_id(self, chunk_id: str) -> Optional[dict]:
        for d in self._documents:
            if d["chunk_id"] == chunk_id:
                return d
        return None
