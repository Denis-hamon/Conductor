"""Layer 5: Cross-encoder reranking.

10-100x more accurate than bi-encoder cosine similarity, as the model sees
query + document together rather than as separate vectors.

Default: FlashRank (CPU, ~50ms/doc, Apache 2.0).
Upgrade: BGE-reranker-v2-m3 via rerankers[transformers] (GPU, ~80ms/doc).

Reference: HydraDB/Cortex reranking layer with recency_bias parameter.
  - BGE-reranker-v2-m3: SOTA open-source cross-encoder, 100+ languages
  - FlashRank: On-device, no GPU needed, good enough for dev
"""

import logging
from typing import Optional

logger = logging.getLogger("rag.reranker")


class Reranker:
    def __init__(self, model_name: str = "flashrank", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._ranker = None

    def _load(self):
        if self._ranker is not None:
            return
        from rerankers import Reranker as RRLib
        kwargs = {"model_name": self.model_name}
        if self.device == "cpu" and "flashrank" in self.model_name.lower():
            pass
        elif self.device == "cuda":
            kwargs["model_type"] = "cross-encoder"
        logger.info("Loading reranker: %s on %s", self.model_name, self.device)
        self._ranker = RRLib(**kwargs)

    def rerank(self, query: str, documents: list[dict],
               top_k: Optional[int] = None) -> list[dict]:
        if not documents:
            return documents

        self._load()
        doc_ids = [d.get("chunk_id", str(i)) for i, d in enumerate(documents)]
        doc_texts = [d["content"] for d in documents]

        try:
            results = self._ranker.rank(query=query, docs=doc_texts, doc_ids=doc_ids)
            ranked = results.top_k(k=top_k or len(documents))
            reranked = []
            for r in ranked:
                for d in documents:
                    if d.get("chunk_id") == r.doc_id:
                        doc = dict(d)
                        doc["rerank_score"] = float(r.score)
                        reranked.append(doc)
                        break
            return reranked
        except Exception as e:
            logger.warning("Reranking failed (%s), returning original order", e)
            return documents
