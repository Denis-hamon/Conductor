"""Retrieval router — selects backend based on query context.

- If query targets known document IDs → PageIndex (intra-doc reasoning)
- If query is cross-corpus → VectorBackend (hybrid similarity search)
- Fallback: always VectorBackend
"""

import logging
from typing import Optional

logger = logging.getLogger("rag.router")


class RetrievalRouter:
    def __init__(self, vector_backend, pageindex_backend=None):
        self.vector = vector_backend
        self.pageindex = pageindex_backend

    def query(self, query: str, doc_ids: Optional[list[str]] = None,
              top_k: int = 10, prefer: str = "auto") -> list[dict]:
        if prefer == "pageindex" and self.pageindex:
            return self.pageindex.query(query, doc_ids=doc_ids, top_k=top_k)

        if prefer == "vector" or not self.pageindex:
            return self.vector.query(query, top_k=top_k)

        if doc_ids and len(doc_ids) <= 3 and self.pageindex:
            results = self.pageindex.query(query, doc_ids=doc_ids, top_k=top_k)
            if results:
                return results

        return self.vector.query(query, top_k=top_k)
