"""Full RAG pipeline — coordinates all retrieval backends into a single flow.

Two backends, composable:
  - VectorBackend: hybrid dense+sparse retrieval, cross-document, for scale
  - PageIndexBackend: reasoning-based tree navigation, intra-document, for precision

Usage:
    pipeline = RAGPipeline()
    pipeline.ingest(documents)
    results = pipeline.query(query)
    answer = pipeline.answer(query)
"""

import logging
from typing import Optional

from .backends import VectorBackend, PageIndexBackend
from .router import RetrievalRouter

logger = logging.getLogger("rag.pipeline")


class RAGPipeline:
    def __init__(
        self,
        embed_model: str = "all-MiniLM-L6-v2",
        reranker_model: str = "flashrank",
        chunk_size: int = 512,
        search_alpha: float = 0.7,
        top_k: int = 20,
        rerank_top_k: int = 10,
        device: str = "cpu",
        use_pageindex: bool = False,
        pageindex_workspace: Optional[str] = None,
        pageindex_model: str = "gpt-4o",
    ):
        self.vector = VectorBackend(
            embed_model=embed_model,
            reranker_model=reranker_model,
            chunk_size=chunk_size,
            search_alpha=search_alpha,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            device=device,
        )
        self.pageindex = None
        if use_pageindex:
            try:
                self.pageindex = PageIndexBackend(
                    workspace=pageindex_workspace,
                    model=pageindex_model,
                )
            except Exception as e:
                logger.warning("PageIndex not available (%s)", e)

        self.router = RetrievalRouter(self.vector, self.pageindex)
        self.rerank_top_k = rerank_top_k

    def ingest(self, documents: list[dict], backend: str = "auto"):
        if backend in ("auto", "vector"):
            self.vector.ingest(documents)
        if backend in ("auto", "pageindex") and self.pageindex:
            self.pageindex.ingest(documents)

    def ingest_texts(self, texts: list[str], metadatas: Optional[list[dict]] = None):
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            docs.append({"content": text, **meta})
        self.ingest(docs)

    def query(self, query: str, top_k: Optional[int] = None,
              doc_ids: Optional[list[str]] = None, prefer: str = "auto") -> list[dict]:
        return self.router.query(query, doc_ids=doc_ids, top_k=top_k or self.rerank_top_k, prefer=prefer)

    def answer(self, query: str, llm_complete=None, top_k: int = 5,
               doc_ids: Optional[list[str]] = None) -> dict:
        context_docs = self.query(query, top_k=top_k, doc_ids=doc_ids)
        if not context_docs:
            return {"answer": "No relevant context found.", "context": [], "citations": []}

        context_text = "\n\n".join(
            f"[{i + 1}] {d['content']}"
            for i, d in enumerate(context_docs)
        )
        sources = [d.get("chunk_id", d.get("doc_id", str(i))) for i, d in enumerate(context_docs)]

        prompt = (
            "Answer the question based ONLY on the provided context. "
            "Cite sources using [source: <id>]. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        if llm_complete is None:
            answer = f"[Simulated answer based on {len(context_docs)} sources]"
        else:
            answer = llm_complete([{"role": "user", "content": prompt}])

        return {
            "answer": answer,
            "context": context_docs,
            "citations": sources,
            "prompt": prompt,
        }
