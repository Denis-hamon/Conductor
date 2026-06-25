"""PageIndex backend — reasoning-based retrieval over document tree index.

Uses PageIndex: a hierarchical tree index built from document structure.
Instead of vector similarity, the LLM *reasons* over the tree to find relevant
sections — mirroring how a human expert navigates a long document.

Best for: single-document QA on long, structured documents (contracts, reports).
Not for: cross-document retrieval at scale.

Reference: https://github.com/VectifyAI/PageIndex
  - 98.7% on FinanceBench (financial document QA)
  - No vector DB, no chunking
  - Agentic tree search for context-aware retrieval
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rag.pageindex")


class PageIndexBackend:
    def __init__(
        self,
        workspace: Optional[str] = None,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
    ):
        self.workspace = workspace or str(tempfile.mkdtemp(prefix="pageindex_"))
        self.model = model
        self.api_key = api_key
        self._client = None
        self._documents: dict[str, str] = {}  # doc_name -> doc_id

    @property
    def client(self):
        if self._client is None:
            try:
                from pageindex import PageIndexClient
                self._client = PageIndexClient(
                    api_key=self.api_key,
                    model=self.model,
                    workspace=self.workspace,
                )
                logger.info("PageIndexClient initialized (workspace=%s)", self.workspace)
            except ImportError:
                raise ImportError(
                    "PageIndex not installed. Run: pip install git+https://github.com/VectifyAI/PageIndex.git"
                )
        return self._client

    def ingest(self, documents: list[dict]):
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            doc_name = doc.get("doc_name", doc.get("chunk_id", "doc"))
            path = Path(self.workspace) / f"{doc_name}.md"
            path.write_text(content)
            doc_id = self.client.index(str(path), mode="md")
            self._documents[doc_name] = doc_id
            logger.info("Indexed '%s' → doc_id=%s", doc_name, doc_id)

    def query(self, query: str, doc_ids: Optional[list[str]] = None, top_k: int = 5) -> list[dict]:
        if not self._documents:
            logger.warning("No documents indexed")
            return []

        targets = doc_ids or list(self._documents.values())
        results = []

        for doc_id in targets:
            try:
                structure_json = self.client.get_document_structure(doc_id)
                structure = json.loads(structure_json)
                if "error" in structure:
                    logger.warning("Structure error for %s: %s", doc_id, structure["error"])
                    continue

                doc_json = self.client.get_document(doc_id)
                doc_info = json.loads(doc_json)

                section_pages = self._find_relevant_sections(query, structure, doc_info, top_k)
                for pages in section_pages:
                    content_json = self.client.get_page_content(doc_id, pages)
                    content = json.loads(content_json)
                    if isinstance(content, list):
                        for item in content:
                            results.append({
                                "content": item.get("content", ""),
                                "page": item.get("page", 0),
                                "doc_id": doc_id,
                                "source": "pageindex",
                            })
            except Exception as e:
                logger.warning("PageIndex query failed for %s: %s", doc_id, e)

        return results[:top_k]

    def _find_relevant_sections(self, query: str, structure: list, doc_info: dict, top_k: int) -> list[str]:
        try:
            from ._tree_searcher import search_tree
            return search_tree(query, structure, doc_info, top_k=top_k)
        except ImportError:
            logger.warning("Tree searcher not available, returning structure overview")
            return self._fallback_search(query, structure, top_k)

    def _fallback_search(self, query: str, structure: list, top_k: int) -> list[str]:
        pages = []
        for node in structure:
            if "page_range" in node:
                pages.append(str(node["page_range"]))
            elif "start_index" in node and "end_index" in node:
                pages.append(f"{node['start_index']}-{node['end_index']}")
            if len(pages) >= top_k:
                break
        return pages

    def ingest_text(self, text: str, doc_name: str = "doc"):
        self.ingest([{"content": text, "doc_name": doc_name}])
