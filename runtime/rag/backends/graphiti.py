"""Graphiti backend — temporal knowledge graph retrieval.

Uses Graphiti (Zep, Apache 2.0) for temporal-aware entity extraction
and graph-based retrieval. Adds time-weighted relevance: recent information
is naturally prioritized via the episodic graph structure.

Requires: Neo4j running at the configured URI.
          graphiti-core >= 0.29

Example:
    backend = GraphitiBackend(uri="bolt://localhost:7687", user="neo4j", password="password")
    backend.ingest(documents)
    results = backend.query("Who created Python?")
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("rag.graphiti")


class GraphitiBackend:
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "neo4j",
        llm_api_key: Optional[str] = None,
        embed_model: str = "all-MiniLM-L6-v2",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.llm_api_key = llm_api_key
        self.embed_model = embed_model
        self._graphiti = None
        self._group_id = "default"

    @property
    def graphiti(self):
        if self._graphiti is None:
            from graphiti_core import Graphiti
            from graphiti_core.llm_client import OpenAIClient
            from graphiti_core.embedder import OpenAIClient as EmbedderClient

            llm = OpenAIClient(api_key=self.llm_api_key) if self.llm_api_key else None

            self._graphiti = Graphiti(
                uri=self.uri,
                user=self.user,
                password=self.password,
                llm_client=llm,
            )
            self._graphiti.build_indices_and_constraints()
            logger.info("Graphiti initialized (uri=%s)", self.uri)
        return self._graphiti

    def ingest(self, documents: list[dict]):
        g = self.graphiti
        for doc in documents:
            content = doc.get("content", doc.get("text", ""))
            name = doc.get("chunk_id", doc.get("doc_name", "episode"))
            source = doc.get("source", "document")
            try:
                g.add_episode(
                    name=name,
                    episode_body=content,
                    source_description=source,
                    reference_time=datetime.now(timezone.utc),
                    group_id=self._group_id,
                )
                logger.debug("Ingested episode: %s", name)
            except Exception as e:
                logger.warning("Graphiti ingest failed for %s: %s", name, e)

        logger.info("Graphiti ingested %d documents", len(documents))

    def query(self, query: str, top_k: int = 10) -> list[dict]:
        g = self.graphiti
        try:
            results = g.search(query, group_ids=[self._group_id], num_results=top_k)
        except Exception as e:
            logger.warning("Graphiti search failed: %s", e)
            return []

        if not results or not results.edges:
            return []

        seen = set()
        output = []
        for edge in results.edges:
            content = None
            if hasattr(edge, "fact") and edge.fact:
                content = edge.fact
            elif hasattr(edge, "name") and edge.name:
                content = edge.name
            if content and content not in seen:
                seen.add(content)
                output.append({
                    "content": content,
                    "score": edge.score if hasattr(edge, "score") else 0.0,
                    "source": "graphiti",
                    "metadata": {
                        "source_node": str(edge.source_node_uuid) if hasattr(edge, "source_node_uuid") else "",
                        "target_node": str(edge.target_node_uuid) if hasattr(edge, "target_node_uuid") else "",
                    },
                })
        return output[:top_k]

    def close(self):
        if self._graphiti:
            try:
                self._graphiti.close()
            except Exception:
                pass
