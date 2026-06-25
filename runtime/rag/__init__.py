"""Conductor Fabric — Advanced RAG Pipeline.

Architecture (inspired by HydraDB/Cortex reverse engineering):

  Two composable backends:
    VectorBackend   — hybrid dense+sparse + cross-encoder reranking (scale)
    PageIndexBackend — reasoning-based tree navigation over documents (precision)

  Layers (VectorBackend):
    1. Chunking:   Semchunk
    2. Knowledge:  In-memory index / Graphiti temporal graph (optional)
    3. Filtering:  Metadata pre-retrieval
    4. Retrieval:  Dense (vector) + Sparse (BM25) + Graph traversal
    5. Reranking:  Cross-encoder (FlashRank CPU / BGE-reranker GPU)
    6. Feedback:   Thompson Sampling for self-improvement

  PageIndex (optional, for single-document deep retrieval):
    - Hierarchical tree index from document structure
    - LLM-guided tree search (reasoning-based, not vector similarity)
    - 98.7% on FinanceBench
"""

from .pipeline import RAGPipeline
from .retriever import HybridRetriever, RetrievalResult
from .reranker import Reranker
from .embeddings import EmbeddingModel
from .router import RetrievalRouter
from .backends import VectorBackend, PageIndexBackend

__all__ = [
    "RAGPipeline", "HybridRetriever", "RetrievalResult", "Reranker",
    "EmbeddingModel", "RetrievalRouter", "VectorBackend", "PageIndexBackend",
]
