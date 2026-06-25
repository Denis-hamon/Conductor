"""RAG backends — pluggable retrieval strategies."""

from .vector import VectorBackend
from .pageindex import PageIndexBackend

__all__ = ["VectorBackend", "PageIndexBackend"]
