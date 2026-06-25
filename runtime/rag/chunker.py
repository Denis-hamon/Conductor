"""Layer 1: Intelligent document chunking.

Uses Semchunk (fast, deterministic) by default, with simple word-based
fallback when tokenizer is unavailable.

Reference: HydraDB/Cortex ingestion layer reverse engineering.
  - Semchunk: 3.04s for Gutenberg Corpus (512-token chunks), 85% faster
  - Adaptive chunking: 87% accuracy vs 13% for fixed-size (clinical domain)
"""

import logging
from typing import Optional

logger = logging.getLogger("rag.chunker")


def _make_token_counter():
    """Create a token counter compatible with semchunk.

    Tries tiktoken first, then falls back to a word-count approximation.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text))
    except Exception:
        pass
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_pretrained("gpt2")
        return lambda text: len(tok.encode(text).ids)
    except Exception:
        pass
    return lambda text: len(text.split())


class Chunker:
    def __init__(self, chunk_size: int = 512, overlap: int = 0):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._token_counter = None

    @property
    def token_counter(self):
        if self._token_counter is None:
            self._token_counter = _make_token_counter()
        return self._token_counter

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[dict]:
        try:
            import semchunk
            chunks = semchunk.chunk(
                text,
                chunk_size=self.chunk_size,
                token_counter=self.token_counter,
                overlap=self.overlap if self.overlap > 0 else None,
            )
        except Exception as e:
            logger.warning("Semchunk failed (%s), falling back to simple split", e)
            chunks = self._simple_split(text)

        base_id = (metadata or {}).get("chunk_id", "")
        result = []
        for i, content in enumerate(chunks):
            cid = f"{base_id}_chunk_{i}" if len(chunks) > 1 else (base_id or f"chunk_{i}")
            result.append({
                "content": content,
                "chunk_id": cid,
                "index": i,
                "metadata": {**(metadata or {}), "doc_id": base_id},
            })
        logger.info("Chunked %d chars into %d chunks", len(text), len(result))
        return result

    def _simple_split(self, text: str) -> list[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), self.chunk_size):
            chunk = " ".join(words[i:i + self.chunk_size])
            chunks.append(chunk)
        return chunks
