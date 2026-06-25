"""Corpus builder for RAG benchmarks.

Creates and manages retrieval corpora:
  1. Synthetic: Auto-generated documents+queries for integration testing
  2. Wikipedia: Download + chunk science articles (arXiv/Wikipedia)
  3. Self-RAG: GPQA Diamond Q&A pairs as documents (quickstart)
"""

import json
import logging
import random
import textwrap
from pathlib import Path
from typing import Optional

logger = logging.getLogger("bench.corpus")


class CorpusBuilder:
    def __init__(self, data_dir: str = "data/corpus"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def build_synthetic(self, num_docs: int = 100, num_queries: int = 50) -> tuple[list[dict], list[dict]]:
        """Build a synthetic corpus with known facts + queries."""
        documents = []
        topics = [
            ("Python", "Python is a high-level, general-purpose programming language created by Guido van Rossum in 1991."),
            ("PostgreSQL", "PostgreSQL is a free and open-source relational database management system emphasizing extensibility and SQL compliance."),
            ("Transformers", "The Transformer is a deep learning architecture introduced in 2017, based solely on attention mechanisms."),
            ("Gradient Descent", "Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function."),
            ("HTTP", "HTTP is the foundation of data communication on the World Wide Web, defining how messages are formatted and transmitted."),
        ]
        for i in range(num_docs):
            topic, fact = random.choice(topics)
            doc_id = f"synth_{i:05d}"
            content = (
                f"Document {doc_id} about {topic}. "
                f"{fact} "
                f"This is paragraph {i} covering various aspects of the topic. "
                f"Additional details and context are provided here for retrieval purposes."
            )
            documents.append({"chunk_id": doc_id, "content": content, "source": "synthetic", "topic": topic})

        topic_docs = {}
        for d in documents:
            t = d.get("topic", "")
            if t and t not in topic_docs:
                topic_docs[t] = d["chunk_id"]

        queries = []
        for topic, expected in [("Python", None), ("PostgreSQL", None), ("Transformers", None)]:
            if topic in topic_docs:
                queries.append({"query": f"Who created {topic}?", "expected_chunk": topic_docs[topic]})
            else:
                queries.append({"query": f"Tell me about {topic}", "expected_chunk": topic_docs.get(topic)})

        for i in range(num_queries - len(queries)):
            doc = random.choice(documents)
            queries.append({"query": f"Tell me about {doc['topic']}", "expected_chunk": doc["chunk_id"]})

        with open(self.data_dir / "synthetic_docs.json", "w") as f:
            json.dump(documents, f)
        with open(self.data_dir / "synthetic_queries.json", "w") as f:
            json.dump(queries, f)

        logger.info("Built synthetic corpus: %d docs, %d queries", num_docs, num_queries)
        return documents, queries

    def build_gpqa_corpus(self, gpqa_path: str) -> list[dict]:
        """Build a retrieval corpus from GPQA Diamond Q&A pairs (self-RAG)."""
        path = Path(gpqa_path)
        if not path.exists():
            logger.warning("GPQA file not found: %s", gpqa_path)
            return self._build_gpqa_fallback()

        with open(path) as f:
            questions = json.load(f)

        documents = []
        for i, q in enumerate(questions):
            question = q.get("question", q.get("query", ""))
            correct = q.get("correct_answer", q.get("answer", ""))
            distractors = q.get("distractors", [])
            explanation = q.get("explanation", "")

            content = (
                f"Scientific question: {question}\n"
                f"Correct answer: {correct}\n"
            )
            if explanation:
                content += f"Explanation: {explanation}\n"
            for d in distractors:
                content += f"Distractor: {d}\n"

            documents.append({
                "chunk_id": f"gpqa_{i:05d}",
                "content": content,
                "source": "gpqa_diamond",
                "metadata": {"question": question, "answer": correct,
                              "domain": q.get("domain", "science")},
            })

        with open(self.data_dir / "gpqa_corpus.json", "w") as f:
            json.dump(documents, f)
        logger.info("Built GPQA corpus: %d docs", len(documents))
        return documents

    def _build_gpqa_fallback(self) -> list[dict]:
        return self.build_synthetic(num_docs=50, num_queries=20)[0]

    def load(self, name: str = "synthetic") -> list[dict]:
        path = self.data_dir / f"{name}_docs.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []

    def load_queries(self, name: str = "synthetic") -> list[dict]:
        path = self.data_dir / f"{name}_queries.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return []
