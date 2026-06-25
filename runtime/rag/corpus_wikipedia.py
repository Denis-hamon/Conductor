"""Wikipedia science corpus builder for GPQA Diamond RAG.

Downloads relevant Wikipedia articles based on GPQA Diamond topics
(physics, chemistry, biology, etc.) and builds a retrievable corpus.
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rag.corpus_wikipedia")

GPQA_TOPICS = [
    "Quantum mechanics", "Thermodynamics", "Electromagnetism", "Particle physics",
    "Organic chemistry", "Inorganic chemistry", "Biochemistry", "Molecular biology",
    "Cell biology", "Genetics", "Evolution", "Ecology",
    "Astrophysics", "Condensed matter physics", "Nuclear physics",
    "Chemical bonding", "Chemical reactions", "Periodic table",
    "DNA replication", "Protein synthesis", "Enzymes", "Metabolism",
    "Classical mechanics", "Special relativity", "General relativity",
    "Atomic physics", "Quantum chemistry", "Spectroscopy",
    "Neurobiology", "Immunology", "Microbiology",
    "Geology", "Oceanography", "Atmospheric science",
]


class WikipediaCorpusBuilder:
    def __init__(self, output_dir: str = "bench/data/corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, max_articles: int = 500, chunk_size: int = 512) -> list[dict]:
        try:
            import wikipediaapi
        except ImportError:
            logger.error("wikipedia-api not installed. Run: pip install wikipedia-api")
            return []

        user_agent = "ConductorFabric/1.0 (benchmark RAG corpus builder)"
        api = wikipediaapi.Wikipedia(
            language="en",
            user_agent=user_agent,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
        )

        documents = []
        seen_titles = set()

        for topic in GPQA_TOPICS:
            article = api.page(topic)
            if not article.exists():
                logger.warning("Wikipedia page not found: %s", topic)
                continue

            text = article.text
            if not text or len(text) < 100:
                continue

            title = article.title
            if title in seen_titles:
                continue
            seen_titles.add(title)

            chunks = self._chunk_text(text, chunk_size)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "chunk_id": f"wiki_{topic.lower().replace(' ', '_')}_{i}",
                    "content": chunk,
                    "source": "wikipedia",
                    "topic": topic,
                    "title": title,
                    "url": article.fullurl,
                })

            if len(documents) >= max_articles:
                break
            time.sleep(0.1)

        path = self.output_dir / "wikipedia_science.json"
        with open(path, "w") as f:
            json.dump(documents, f, indent=2)

        logger.info("Wikipedia corpus: %d articles, %d chunks → %s",
                     len(seen_titles), len(documents), path)
        return documents

    def _chunk_text(self, text: str, chunk_size: int = 512) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = []
        current_len = 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            words = para.split()
            para_len = len(words)
            if current_len + para_len > chunk_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += para_len
        if current:
            chunks.append(" ".join(current))
        return chunks or [text]

    def build_from_gpqa(self, gpqa_path: str, max_samples: int = 100) -> list[dict]:
        path = Path(gpqa_path)
        if not path.exists():
            logger.warning("GPQA file not found: %s", gpqa_path)
            return []

        with open(path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        documents = []
        for i, s in enumerate(samples[:max_samples]):
            question = s.get("question", "")
            answer = s.get("answer", "")
            choices = s.get("choices", [])
            content = f"Question: {question}\nAnswer: {answer}"
            for c in choices:
                content += f"\nChoice: {c}"

            documents.append({
                "chunk_id": f"gpqa_self_{i:05d}",
                "content": content,
                "source": "gpqa_self",
                "metadata": {"question": question, "answer": answer},
            })

        path_out = self.output_dir / "gpqa_self_corpus.json"
        with open(path_out, "w") as f:
            json.dump(documents, f, indent=2)
        logger.info("GPQA self-RAG corpus: %d documents → %s", len(documents), path_out)
        return documents
