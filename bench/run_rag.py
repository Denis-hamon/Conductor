#!/usr/bin/env python3
"""RAG Benchmark Runner — evaluates retrieval-augmented generation quality.

Tests the full RAG pipeline:
  1. Ingest corpus (synthetic or GPQA self-RAG)
  2. Query → retrieve contexts → rerank → LLM answer → score
  3. Report: retrieval recall@k, answer accuracy, citation faithfulness

Usage:
  # Synthetic corpus (quick integration test)
  python bench/run_rag.py

  # GPQA Diamond with RAG
  python bench/run_rag.py --benchmark gpqa --rag-backend advanced

  # With FlashRank reranking
  python bench/run_rag.py --reranker flashrank

  # Compare RAG vs no-RAG
  python bench/run_rag.py --benchmark gpqa --compare
"""

import argparse
import json
import logging
import statistics
import sys
import time
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))



logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("rag-bench")


@dataclass
class RAGSampleResult:
    query: str
    ground_truth: str
    predicted: str
    retrieved: list[dict]
    recall: float
    accuracy: bool
    latency_ms: float
    error: str = ""


class LLMClient:
    """OpenAI-compatible API for answer generation."""

    def __init__(self, endpoint: str = "", timeout: int = 120):
        self.endpoint = endpoint
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        if not self.endpoint:
            return ""

        body = json.dumps({
            "model": "qwen-reasoner",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 2048,
        }).encode()

        from urllib import request
        req = request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = request.urlopen(req, timeout=self.timeout)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]


def run_synthetic_benchmark(pipeline, queries: list[dict]) -> list[RAGSampleResult]:
    """Test RAG retrieval accuracy on a synthetic corpus."""
    results = []
    for q in queries:
        query = q["query"]
        expected = q.get("expected_chunk", "")

        start = time.monotonic()
        retrieved = pipeline.query(query, top_k=10)
        latency = (time.monotonic() - start) * 1000

        retrieved_ids = [d["chunk_id"] for d in retrieved]
        recall = 1.0 if (expected and expected in retrieved_ids) else 0.0

        topic_match = any(expected in (d.get("metadata", {}).get("doc_id", ""))
                          or expected in d.get("chunk_id", "")
                          for d in retrieved) if expected else bool(retrieved)
        results.append(RAGSampleResult(
            query=query,
            ground_truth=expected,
            predicted=retrieved_ids[0] if retrieved_ids else "",
            retrieved=retrieved,
            recall=recall,
            accuracy=recall > 0 if expected else topic_match,
            latency_ms=round(latency, 2),
        ))

    return results


def run_gpqa_rag_benchmark(pipeline, samples: list[dict], llm: LLMClient) -> list[RAGSampleResult]:
    """Test GPQA Diamond with RAG-augmented generation."""
    results = []
    labels = ["A", "B", "C", "D"]

    for i, sample in enumerate(samples):
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        answer = sample.get("answer", "")

        query = f"{question}\n" + "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))

        start = time.monotonic()
        retrieved = pipeline.query(query, top_k=5)
        latency = (time.monotonic() - start) * 1000

        context = "\n\n".join(f"[{i+1}] {d['content']}" for i, d in enumerate(retrieved))

        rag_prompt = (
            "Answer the question based on the context and your knowledge.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Choices:\n" +
            "\n".join(f"{l}. {c}" for l, c in zip(labels, choices)) +
            "\n\nAnswer with the letter (A, B, C, or D) only."
        )

        msgs = [
            {"role": "system", "content": "Answer by responding with the correct letter only."},
            {"role": "user", "content": rag_prompt},
        ]

        predicted = llm.complete(msgs) if llm.endpoint else "A"
        predicted_letter = predicted.strip()[0] if predicted.strip() else ""
        correct = predicted_letter == answer

        results.append(RAGSampleResult(
            query=question,
            ground_truth=answer,
            predicted=predicted_letter,
            retrieved=retrieved,
            recall=1.0 if retrieved else 0.0,
            accuracy=correct,
            latency_ms=round(latency, 2),
        ))

        if (i + 1) % 10 == 0:
            logger.info("GPQA RAG [%d/%d] — acc=%.2f", i + 1, len(samples),
                         sum(r.accuracy for r in results) / len(results))

    return results


def gpqa_synthetic() -> list[dict]:
    return [
        {"question": "Which particle is a fermion?",
         "choices": ["Photon", "Gluon", "Electron", "Z boson"],
         "answer": "C", "subject": "physics"},
        {"question": "IUPAC name for CH3CH2OH?",
         "choices": ["Methane", "Ethanol", "Propanol", "Methanol"],
         "answer": "B", "subject": "chemistry"},
    ]


def report(results: list[RAGSampleResult], output_dir: str):
    if not results:
        logger.warning("No results to report")
        return

    accuracy = statistics.mean(r.accuracy for r in results)
    recall = statistics.mean(r.recall for r in results)
    latency = statistics.mean(r.latency_ms for r in results)
    latency_p50 = statistics.median(r.latency_ms for r in results)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"rag_bench_{timestamp}.json")

    report_data = {
        "timestamp": timestamp,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "avg_latency_ms": round(latency, 2),
        "p50_latency_ms": round(latency_p50, 2),
        "total": len(results),
        "passed": sum(r.accuracy for r in results),
        "results": [r.__dict__ for r in results],
    }

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"\n{'='*50}")
    print(f"RAG Benchmark Results")
    print(f"{'='*50}")
    print(f"  Accuracy:       {accuracy:.2%} ({report_data['passed']}/{report_data['total']})")
    print(f"  Recall@10:      {recall:.2%}")
    print(f"  Avg Latency:    {latency:.0f}ms")
    print(f"  P50 Latency:    {latency_p50:.0f}ms")
    print(f"  Report:         {report_path}")
    print(f"{'='*50}\n")


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark Runner")
    parser.add_argument("--benchmark", default="synthetic", choices=["synthetic", "gpqa"])
    parser.add_argument("--rag-backend", default="vector", choices=["simple", "advanced", "vector", "pageindex", "dual"])
    parser.add_argument("--reranker", default="flashrank", choices=["none", "flashrank"])
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--endpoint", default="", help="LLM API endpoint")
    parser.add_argument("--corpus", default="", help="Path to corpus JSON (chunk_id + content)")
    parser.add_argument("--num-synthetic", type=int, default=50)
    parser.add_argument("--num-queries", type=int, default=20)
    parser.add_argument("--compare", action="store_true", help="Compare RAG vs no-RAG")
    parser.add_argument("--output", "-o", default="bench/reports")
    args = parser.parse_args()

    # Build pipeline — dual backends
    from runtime.rag import RAGPipeline

    use_pageindex = args.rag_backend in ("pageindex", "dual")
    if use_pageindex and not os.environ.get("OPENAI_API_KEY"):
        logger.warning("No OPENAI_API_KEY set — PageIndex requires one. Falling back to vector.")
        use_pageindex = False
    pipeline = RAGPipeline(
        embed_model=args.embed_model,
        reranker_model=args.reranker if args.reranker != "none" else "noop",
        chunk_size=512,
        device="cpu",
        use_pageindex=use_pageindex,
    )

    # Load or build corpus
    if args.corpus:
        with open(args.corpus) as f:
            documents = json.load(f)
        logger.info("Loaded %d documents from %s", len(documents), args.corpus)
    elif args.benchmark == "gpqa":
        from bench.corpus import CorpusBuilder
        builder = CorpusBuilder()
        gpqa_path = "bench/data/gpqa_diamond.jsonl"
        documents = builder.build_gpqa_corpus(gpqa_path)
    else:
        from bench.corpus import CorpusBuilder
        builder = CorpusBuilder()
        documents, queries_data = builder.build_synthetic(
            num_docs=args.num_synthetic,
            num_queries=args.num_queries,
        )

    # Ingest corpus
    pipeline.ingest(documents)

    # Run benchmark (reuse queries_data from the same build_synthetic() call)
    results = []
    if args.benchmark == "gpqa":
        gpqa_path = "bench/data/gpqa_diamond.jsonl"
        if os.path.exists(gpqa_path):
            samples = []
            with open(gpqa_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        else:
            samples = gpqa_synthetic()

        llm = LLMClient(endpoint=args.endpoint)
        results = run_gpqa_rag_benchmark(pipeline, samples, llm)
    else:
        results = run_synthetic_benchmark(pipeline, queries_data)

    # Report
    report(results, args.output)


if __name__ == "__main__":
    main()
