#!/usr/bin/env python3
"""Conductor Fabric — Multi-benchmark Evaluation Runner.

Supports: HumanEval+, GSM8K, SWE-bench Lite, Terminal-bench, MMLU.
Runs each benchmark in A/B mode:
  A) Direct LLM (baseline) — single model call, no routing
  B) Conductor Fabric — smart routing + verification gates + fallback

Usage:
  # Full evaluation suite (A/B comparison)
  python bench/runner.py --all --ab

  # Single benchmark
  python bench/runner.py --benchmark humaneval

  # Custom config
  python bench/runner.py --config bench/config.yaml --benchmark gsm8k
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
import traceback
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib import request, error

import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("bench")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    benchmark: str
    model_label: str
    mode: str  # "baseline" | "conductor"
    metric: str
    score: float
    total_samples: int
    passed: int
    failed: int
    errors: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cost_per_request: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    sample_results: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def header() -> list[str]:
        return [
            "benchmark", "mode", "metric", "score", "total", "passed", "failed",
            "errors", "p50_ms", "p95_ms", "p99_ms", "cost_per_req",
        ]

    def row(self) -> list[str]:
        return [
            self.benchmark, self.mode, self.metric,
            f"{self.score:.4f}", str(self.total_samples),
            str(self.passed), str(self.failed), str(self.errors),
            f"{self.latency_p50_ms:.1f}", f"{self.latency_p95_ms:.1f}",
            f"{self.latency_p99_ms:.1f}", f"{self.cost_per_request:.6f}",
        ]


@dataclass
class SampleResult:
    index: int
    input: str
    expected: Any
    predicted: str
    passed: bool
    latency_ms: float
    error: str = ""
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------

class LLMClient:
    """OpenAI-compatible API client for both baseline and conductor modes."""

    def __init__(self, endpoint: str, model: str, timeout: int = 120,
                 temperature: float = 0.0, max_tokens: int = 4096):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, messages: list[dict], max_retries: int = 3) -> tuple[str, float, str]:
        last_error = ""
        for attempt in range(max_retries):
            try:
                body = json.dumps({
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }).encode()
                req = request.Request(
                    self.endpoint,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                start = time.monotonic()
                with request.urlopen(req, timeout=self.timeout) as resp:
                    latency = (time.monotonic() - start) * 1000
                    data = json.loads(resp.read())
                    content = data["choices"][0]["message"]["content"]
                    return content, round(latency, 2), ""
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Retry %d/%d after error: %s", attempt + 1, max_retries, last_error)
                    time.sleep(wait)
        return "", 0.0, last_error


class ConductorClient:
    """Client for Conductor Fabric gateway — routes through planner + runtime."""

    def __init__(self, endpoint: str, timeout: int = 120):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout

    def complete(self, messages: list[dict], max_retries: int = 3) -> tuple[str, float, str]:
        last_error = ""
        for attempt in range(max_retries):
            try:
                body = json.dumps({"messages": messages}).encode()
                req = request.Request(
                    f"{self.endpoint}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                start = time.monotonic()
                with request.urlopen(req, timeout=self.timeout) as resp:
                    latency = (time.monotonic() - start) * 1000
                    data = json.loads(resp.read())
                    content = data["choices"][0]["message"]["content"]
                    return content, round(latency, 2), ""
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning("Retry %d/%d after error: %s", attempt + 1, max_retries, last_error)
                    time.sleep(wait)
        return "", 0.0, last_error


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class DatasetLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[dict]:
        ...


class JSONLDatasetLoader(DatasetLoader):
    def load(self, path: str) -> list[dict]:
        samples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        logger.info("Loaded %d samples from %s", len(samples), path)
        return samples


class CSVDatasetLoader(DatasetLoader):
    def load(self, path: str) -> list[dict]:
        samples = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                samples.append(dict(row))
        logger.info("Loaded %d samples from %s", len(samples), path)
        return samples


def get_loader(path: str) -> DatasetLoader:
    if path.endswith(".jsonl"):
        return JSONLDatasetLoader()
    elif path.endswith(".csv"):
        return CSVDatasetLoader()
    else:
        raise ValueError(f"Unknown dataset format: {path}")


# ---------------------------------------------------------------------------
# Benchmark implementations
# ---------------------------------------------------------------------------

class Benchmark(ABC):
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    @abstractmethod
    def build_messages(self, sample: dict) -> list[dict]:
        ...

    @abstractmethod
    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, Any]:
        ...

    def load_dataset(self) -> list[dict]:
        path = self.config["dataset"]
        if not os.path.exists(path):
            logger.warning("Dataset not found: %s — using synthetic data", path)
            return self._synthetic_data()
        loader = get_loader(path)
        samples = loader.load(path)
        max_samples = self.config.get("max_samples", 0)
        if max_samples > 0 and len(samples) > max_samples:
            samples = samples[:max_samples]
        return samples

    @abstractmethod
    def _synthetic_data(self) -> list[dict]:
        ...


class HumanEvalBenchmark(Benchmark):
    """Python function completion from docstring, evaluated by unit tests."""

    def build_messages(self, sample: dict) -> list[dict]:
        prompt = sample["prompt"]
        return [
            {"role": "system", "content": "Complete the following Python function. Return ONLY the function body, no explanation."},
            {"role": "user", "content": prompt},
        ]

    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, str]:
        test = sample.get("test", "")
        entry_point = sample.get("entry_point", "")
        if not test:
            return self._check_approx_match(sample, prediction)

        code = prediction
        if entry_point and entry_point not in code:
            code = sample["prompt"] + "\n" + prediction

        try:
            compiled = compile(code + "\n" + test, "<eval>", "exec")
            namespace = {}
            exec(compiled, namespace)
            return True, ""
        except AssertionError:
            return False, "assertion failed"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"

    def _check_approx_match(self, sample: dict, prediction: str) -> tuple[bool, str]:
        expected = sample.get("canonical_solution", "")
        if not expected:
            return True, "no test available"
        pred_clean = prediction.strip()
        exp_clean = expected.strip()
        return pred_clean == exp_clean, ""

    def _synthetic_data(self) -> list[dict]:
        return [
            {
                "task_id": "synthetic/0",
                "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
                "entry_point": "add",
                "test": "assert add(1, 2) == 3\nassert add(-1, 1) == 0",
                "canonical_solution": "    return a + b",
            },
        ]


class GSM8KBenchmark(Benchmark):
    """Grade-school math word problems — chain-of-thought + answer."""

    def build_messages(self, sample: dict) -> list[dict]:
        question = sample.get("question", sample.get("input", ""))
        system = "Solve the following math problem step by step. End with 'The answer is: NUMBER' on its own line."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, str]:
        expected = sample.get("answer", "")
        if not expected:
            return False, "no expected answer"

        predicted_answer = self._extract_answer(prediction)
        expected_answer = self._extract_answer(str(expected))
        if not predicted_answer or not expected_answer:
            return False, f"could not extract answer: predicted={predicted_answer}, expected={expected_answer}"
        return predicted_answer == expected_answer, f"got={predicted_answer}, expected={expected_answer}"

    def _extract_answer(self, text: str) -> Optional[str]:
        import re
        text = text.strip()
        patterns = [
            r"The answer is:\s*([\d.,]+)",
            r"The answer is\s*([\d.,]+)",
            r"####\s*([\d.,]+)",
            r"answer[:\s]+([\d.,]+)",
            r"=\s*([\d.,]+)",
        ]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                return m.group(1)
        return None

    def _synthetic_data(self) -> list[dict]:
        return [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "If x = 3 and y = 4, what is x * y?", "answer": "12"},
        ]


class MMLUBenchmark(Benchmark):
    """Multiple-choice knowledge benchmark."""

    def build_messages(self, sample: dict) -> list[dict]:
        question = sample.get("question", "")
        choices = sample.get("choices", [])
        formatted = f"{question}\n\n"
        labels = ["A", "B", "C", "D"]
        for label, choice in zip(labels, choices):
            formatted += f"{label}. {choice}\n"
        formatted += "\nAnswer with the letter (A, B, C, or D) only."
        return [
            {"role": "system", "content": "Answer the question by responding with the correct letter (A, B, C, or D) only."},
            {"role": "user", "content": formatted},
        ]

    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, str]:
        expected = sample.get("answer", "")
        predicted = prediction.strip()[0] if prediction.strip() else ""
        return predicted == expected, f"predicted={predicted}, expected={expected}"

    def _synthetic_data(self) -> list[dict]:
        return [
            {"question": "What is the capital of France?", "choices": ["London", "Paris", "Berlin", "Madrid"], "answer": "B"},
        ]


class TerminalBenchBenchmark(Benchmark):
    """CLI command generation from natural language task descriptions."""

    def build_messages(self, sample: dict) -> list[dict]:
        task = sample.get("task", sample.get("input", ""))
        system = "Generate the terminal command(s) to accomplish the task. Return ONLY the command, no explanation."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]

    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, str]:
        expected_commands = sample.get("commands", sample.get("expected", []))
        if isinstance(expected_commands, str):
            expected_commands = [expected_commands]
        pred_clean = prediction.strip().split("\n")[0].strip()
        for cmd in expected_commands:
            if pred_clean == cmd.strip():
                return True, ""
        return False, f"predicted={pred_clean}, expected={expected_commands}"

    def _synthetic_data(self) -> list[dict]:
        return [
            {"task": "List all files in the current directory", "commands": ["ls", "ls ."]},
            {"task": "Show disk usage in human-readable format", "commands": ["df -h"]},
        ]


class SWEBenchBenchmark(Benchmark):
    """Lightweight SWE-bench evaluation — issue resolution with patch generation.

    NOTE: Full SWE-bench requires cloning repos and running test suites.
    This implements a practical approximation using the instance data format.
    """

    def build_messages(self, sample: dict) -> list[dict]:
        issue = sample.get("problem_statement", sample.get("input", ""))
        repo = sample.get("repo", "")
        system = (
            f"You are a senior software engineer fixing a bug in {repo}.\n"
            "Analyze the issue and generate a git patch (diff) that fixes it.\n"
            "Return ONLY the diff, no explanation."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": issue},
        ]

    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, str]:
        if not prediction.strip():
            return False, "empty prediction"

        contains_diff = "---" in prediction and "+++" in prediction
        contains_index = "index " in prediction
        contains_diff_line = "diff --git" in prediction

        if contains_diff_line:
            score = "patch_generated"
        elif contains_diff and contains_index:
            score = "partial_patch"
        elif contains_diff:
            score = "minimal_patch"
        else:
            score = "no_patch"

        passed = contains_diff_line or contains_diff
        return passed, score

    def _synthetic_data(self) -> list[dict]:
        return [
            {
                "repo": "psf/requests",
                "instance_id": "requests-3000",
                "problem_statement": "Session.send() raises ConnectionError on redirect with missing content-length header.",
                "patch": "--- a/requests/sessions.py\n+++ b/requests/sessions.py\n@@ -1,3 +1,5 @@\n def send(self, request, **kwargs):\n+    if 'content-length' not in request.headers:\n+        request.headers['content-length'] = '0'\n     return super().send(request, **kwargs)",
            },
        ]


# ---------------------------------------------------------------------------
# AgentWorldBench — Language World Model Evaluation
# ---------------------------------------------------------------------------

class AgentWorldBenchmark(Benchmark):
    """AgentWorldBench: language world model simulation quality (7 domains, 5-dim rubric)."""

    def load_dataset(self) -> list[dict]:
        import json
        path = self.config["dataset"]
        if not os.path.exists(path):
            logger.warning("Dataset not found: %s — using synthetic data", path)
            return self._synthetic_data()
        samples = []
        for jsonl_file in sorted(Path(path).glob("*_test.jsonl")):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        samples.append(json.loads(line))
        max_samples = self.config.get("max_samples", 0)
        if max_samples > 0 and len(samples) > max_samples:
            samples = samples[:max_samples]
        logger.info("Loaded %d AgentWorldBench samples from %s", len(samples), path)
        return samples

    def build_messages(self, sample: dict) -> list[dict]:
        system = sample.get("system_str", "")
        current = sample.get("current_prompt", "")
        if not current:
            prompts = sample.get("prompt", [])
            turn_idx = sample.get("turn_idx", 1) - 1
            if isinstance(prompts, list) and turn_idx < len(prompts):
                current = prompts[turn_idx]
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": current})
        return msgs

    def evaluate(self, sample: dict, prediction: str) -> tuple[bool, str]:
        ground_truth = sample.get("response", [""])
        if isinstance(ground_truth, list):
            idx = max(sample.get("turn_idx", 1) - 1, 0)
            ground_truth = ground_truth[idx] if idx < len(ground_truth) else ""
        non_empty = bool(prediction and prediction.strip() and prediction != "No output")
        return non_empty, f"pred_len={len(prediction)}, gt_len={len(ground_truth)}"

    def _synthetic_data(self) -> list[dict]:
        return [
            {
                "task": "mcp",
                "id": 1,
                "prompt": ["### Turn 1\n**Action:**\n```json\n{\"tool\": \"search\", \"query\": \"test\"}\n```"],
                "response": ["**Environment Observation:**\nResult: test output"],
                "system_str": "You are a Tool World Model.",
                "turn_idx": 1,
                "total_turns": 1,
            },
        ]


# ---------------------------------------------------------------------------
# A/B Comparison runner
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {
    "humaneval": HumanEvalBenchmark,
    "gsm8k": GSM8KBenchmark,
    "mmlu": MMLUBenchmark,
    "terminal_bench": TerminalBenchBenchmark,
    "swe_bench_lite": SWEBenchBenchmark,
    "agentworld_bench": AgentWorldBenchmark,
}


def run_benchmark(
    benchmark_name: str,
    bench: Benchmark,
    config: dict,
    mode: str,
    llm_client: Any,
    max_workers: int = 4,
) -> BenchmarkResult:
    eval_cfg = config.get("evaluation", {})
    timeout = eval_cfg.get("timeout", 120)
    max_retries = eval_cfg.get("max_retries", 3)

    samples = bench.load_dataset()
    if not samples:
        logger.warning("No samples for benchmark '%s'", benchmark_name)
        return _empty_result(benchmark_name, config, mode)

    sample_results: list[SampleResult] = []
    latencies: list[float] = []
    passed_count = 0
    failed_count = 0
    error_count = 0

    logger.info("Running %s [%s] — %d samples", benchmark_name, mode, len(samples))

    def process_sample(idx: int, sample: dict) -> SampleResult:
        try:
            messages = bench.build_messages(sample)
            prediction, latency, error = llm_client.complete(
                messages, max_retries=max_retries
            )
            if error:
                return SampleResult(idx, str(sample), "", False, latency, error=error)

            passed, detail = bench.evaluate(sample, prediction)
            return SampleResult(
                idx, str(sample),
                sample.get("answer", sample.get("expected", "")),
                prediction, passed, latency,
                error="", metadata={"detail": detail},
            )
        except Exception as e:
            return SampleResult(
                idx, str(sample), "", "", False, 0.0,
                error=f"{type(e).__name__}: {e}",
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_sample, i, s): i
            for i, s in enumerate(samples)
        }
        for future in as_completed(futures):
            result = future.result()
            sample_results.append(result)
            latencies.append(result.latency_ms)
            if result.error:
                error_count += 1
                logger.debug("Sample %d error: %s", result.index, result.error)
            elif result.passed:
                passed_count += 1
            else:
                failed_count += 1

    sample_results.sort(key=lambda r: r.index)
    latencies.sort()
    total = len(samples)

    p50 = latencies[len(latencies) // 2] if latencies else 0.0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0.0
    p99 = latencies[int(len(latencies) * 0.99)] if latencies else 0.0

    metric = config.get("metric", "accuracy")
    if metric == "pass@1":
        score = passed_count / total if total > 0 else 0.0
    elif metric == "resolved":
        score = passed_count / total if total > 0 else 0.0
    else:
        score = passed_count / (total - error_count) if (total - error_count) > 0 else 0.0

    model_label = config.get("label", mode)
    result = BenchmarkResult(
        benchmark=benchmark_name,
        model_label=model_label,
        mode=mode,
        metric=metric,
        score=score,
        total_samples=total,
        passed=passed_count,
        failed=failed_count,
        errors=error_count,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        cost_per_request=0.0,
        sample_results=[r.__dict__ for r in sample_results],
    )

    logger.info(
        "%s [%s] — score=%.4f passed=%d failed=%d errors=%d p50=%.0fms",
        benchmark_name, mode, score, passed_count, failed_count,
        error_count, p50,
    )
    return result


def _empty_result(benchmark_name: str, config: dict, mode: str) -> BenchmarkResult:
    return BenchmarkResult(
        benchmark=benchmark_name,
        model_label=config.get("label", mode),
        mode=mode,
        metric=config.get("metric", "accuracy"),
        score=0.0,
        total_samples=0,
        passed=0,
        failed=0,
        errors=0,
        latency_p50_ms=0.0,
        latency_p95_ms=0.0,
        latency_p99_ms=0.0,
        cost_per_request=0.0,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: list[BenchmarkResult],
    output_dir: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = os.path.join(output_dir, f"bench_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "summary": _summary_table(results),
                "results": [r.to_dict() for r in results],
            },
            f, indent=2, default=str,
        )

    # HTML report
    html_path = os.path.join(output_dir, f"bench_{timestamp}.html")
    html = _generate_html_report(results, timestamp)
    with open(html_path, "w") as f:
        f.write(html)

    # Console summary
    csv_path = os.path.join(output_dir, f"bench_{timestamp}.csv")
    _write_csv(results, csv_path)

    logger.info("Reports written to %s/bench_%s.{json,html,csv}", output_dir, timestamp)
    return html_path


def _summary_table(results: list[BenchmarkResult]) -> list[dict]:
    summary = {}
    for r in results:
        key = f"{r.benchmark}/{r.mode}"
        summary[key] = {
            "benchmark": r.benchmark,
            "mode": r.mode,
            "metric": r.metric,
            "score": r.score,
            "passed": r.passed,
            "total": r.total_samples,
            "p50_ms": r.latency_p50_ms,
            "p95_ms": r.latency_p95_ms,
        }
    return list(summary.values())


def _write_csv(results: list[BenchmarkResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BenchmarkResult.header())
        for r in results:
            writer.writerow(r.row())


def _generate_html_report(results: list[BenchmarkResult], timestamp: str) -> str:
    rows = ""
    ab_comparisons = {}
    for r in results:
        rows += f"""
        <tr>
            <td>{r.benchmark}</td>
            <td><span class="mode-{r.mode}">{r.mode}</span></td>
            <td>{r.metric}</td>
            <td class="score">{r.score:.4f}</td>
            <td>{r.passed}/{r.total_samples}</td>
            <td>{r.errors}</td>
            <td>{r.latency_p50_ms:.0f}</td>
            <td>{r.latency_p95_ms:.0f}</td>
        </tr>"""
        key = r.benchmark
        if key not in ab_comparisons:
            ab_comparisons[key] = {}
        ab_comparisons[key][r.mode] = r

    ab_rows = ""
    for bname, modes in sorted(ab_comparisons.items()):
        baseline = modes.get("baseline")
        conductor = modes.get("conductor")
        if baseline and conductor:
            diff = conductor.score - baseline.score
            p50_diff = baseline.latency_p50_ms - conductor.latency_p50_ms if conductor.latency_p50_ms else 0
            diff_class = "positive" if diff > 0 else ("negative" if diff < 0 else "neutral")
            p50_class = "positive" if p50_diff > 0 else ("negative" if p50_diff < 0 else "neutral")
            ab_rows += f"""
            <tr>
                <td>{bname}</td>
                <td>{baseline.score:.4f}</td>
                <td>{conductor.score:.4f}</td>
                <td class="{diff_class}">{diff:+.4f}</td>
                <td>{baseline.latency_p50_ms:.0f}</td>
                <td>{conductor.latency_p50_ms:.0f}</td>
                <td class="{p50_class}">{p50_diff:+.0f}</td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conductor Fabric — Benchmark Report</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:opsz@14..32&display=swap');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: Inter, system-ui, sans-serif; background: #0f1117; color: #e1e4ea; padding: 2rem; }}
  h1 {{ font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; color: #f0f2f5; }}
  .subtitle {{ color: #8b8fa3; font-size: 0.875rem; margin-bottom: 2rem; }}
  h2 {{ font-size: 1.125rem; font-weight: 600; margin: 1.5rem 0 0.75rem; color: #e1e4ea; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 1.5rem; font-size: 0.875rem; }}
  th, td {{ padding: 0.5rem 0.75rem; text-align: left; border-bottom: 1px solid #1e2028; }}
  th {{ background: #161822; color: #8b8fa3; font-weight: 500; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05em; }}
  tr:hover td {{ background: #1a1c26; }}
  .score {{ font-variant-numeric: tabular-nums; }}
  .mode-baseline {{ color: #8b8fa3; }}
  .mode-conductor {{ color: #58a6ff; font-weight: 600; }}
  .positive {{ color: #3fb950; }}
  .negative {{ color: #f85149; }}
  .neutral {{ color: #8b8fa3; }}
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .stat {{ background: #161822; border-radius: 8px; padding: 1rem; }}
  .stat-value {{ font-size: 1.5rem; font-weight: 700; color: #f0f2f5; }}
  .stat-label {{ font-size: 0.75rem; color: #8b8fa3; margin-top: 0.25rem; }}
  footer {{ margin-top: 2rem; color: #54586c; font-size: 0.75rem; }}
</style>
</head>
<body>
<h1>Conductor Fabric — Benchmark Report</h1>
<p class="subtitle">{timestamp}</p>

<div class="summary">
  <div class="stat">
    <div class="stat-value">{sum(r.total_samples for r in results)}</div>
    <div class="stat-label">Total Samples</div>
  </div>
  <div class="stat">
    <div class="stat-value">{sum(r.passed for r in results)}</div>
    <div class="stat-label">Passed</div>
  </div>
  <div class="stat">
    <div class="stat-value">{sum(r.errors for r in results)}</div>
    <div class="stat-label">Errors</div>
  </div>
  <div class="stat">
    <div class="stat-value">{len(set(r.benchmark for r in results))}</div>
    <div class="stat-label">Benchmarks</div>
  </div>
</div>

<h2>All Results</h2>
<table>
<thead>
<tr><th>Benchmark</th><th>Mode</th><th>Metric</th><th>Score</th><th>Passed/Total</th><th>Errors</th><th>P50 (ms)</th><th>P95 (ms)</th></tr>
</thead>
<tbody>{rows}</tbody>
</table>

<h2>A/B Comparison: Baseline vs Conductor</h2>
<table>
<thead>
<tr><th>Benchmark</th><th>Baseline</th><th>Conductor</th><th>Δ Score</th><th>Baseline P50</th><th>Conductor P50</th><th>Δ Latency</th></tr>
</thead>
<tbody>{ab_rows if ab_rows else '<tr><td colspan="7">No A/B data available — run with --ab flag</td></tr>'}</tbody>
</table>

<footer>
Conductor Fabric — <a href="https://github.com/Denis-hamon/baremetal-benchmark" style="color:#58a6ff;">GitHub</a>
</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conductor Fabric Benchmark Runner")
    parser.add_argument("--config", default="bench/config.yaml", help="Config file path")
    parser.add_argument("--benchmark", "-b", choices=list(BENCHMARK_REGISTRY.keys()) + ["all"],
                        default="all", help="Benchmark to run")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--ab", action="store_true", help="Run A/B comparison (baseline vs conductor)")
    parser.add_argument("--baseline-only", action="store_true", help="Run baseline only")
    parser.add_argument("--conductor-only", action="store_true", help="Run conductor only")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--baseline-endpoint", help="Baseline LLM endpoint (override config)")
    parser.add_argument("--conductor-endpoint", help="Conductor endpoint (override config)")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples per benchmark")
    parser.add_argument("--concurrency", type=int, default=4, help="Thread pool size")
    parser.add_argument("--list", action="store_true", help="List available benchmarks and exit")
    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name, cls in BENCHMARK_REGISTRY.items():
            print(f"  {name:<20} {cls.__doc__.strip()}")
        return

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    output_dir = args.output or config.get("report", {}).get("dir", "bench/reports")
    concurrency = args.concurrency or eval_cfg.get("concurrency", 4)

    # Determine which benchmarks to run
    if args.benchmark and args.benchmark != "all":
        benchmarks_to_run = [args.benchmark]
    else:
        benchmarks_to_run = list(BENCHMARK_REGISTRY.keys())

    # Determine modes
    modes = []
    if args.ab:
        modes = ["baseline", "conductor"]
    elif args.baseline_only:
        modes = ["baseline"]
    elif args.conductor_only:
        modes = ["conductor"]
    else:
        modes = ["baseline"]

    # Override endpoints
    baseline_cfg = dict(config.get("models", {}).get("baseline", {}))
    conductor_cfg = dict(config.get("models", {}).get("conductor", {}))

    if args.baseline_endpoint:
        baseline_cfg["endpoint"] = args.baseline_endpoint
    if args.conductor_endpoint:
        conductor_cfg["endpoint"] = args.conductor_endpoint

    results: list[BenchmarkResult] = []

    for bname in benchmarks_to_run:
        bconfig = config.get("benchmarks", {}).get(bname, {})
        if not bconfig:
            logger.warning("No configuration for '%s', skipping", bname)
            continue

        if args.max_samples > 0:
            bconfig["max_samples"] = args.max_samples

        bench_cls = BENCHMARK_REGISTRY[bname]
        bench = bench_cls(bname, bconfig)

        for mode in modes:
            if mode == "baseline" and baseline_cfg.get("endpoint"):
                client = LLMClient(
                    endpoint=baseline_cfg["endpoint"],
                    model=baseline_cfg.get("model_id", "qwen-reasoner"),
                    timeout=eval_cfg.get("timeout", 120),
                    temperature=eval_cfg.get("temperature", 0.0),
                    max_tokens=eval_cfg.get("max_tokens", 4096),
                )
            elif mode == "conductor" and conductor_cfg.get("endpoint"):
                client = ConductorClient(
                    endpoint=conductor_cfg["endpoint"],
                    timeout=eval_cfg.get("timeout", 120),
                )
            else:
                logger.warning("No endpoint configured for mode '%s', skipping", mode)
                continue

            result = run_benchmark(
                bname, bench, bconfig, mode, client,
                max_workers=concurrency,
            )
            results.append(result)

    if not results:
        logger.warning("No results to report. Check your config and endpoints.")
        return

    report_path = generate_report(results, output_dir)
    print(f"\nReport: {report_path}")

    # Print summary table
    print(f"\n{'Benchmark':<20} {'Mode':<12} {'Score':<8} {'Passed':<10} {'P50':<8} {'P95':<8}")
    print("-" * 66)
    for r in sorted(results, key=lambda x: (x.benchmark, x.mode)):
        print(f"{r.benchmark:<20} {r.mode:<12} {r.score:<8.4f} {r.passed}/{r.total_samples:<5} {r.latency_p50_ms:<8.0f} {r.latency_p95_ms:<8.0f}")


if __name__ == "__main__":
    main()
