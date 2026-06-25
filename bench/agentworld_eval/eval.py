#! /usr/bin/env python3
"""AgentWorldBench — evaluation pipeline (infer → judge → score).

Usage:
    # Step 1: Download dataset first
    bash bench/agentworld_eval/download.sh

    # Step 2: Run world model inference
    python -m bench.agentworld_eval.eval infer \\
        --data-dir bench/data/agentworld_bench \\
        --model-base-url http://localhost:8000/v1 \\
        --model-name Qwen/Qwen-AgentWorld-35B-A3B \\
        --output-dir bench/reports/agentworld

    # Step 3: Run judge scoring (requires a judge model like GPT-5.2)
    python -m bench.agentworld_eval.eval judge \\
        --predictions bench/reports/agentworld/predictions.jsonl \\
        --judge-base-url https://api.openai.com/v1 \\
        --judge-model gpt-5.2-2025-12-11 \\
        --output-dir bench/reports/agentworld

    # Step 4: Aggregate scores
    python -m bench.agentworld_eval.eval score \\
        --predictions bench/reports/agentworld/judged.jsonl
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from urllib import request

from .task_configs import SCORE_DIMENSIONS, DOMAIN_DISPLAY, TASK_CONFIGS, JUDGE_USER_PROMPT
from .judge_parser import parse_judge_output, load_judge_system_prompts
from .output_parser import parse_model_output, clean_response_marker

INVALID_VALUE = 0
logger = logging.getLogger("agentworld.eval")


def load_data(data_dir: str) -> list[dict]:
    data_dir = Path(data_dir)
    jobs = []
    for jsonl_file in sorted(data_dir.glob("*_test.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    jobs.append(json.loads(line))
    logger.info("Loaded %d evaluation samples from %s", len(jobs), data_dir)
    return jobs


def get_subtask(job: dict) -> str:
    task = job.get("task", "mcp")
    return task.split("/")[-1] if "/" in task else task


# ─── API client ───────────────────────────────────────────────────────────────

class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 32768, temperature: float = 0.6, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }).encode()
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"] or ""


# ─── Inference ───────────────────────────────────────────────────────────────

def run_inference(jobs: list[dict], client: OpenAICompatClient) -> list[dict]:
    total = len(jobs)
    for i, job in enumerate(jobs):
        subtask = get_subtask(job)
        system_prompt = job.get("system_str", "")
        current_prompt = job.get("current_prompt", "")
        if not current_prompt:
            prompts = job.get("prompt", [])
            turn_idx = job.get("turn_idx", 1) - 1
            if isinstance(prompts, list) and turn_idx < len(prompts):
                current_prompt = prompts[turn_idx]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": current_prompt})

        try:
            gen = client.complete(messages)
        except Exception as e:
            logger.warning("[%d/%d] Inference failed for %s: %s", i + 1, total, job.get("id"), e)
            gen = ""

        job["gen"] = gen
        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info("[%d/%d] Inference progress", i + 1, total)

    return jobs


# ─── Judge ───────────────────────────────────────────────────────────────────

def build_judge_messages(job: dict, model_output: str,
                         judge_system_prompts: dict) -> list[dict]:
    subtask = get_subtask(job)
    prompts = job.get("prompt", [])
    responses = job.get("response", [])
    if isinstance(prompts, str):
        prompts = [prompts]
    if isinstance(responses, str):
        responses = [responses]

    turn_idx = max(job.get("turn_idx", 1) - 1, 0)

    context = ""
    for i in range(turn_idx):
        if i < len(prompts) and i < len(responses):
            context += prompts[i] + "\n" + responses[i] + "\n\n"
    if context:
        context = f"# Context (Historical Interactions):\n\n{context}"

    current_prompt = job.get("current_prompt", "")
    if not current_prompt and turn_idx < len(prompts):
        current_prompt = prompts[turn_idx]

    ground_truth_raw = responses[turn_idx] if turn_idx < len(responses) else ""
    model_output_clean = clean_response_marker(model_output, subtask)
    ground_truth_clean = clean_response_marker(ground_truth_raw, subtask)

    user_prompt = JUDGE_USER_PROMPT.format(
        context=context,
        world_model_input=f"# Current Turn:\n\n{current_prompt}",
        predicted_observation=f"**World Model Output (Simulated):**\n```\n{model_output_clean}\n```",
        ground_truth=f"**Ground Truth (Real Output):**\n```\n{ground_truth_clean}\n```",
    ).strip()

    return [
        {"role": "system", "content": judge_system_prompts.get(subtask, "")},
        {"role": "user", "content": user_prompt},
    ]


def run_judge(jobs: list[dict], client: OpenAICompatClient,
              max_retries: int = 3) -> list[dict]:
    judge_system_prompts = load_judge_system_prompts()
    total = len(jobs)

    for i, job in enumerate(jobs):
        gen = job.get("gen", "")
        if not gen:
            job["failed"] = 1.0
            job["error_message"] = "No model generation"
            continue

        subtask = get_subtask(job)
        config = TASK_CONFIGS.get(subtask, TASK_CONFIGS["mcp"])
        response_tag = config.get("response_tag", "predicted_observation")
        judge_response_tag = config.get("judge_response_tag", "final_evaluation")

        model_output = parse_model_output(gen, response_tag)
        judge_messages = build_judge_messages(job, model_output, judge_system_prompts)

        parsed = None
        for attempt in range(max_retries):
            try:
                raw_output = client.complete(judge_messages)
            except Exception as e:
                logger.warning("[%d/%d] Judge call failed (attempt %d): %s",
                               i + 1, total, attempt + 1, e)
                time.sleep(2)
                continue

            parsed = parse_judge_output(raw_output, response_tag=judge_response_tag)
            if parsed["success"]:
                break
            logger.warning("[%d/%d] Judge parse failed (attempt %d)", i + 1, total, attempt + 1)

        if parsed and parsed["success"]:
            scores = parsed.get("scores", {})
            job.update({
                "total_score": parsed.get("total_score", 0.0),
                "format": scores.get("format", 0),
                "factuality": scores.get("factuality", 0),
                "consistency": scores.get("consistency", 0),
                "realism": scores.get("realism", 0),
                "quality": scores.get("quality", 0),
                "failed": 0.0,
                "strengths": parsed.get("strengths", []),
                "weaknesses": parsed.get("weaknesses", []),
                "extracted_output": model_output,
                "judge_raw_output": parsed.get("judge_raw_output", ""),
            })
        else:
            job.update({
                "total_score": INVALID_VALUE,
                "failed": 1.0,
                "error_message": "Judge scoring failed",
                "extracted_output": model_output,
            })

        if (i + 1) % 10 == 0 or (i + 1) == total:
            logger.info("[%d/%d] Judge progress", i + 1, total)

    return jobs


# ─── Score aggregation ───────────────────────────────────────────────────────

def aggregate_scores(jobs: list[dict]) -> None:
    subtask_jobs: dict = defaultdict(list)
    for job in jobs:
        subtask_jobs[get_subtask(job)].append(job)

    def normalize(raw: float) -> float:
        return (raw - 1) / 4 * 100

    print("\n" + "=" * 70)
    print("AgentWorldBench Evaluation Results")
    print("=" * 70)

    all_totals = []
    for subtask in sorted(subtask_jobs):
        sj = subtask_jobs[subtask]
        valid = [j for j in sj if j.get("failed", 1.0) == 0.0]
        failed = len(sj) - len(valid)
        display = DOMAIN_DISPLAY.get(subtask, subtask.capitalize())
        print(f"\n--- {display} ({len(valid)}/{len(sj)} valid, {failed} failed) ---")
        if not valid:
            print("  No valid results.")
            continue
        for dim in SCORE_DIMENSIONS + ["total_score"]:
            values = [j.get(dim, 0) for j in valid if j.get(dim, INVALID_VALUE) != INVALID_VALUE]
            avg = mean(values) if values else 0.0
            score = normalize(avg)
            if dim == "total_score":
                all_totals.extend(values)
            print(f"  {dim:>15s}: {score:.2f}")

    if all_totals:
        overall = normalize(mean(all_totals))
        print(f"\n{'=' * 70}")
        print(f"Overall: {overall:.2f}")
        print(f"Total samples: {len(jobs)}, Valid: {len(all_totals)}, "
              f"Failed: {len(jobs) - len(all_totals)}")
        print("=" * 70)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="AgentWorldBench Evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_infer = subparsers.add_parser("infer", help="Run world model inference")
    p_infer.add_argument("--data-dir", required=True)
    p_infer.add_argument("--model-base-url", required=True)
    p_infer.add_argument("--model-name", required=True)
    p_infer.add_argument("--model-api-key", default="EMPTY")
    p_infer.add_argument("--output-dir", default="./results")
    p_infer.add_argument("--max-tokens", type=int, default=32768)
    p_infer.add_argument("--temperature", type=float, default=0.6)

    p_judge = subparsers.add_parser("judge", help="Run LLM judge scoring")
    p_judge.add_argument("--predictions", required=True)
    p_judge.add_argument("--judge-base-url", required=True)
    p_judge.add_argument("--judge-model", required=True)
    p_judge.add_argument("--judge-api-key", default=None)
    p_judge.add_argument("--output-dir", default="./results")
    p_judge.add_argument("--max-tokens", type=int, default=32768)
    p_judge.add_argument("--temperature", type=float, default=0.6)
    p_judge.add_argument("--max-retries", type=int, default=3)

    p_score = subparsers.add_parser("score", help="Aggregate scores")
    p_score.add_argument("--predictions", required=True)

    args = parser.parse_args()

    if args.command == "infer":
        jobs = load_data(args.data_dir)
        client = OpenAICompatClient(args.model_base_url, args.model_api_key, args.model_name,
                                    args.max_tokens, args.temperature)
        jobs = run_inference(jobs, client)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "predictions.jsonl"
        with open(out_path, "w") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        logger.info("Predictions saved to %s", out_path)

    elif args.command == "judge":
        with open(args.predictions) as f:
            jobs = [json.loads(line) for line in f if line.strip()]
        api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        client = OpenAICompatClient(args.judge_base_url, api_key, args.judge_model,
                                    args.max_tokens, args.temperature)
        jobs = run_judge(jobs, client, args.max_retries)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "judged.jsonl"
        with open(out_path, "w") as f:
            for job in jobs:
                f.write(json.dumps(job, ensure_ascii=False) + "\n")
        logger.info("Judged results saved to %s", out_path)
        aggregate_scores(jobs)

    elif args.command == "score":
        with open(args.predictions) as f:
            jobs = [json.loads(line) for line in f if line.strip()]
        aggregate_scores(jobs)


if __name__ == "__main__":
    main()
