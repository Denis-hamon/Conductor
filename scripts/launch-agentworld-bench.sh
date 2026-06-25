#!/usr/bin/env bash
# Launch AgentWorldBench evaluation on GPU server
# Usage: bash scripts/launch-agentworld-bench.sh [infer|judge|score|all]
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/Conductor}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen-AgentWorld-35B-A3B}"
MODEL_URL="${MODEL_URL:-http://localhost:8000/v1}"
JUDGE_URL="${JUDGE_URL:-}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.2-2025-12-11}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/bench/reports/agentworld}"
DATA_DIR="${DATA_DIR:-$REPO_DIR/bench/data/agentworld_bench}"
DEPLOYMENT="${DEPLOYMENT:-$REPO_DIR/pool/sglang/agentworld-deployment.yaml}"

cd "$REPO_DIR"
source .venv/bin/activate

cmd="${1:-all}"

ensure_data() {
    if [ ! -f "$DATA_DIR/agentworld_bench.parquet" ]; then
        echo "==> Dataset not found. Downloading..."
        bash bench/agentworld_eval/download.sh
    fi
}

start_sglang() {
    if curl -s "$MODEL_URL/models" >/dev/null 2>&1; then
        echo "==> SGLang already running at $MODEL_URL"
        return 0
    fi
    echo "==> Starting SGLang with $DEPLOYMENT..."
    if command -v kubectl &>/dev/null; then
        kubectl apply -f "$DEPLOYMENT"
        echo "==> Waiting for SGLang pod to be ready..."
        kubectl wait --for=condition=ready pod -l app=agentworld --timeout=600s
    elif command -v docker &>/dev/null; then
        docker run --gpus all -d --name agentworld \
            -p 8000:8000 \
            -v /mnt/models:/mnt/models \
            lmsysorg/sglang:latest \
            python3 -m sglang.launch_server \
            --model "$MODEL_NAME" \
            --tp 2 \
            --quantization fp8 \
            --host 0.0.0.0 --port 8000
    else
        echo "ERROR: neither kubectl nor docker available"
        exit 1
    fi
    sleep 10
    for i in $(seq 1 60); do
        if curl -s "$MODEL_URL/models" >/dev/null 2>&1; then
            echo "==> SGLang ready after ${i}s"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: SGLang did not start"
    exit 1
}

do_infer() {
    ensure_data
    echo "==> Running inference..."
    python3 -m bench.agentworld_eval.eval infer \
        --data-dir "$DATA_DIR" \
        --model-base-url "$MODEL_URL" \
        --model-name "$MODEL_NAME" \
        --output-dir "$OUTPUT_DIR"
    echo "==> Predictions saved to $OUTPUT_DIR/predictions.jsonl"
}

do_judge() {
    if [ ! -f "$OUTPUT_DIR/predictions.jsonl" ]; then
        echo "ERROR: $OUTPUT_DIR/predictions.jsonl not found. Run infer first."
        exit 1
    fi
    if [ -z "$JUDGE_URL" ]; then
        echo "ERROR: JUDGE_URL not set. Export JUDGE_URL and JUDGE_API_KEY"
        exit 1
    fi
    echo "==> Running judge scoring..."
    python3 -m bench.agentworld_eval.eval judge \
        --predictions "$OUTPUT_DIR/predictions.jsonl" \
        --judge-base-url "$JUDGE_URL" \
        --judge-model "$JUDGE_MODEL" \
        --judge-api-key "$JUDGE_API_KEY" \
        --output-dir "$OUTPUT_DIR"
}

do_score() {
    if [ ! -f "$OUTPUT_DIR/judged.jsonl" ]; then
        echo "ERROR: $OUTPUT_DIR/judged.jsonl not found. Run judge first."
        exit 1
    fi
    echo "==> Aggregating scores..."
    python3 -m bench.agentworld_eval.eval score \
        --predictions "$OUTPUT_DIR/judged.jsonl"
}

case "$cmd" in
    infer)  start_sglang; do_infer ;;
    judge)  do_judge ;;
    score)  do_score ;;
    all)
        start_sglang
        do_infer
        do_judge
        do_score
        ;;
    *)
        echo "Usage: $0 [infer|judge|score|all]"
        exit 1
        ;;
esac
