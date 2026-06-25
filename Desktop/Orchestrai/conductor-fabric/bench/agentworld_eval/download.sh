#!/usr/bin/env bash
# AgentWorldBench — Download full dataset (Parquet) + create judge prompts
# Usage: bash bench/agentworld_eval/download.sh
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/../data" && pwd)/agentworld_bench"
PROMPTS_DIR="$(cd "$(dirname "$0")" && pwd)/prompts"
PARQUET_URL="https://huggingface.co/api/datasets/Qwen/AgentWorldBench/parquet/default/test/0.parquet"
PARQUET_FILE="$DATA_DIR/agentworld_bench.parquet"

mkdir -p "$DATA_DIR" "$PROMPTS_DIR"

echo "=== AgentWorldBench — Download ==="

echo "[1/3] Downloading Parquet dataset (2170 samples, ~257 MB)..."
if [ -f "$PARQUET_FILE" ] && [ -s "$PARQUET_FILE" ]; then
    echo "      Already cached ($(ls -lh "$PARQUET_FILE" | awk '{print $5}'))"
else
    if command -v wget &>/dev/null; then
        wget -q --show-progress "$PARQUET_URL" -O "$PARQUET_FILE"
    elif command -v curl &>/dev/null; then
        curl -sL "$PARQUET_URL" -o "$PARQUET_FILE"
    else
        python3 -c "
import urllib.request, ssl
ctx = ssl._create_unverified_context()
urllib.request.urlretrieve('$PARQUET_URL', '$PARQUET_FILE')
"
    fi
    echo "      Downloaded ($(ls -lh "$PARQUET_FILE" | awk '{print $5}'))"
fi

echo "[2/3] Converting Parquet to per-domain JSONL..."
python3 << 'PYEOF'
import json, os, pyarrow.parquet as pq

parquet_path = "$PARQUET_FILE"
out_dir = "$DATA_DIR"

table = pq.read_table(parquet_path)
domains = table.column("task").to_pylist()
unique = sorted(set(domains))
print(f"      Domains: {unique}")

col_names = table.column_names
for domain in unique:
    rows = []
    for i in range(len(domains)):
        if domains[i] != domain:
            continue
        row = {}
        for col in col_names:
            v = table.column(col)[i]
            if hasattr(v, "as_py"):
                v = v.as_py()
            row[col] = v
        rows.append(row)
    path = os.path.join(out_dir, f"{domain}_test.jsonl")
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    print(f"      {domain}: {len(rows)} samples")
PYEOF

echo "[3/3] Creating judge system prompts..."
python3 -c "
from pathlib import Path
import urllib.request, ssl, json

prompts_dir = Path('$PROMPTS_DIR')
domains = ['mcp', 'search', 'terminal', 'swe', 'android', 'web', 'os']

# Try to download actual judge system prompts from HF repo
base = 'https://raw.githubusercontent.com/QwenLM/Qwen-AgentWorld/main/prompts'
ctx = ssl._create_unverified_context()
for d in domains:
    domain_dir = prompts_dir / d
    domain_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = domain_dir / 'judge_system_prompt.txt'
    if not prompt_file.exists():
        url = f'{base}/{d}/judge_system_prompt.txt'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
                content = resp.read().decode()
            prompt_file.write_text(content)
            print(f'      {d}: downloaded judge prompt ({len(content)} chars)')
        except Exception:
            print(f'      {d}: using minimal judge prompt')
"

echo ""
echo "=== AgentWorldBench ready ==="
samples=$(find "$DATA_DIR" -name "*_test.jsonl" -exec wc -l {} + | tail -1 | awk '{print $1}')
echo "  Dataset: $DATA_DIR ($samples total samples)"
echo "  Next steps:"
echo "    make bench-agentworld-infer MODEL_URL=http://... MODEL_NAME=..."
echo "    make bench-agentworld-judge JUDGE_URL=https://... JUDGE_MODEL=gpt-5.2-..."
echo "    make bench-agentworld-score"
