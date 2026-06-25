# Conductor Fabric — Benchmark Runner

Multi-benchmark evaluation framework with A/B comparison (direct LLM vs Conductor Fabric).

## Benchmarks

| Benchmark | Domain | Metric | Samples | Description |
|-----------|--------|--------|---------|-------------|
| HumanEval+ | Code | pass@1 | 164 | Python function completion with unit tests |
| GSM8K | Math | accuracy | 200+ | Grade-school math word problems |
| MMLU | Knowledge | accuracy | 200+ | Multi-subject multiple choice |
| Terminal-bench | CLI | accuracy | 40+ | Shell command generation |
| SWE-bench Lite | Code | resolved | 5+ | GitHub issue resolution |

## Usage

```bash
# Download benchmark datasets
make bench-download

# Run all benchmarks with A/B comparison
make bench-all

# Run single benchmark
make bench-humaneval

# Custom run
python bench/runner.py --benchmark gsm8k --ab --max-samples 50

# GPU mode (on HGR-AI-2)
make bench-docker ARGS="--all --ab"

# List available benchmarks
python bench/runner.py --list
```

## Output

Reports are written to `bench/reports/` in three formats:
- `bench_YYYYMMDD_HHMMSS.html` — interactive HTML report with A/B comparison table
- `bench_YYYYMMDD_HHMMSS.json` — full results with per-sample data
- `bench_YYYYMMDD_HHMMSS.csv` — compact summary table

## Adding a New Benchmark

1. Add a benchmark class in `runner.py` extending `Benchmark`
2. Implement `build_messages()`, `evaluate()`, and `_synthetic_data()`
3. Register it in `BENCHMARK_REGISTRY`
4. Add dataset config in `bench/config.yaml`
5. Create a download command in `download-datasets.sh`
