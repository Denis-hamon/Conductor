.PHONY: dev dev-gpu test lint build clean benchmark bench-download setup

# ──────────────────────────────────────────────
# Local development (no GPU)
# ──────────────────────────────────────────────

dev:
	docker compose up -d

dev-logs:
	docker compose logs -f

dev-stop:
	docker compose down

dev-clean:
	docker compose down -v

# ──────────────────────────────────────────────
# With GPU model serving (HGR-AI-2, L40S)
# ──────────────────────────────────────────────

dev-gpu:
	docker compose --profile gpu up -d

dev-gpu-logs:
	docker compose --profile gpu logs -f

dev-gpu-stop:
	docker compose --profile gpu down

dev-gpu-clean:
	docker compose --profile gpu down -v

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────

test:
	cd gateway && go test ./...
	cd conductor && python -m pytest
	cd runtime && python -m pytest

test-gateway:
	cd gateway && go test -v ./...

test-conductor:
	cd conductor && python -m pytest -v

test-runtime:
	cd runtime && python -m pytest -v

# ──────────────────────────────────────────────
# Linting
# ──────────────────────────────────────────────

lint:
	cd gateway && golangci-lint run
	cd conductor && ruff check .
	cd runtime && ruff check .

# ──────────────────────────────────────────────
# Building
# ──────────────────────────────────────────────

build:
	cd gateway && go build -o bin/gateway ./cmd/
	cd conductor && pip install -e .
	cd runtime && pip install -e .
	cd bench && pip install -e .

build-docker:
	docker compose build

build-docker-gpu:
	docker compose --profile gpu build

# ──────────────────────────────────────────────
# Benchmark datasets
# ──────────────────────────────────────────────

bench-download:
	bash bench/data/download-datasets.sh

# Run benchmarks locally (no GPU — simulated mode)
bench:
	python bench/runner.py $(ARGS)

bench-all:
	python bench/runner.py --all --ab $(ARGS)

bench-humaneval:
	python bench/runner.py -b humaneval --ab $(ARGS)

bench-gsm8k:
	python bench/runner.py -b gsm8k --ab $(ARGS)

bench-swe:
	python bench/runner.py -b swe_bench_lite --ab $(ARGS)

bench-terminal:
	python bench/runner.py -b terminal_bench --ab $(ARGS)

bench-mmlu:
	python bench/runner.py -b mmlu --ab $(ARGS)

# AgentWorldBench commands
bench-agentworld-download:
	bash bench/agentworld_eval/download.sh

bench-agentworld-infer:
	python -m bench.agentworld_eval.eval infer \
		--data-dir bench/data/agentworld_bench \
		--model-base-url $(MODEL_URL) \
		--model-name $(MODEL_NAME) \
		--output-dir bench/reports/agentworld

bench-agentworld-judge:
	python -m bench.agentworld_eval.eval judge \
		--predictions bench/reports/agentworld/predictions.jsonl \
		--judge-base-url $(JUDGE_URL) \
		--judge-model $(JUDGE_MODEL) \
		--output-dir bench/reports/agentworld

bench-agentworld-score:
	python -m bench.agentworld_eval.eval score \
		--predictions bench/reports/agentworld/judged.jsonl

bench-agentworld: bench-agentworld-infer bench-agentworld-judge bench-agentworld-score

# Run benchmarks via Docker (GPU mode)
bench-docker:
	docker compose --profile bench run --rm bench $(ARGS)

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────

setup:
	bash scripts/setup-hgr-ai-2.sh

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────

clean:
	rm -rf gateway/bin conductor/dist runtime/dist
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .benchmarks -exec rm -rf {} + 2>/dev/null || true

# ──────────────────────────────────────────────
# GPU Monitoring
# ──────────────────────────────────────────────

gpu-stats:
	nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

gpu-watch:
	watch -n 1 nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv
