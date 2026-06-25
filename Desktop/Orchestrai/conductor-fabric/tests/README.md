# Tests — OVHcloud Conductor Fabric

## Structure

```
tests/
├── conftest.py                 # Shared fixtures, factories, helpers
├── unit/                       # Unit tests (fast, no external deps)
│   ├── test_workflow_plan.py   #   WorkflowPlan schema + routing
│   ├── test_verification_sandbox.py  # Sandbox code verification
│   ├── test_verification_rag.py      # RAG citation verification
│   ├── test_guardrails_pii.py        # PII detection guardrails
│   └── test_reward.py                # Reward calculation
├── integration/                # Integration tests (require pool/db)
├── api/                        # API contract tests
└── README.md
```

## Setup

```bash
# Python dependencies
pip install pytest pytest-asyncio pytest-benchmark pytest-cov

# Go dependencies (gateway only)
cd gateway && go mod tidy
```

## Running Tests

```bash
# All tests
make test

# Go tests only
make test-go

# Python tests only
make test-python

# ATDD acceptance tests (P0)
make test-python-atdd

# Security tests (sandbox escape, PII)
make test-python-security

# With coverage
make test-python-coverage
```

## Markers

| Marker       | Purpose                                    |
|-------------|--------------------------------------------|
| `@unit`     | Fast, no external dependencies (default)     |
| `@integration` | Requires model pool, ClickHouse, or DB   |
| `@slow`     | Performance benchmarks                      |
| `@security` | Security tests (sandbox escape, PII)         |
| `@atdd`     | ATDD acceptance tests (given/when/then)      |
| `@chaos`    | Chaos engineering tests (RPO/RTO)            |

## Fixtures

Shared fixtures in `conftest.py`:

- `sample_code_block` — Python code string for sandbox tests
- `sample_unit_tests` — pytest unit tests string
- `sample_prompt` — Standard OpenAI-format request
- `benchmark_dataset` — 100-prompt benchmark (SM-1/SM-2/SM-3)
- `valid_workflow_plan()` — Factory for valid WorkflowPlan dicts
- `valid_verification_result()` — Factory for valid VerificationResult dicts
- `load_json_schema(name)` — Load JSON Schema from `shared/schemas/`

## CI Integration

Tests run in CI via GitHub Actions:

1. **Unit gate** — `make test-python-unit` + `make test-go-short` (every PR, < 2min)
2. **ATDD gate** — `make test-python-atdd` (every PR, < 1min)
3. **Integration gate** — `make test-python-integration` (nightly, requires GPU pool)
4. **Security gate** — `make test-python-security` (every release)
5. **Performance gate** — k6 benchmark suite (every release, SM-1/SM-2/SM-3)

## Best Practices

1. **Given/When/Then** — Use BDD format in ATDD tests (`@pytest.mark.atdd`)
2. **Isolation** — Each test creates its own data; no shared state
3. **Factories** — Use `valid_workflow_plan()` and `valid_verification_result()` overrides
4. **Fixtures** — Session-scoped for expensive resources (model pool connections)
5. **No network** — Unit tests never call external services; mock all HTTP calls
6. **Tag by priority** — `@p0`, `@p1` markers for selective execution
