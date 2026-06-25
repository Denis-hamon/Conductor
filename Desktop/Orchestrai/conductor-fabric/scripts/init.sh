#!/usr/bin/env bash
# Conductor Fabric — Project Initialization Script
# Usage: ./scripts/init.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "Initializing Conductor Fabric project structure..."

mkdir -p "$ROOT_DIR"/{gateway,conductor,runtime,pool,shared/schemas,infra,scripts,prompts/{code,rag,reason,general,mcp},bench/{data,judge,reports},.github/workflows}

# Go module
if [ ! -f "$ROOT_DIR/gateway/go.mod" ]; then
    cat > "$ROOT_DIR/gateway/go.mod" <<'GOMOD'
module github.com/ovhcloud/conductor-fabric/gateway

go 1.24
GOMOD
    echo "  ✓ gateway/go.mod created"
fi

# Python projects
if [ ! -f "$ROOT_DIR/conductor/pyproject.toml" ]; then
    cat > "$ROOT_DIR/conductor/pyproject.toml" <<'PYPROJ'
[project]
name = "conductor-fabric-conductor"
version = "0.1.0"
description = "Conductor - heuristic routing and workflow plan generation"
requires-python = ">=3.13"
dependencies = []

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"
PYPROJ
    echo "  ✓ conductor/pyproject.toml created"
fi

if [ ! -f "$ROOT_DIR/runtime/pyproject.toml" ]; then
    cat > "$ROOT_DIR/runtime/pyproject.toml" <<'PYPROJ'
[project]
name = "conductor-fabric-runtime"
version = "0.1.0"
description = "LangGraph Runtime for workflow plan execution"
requires-python = ">=3.13"
dependencies = []

[build-system]
requires = ["setuptools>=75"]
build-backend = "setuptools.build_meta"
PYPROJ
    echo "  ✓ runtime/pyproject.toml created"
fi

# Dockerfiles
for comp in gateway conductor runtime; do
    if [ ! -f "$ROOT_DIR/Dockerfile.$comp" ]; then
        echo "  ⚠ Dockerfile.$comp missing — create manually"
    fi
done

# docker-compose
if [ ! -f "$ROOT_DIR/docker-compose.yml" ]; then
    echo "  ⚠ docker-compose.yml missing — create manually"
fi

# .gitignore
if [ ! -f "$ROOT_DIR/.gitignore" ]; then
    echo "  ⚠ .gitignore missing — create manually"
fi

echo ""
echo "✓ Project structure initialized at $ROOT_DIR"
echo "  Next: run 'make dev' to start local dependencies"
