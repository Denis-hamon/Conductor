#!/usr/bin/env bash
# ============================================================================
# Conductor Fabric — HGR-AI-2 (2× L40S) Setup Script
# ============================================================================
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Denis-hamon/baremetal-benchmark/main/scripts/setup-hgr-ai-2.sh | bash
#
# Or locally:
#   bash scripts/setup-hgr-ai-2.sh
#
# This script installs and configures everything needed to run Conductor Fabric
# with vLLM model serving on an OVHcloud HGR-AI-2 server with 2× NVIDIA L40S.
#
# Prerequisites:
#   - Ubuntu 22.04+ or Debian 12+
#   - 2× NVIDIA L40S (48GB VRAM each)
#   - Root or sudo access
#   - Internet access
# ============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---- Config ----
REPO_URL="${REPO_URL:-https://github.com/Denis-hamon/baremetal-benchmark.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-/opt/conductor-fabric}"
MODELS_DIR="${MODELS_DIR:-/mnt/models}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
DOCKER_COMPOSE_VERSION="${DOCKER_COMPOSE_VERSION:-v2.34.0}"
SKIP_NVIDIA="${SKIP_NVIDIA:-false}"

# ---- Step tracking ----
TOTAL_STEPS=8
CURRENT_STEP=0

step() {
    CURRENT_STEP=$((CURRENT_STEP + 1))
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  Step $CURRENT_STEP/$TOTAL_STEPS : $*${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
}

# ============================================================================
# STEP 1: System dependencies
# ============================================================================
step "Installing system dependencies"

if [ "$(id -u)" -ne 0 ]; then
    log_warn "Some steps require sudo. You may be prompted for your password."
fi

sudo apt-get update -qq
sudo apt-get install -y -qq \
    ca-certificates curl gnupg lsb-release \
    git make build-essential python3 python3-pip python3-venv \
    nvtop htop iotop \
    jq unzip p7zip-full \
    ufw

log_ok "System packages installed"

# ============================================================================
# STEP 2: NVIDIA drivers & container toolkit
# ============================================================================
step "Installing NVIDIA drivers & container toolkit"

if [ "$SKIP_NVIDIA" = "true" ]; then
    log_warn "SKIP_NVIDIA=true — skipping GPU driver installation"
else
    # Check existing drivers
    if command -v nvidia-smi &>/dev/null; then
        log_ok "NVIDIA drivers already installed"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
    else
        log_info "Installing NVIDIA drivers..."
        sudo apt-get install -y -qq nvidia-driver-550-server || {
            log_info "Trying nvidia-driver-545..."
            sudo apt-get install -y -qq nvidia-driver-545-server
        }
        log_ok "NVIDIA drivers installed — reboot may be required"
    fi

    # NVIDIA Container Toolkit
    if command -v nvidia-ctk &>/dev/null; then
        log_ok "NVIDIA Container Toolkit already installed"
    else
        log_info "Installing NVIDIA Container Toolkit..."
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        sudo apt-get update -qq
        sudo apt-get install -y -qq nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        log_ok "NVIDIA Container Toolkit installed"
    fi
fi

# ============================================================================
# STEP 3: Docker & Docker Compose
# ============================================================================
step "Installing Docker & Docker Compose"

if command -v docker &>/dev/null; then
    log_ok "Docker already installed ($(docker --version))"
else
    log_info "Installing Docker..."
    curl -fsSL https://get.docker.com | bash
    sudo usermod -aG docker "$USER"
    log_ok "Docker installed — you may need to logout/login for docker group to take effect"
fi

if docker compose version &>/dev/null; then
    log_ok "Docker Compose already installed ($(docker compose version))"
else
    log_info "Installing Docker Compose..."
    sudo curl -fsSL "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-linux-$(uname -m)" \
        -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    log_ok "Docker Compose installed"
fi

# ============================================================================
# STEP 4: Clone repository & configure
# ============================================================================
step "Cloning Conductor Fabric repository"

if [ -d "$INSTALL_DIR" ]; then
    log_info "Directory $INSTALL_DIR exists, updating..."
    cd "$INSTALL_DIR"
    git pull origin "$REPO_BRANCH"
else
    sudo mkdir -p "$(dirname "$INSTALL_DIR")"
    sudo chown "$USER:$(id -gn)" "$(dirname "$INSTALL_DIR")"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
log_ok "Repository cloned to $INSTALL_DIR"

# Create directories for vLLM model storage
sudo mkdir -p "$MODELS_DIR"
sudo chown "$USER:$(id -gn)" "$MODELS_DIR"
log_ok "Model storage: $MODELS_DIR"

# ============================================================================
# STEP 5: Configure models for L40S
# ============================================================================
step "Configuring models for L40S (2× 48GB VRAM)"

# Ensure models.yaml is L40S-optimized
if [ -f "pool/models.yaml" ]; then
    log_ok "models.yaml found, verifying L40S config..."
    if grep -q "l40s" pool/models.yaml; then
        log_ok "models.yaml already configured for L40S"
    else
        log_warn "models.yaml is not L40S-optimized — verify manually"
    fi
fi

# Create Docker Compose override for vLLM model pool
cat > docker-compose.vllm.yml <<'OVERRIDE'
name: conductor-fabric-vllm

services:
  vllm-coder:
    image: vllm/vllm-openai:latest
    container_name: vllm-coder
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
    volumes:
      - /mnt/models:/root/.cache/huggingface
    command:
      - "--model"
      - "Qwen/Qwen3-Coder-30B-A3B"
      - "--quantization"
      - "fp8"
      - "--max-model-len"
      - "65536"
      - "--gpu-memory-utilization"
      - "0.90"
      - "--port"
      - "8001"
      - "--host"
      - "0.0.0.0"
    ports:
      - "8001:8001"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  vllm-reasoner:
    image: vllm/vllm-openai:latest
    container_name: vllm-reasoner
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
    volumes:
      - /mnt/models:/root/.cache/huggingface
    command:
      - "--model"
      - "Qwen/Qwen3-32B-A3B"
      - "--quantization"
      - "fp8"
      - "--max-model-len"
      - "131072"
      - "--gpu-memory-utilization"
      - "0.90"
      - "--port"
      - "8002"
      - "--host"
      - "0.0.0.0"
    ports:
      - "8002:8002"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s

  vllm-general:
    image: vllm/vllm-openai:latest
    container_name: vllm-general
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
    volumes:
      - /mnt/models:/root/.cache/huggingface
    command:
      - "--model"
      - "google/gemma-3-12b-it"
      - "--quantization"
      - "fp8"
      - "--max-model-len"
      - "32768"
      - "--gpu-memory-utilization"
      - "0.40"
      - "--port"
      - "8003"
      - "--host"
      - "0.0.0.0"
      - "--trust-remote-code"
      - "--enforce-eager"
    ports:
      - "8003:8003"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["1"]
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8003/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s
OVERRIDE

log_ok "vLLM Docker Compose file created: docker-compose.vllm.yml"

# ============================================================================
# STEP 6: Firewall & security
# ============================================================================
step "Configuring firewall"

sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8080/tcp comment "Conductor Fabric Gateway"
sudo ufw allow 8001/tcp comment "vLLM Coder"
sudo ufw allow 8002/tcp comment "vLLM Reasoner"
sudo ufw allow 8003/tcp comment "vLLM General"
sudo ufw --force enable
log_ok "Firewall configured"

# ============================================================================
# STEP 7: Download models & start services
# ============================================================================
step "Starting services"

# Create .env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    log_info "Created .env from .env.example — edit to set your HF_TOKEN"
fi

# Source env
set -a; source .env 2>/dev/null || true; set +a

# Start infrastructure (Postgres, Valkey, ClickHouse)
log_info "Starting infrastructure (Postgres + Valkey + ClickHouse)..."
docker compose -f docker-compose.yml up -d
log_ok "Infrastructure started"

# Download models (this will pull from HuggingFace)
log_info "Pulling model images (first time may take 5-10 minutes)..."
docker compose -f docker-compose.vllm.yml pull
log_ok "Model images pulled"

log_info "Starting vLLM model servers..."
docker compose -f docker-compose.vllm.yml up -d
log_ok "vLLM services started"

# Build and start Conductor Fabric stack
log_info "Building Conductor Fabric services..."
docker compose -f docker-compose.yml -f docker-compose.vllm.yml build gateway conductor runtime
docker compose -f docker-compose.yml -f docker-compose.vllm.yml up -d gateway conductor runtime
log_ok "Conductor Fabric stack started"

# ============================================================================
# STEP 8: Verify & benchmark datasets
# ============================================================================
step "Verifying installation & downloading benchmark datasets"

echo ""
echo -e "${BLUE}────── Service Status ──────${NC}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

echo ""
echo -e "${BLUE}────── GPU Status ──────${NC}"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"

echo ""
echo -e "${BLUE}────── Model Health ──────${NC}"
for port in 8001 8002 8003; do
    if curl -sf "http://localhost:$port/health" >/dev/null 2>&1; then
        log_ok "vLLM on port $port is healthy"
    else
        log_warn "vLLM on port $port is not ready yet (still loading model?)"
    fi
done

echo ""
echo -e "${BLUE}────── Downloading Benchmark Datasets ──────${NC}"
bash bench/data/download-datasets.sh

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Conductor Fabric — Setup Complete!                         ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Services:                                                    ║${NC}"
echo -e "${GREEN}║    Gateway:     http://localhost:8080                          ║${NC}"
echo -e "${GREEN}║    Conductor:  http://localhost:9090                          ║${NC}"
echo -e "${GREEN}║    Runtime:    http://localhost:7070                          ║${NC}"
echo -e "${GREEN}║    vLLM Coder:     http://localhost:8001 (GPU 0)                ║${NC}"
echo -e "${GREEN}║    vLLM Reasoner:  http://localhost:8002 (GPU 1)                ║${NC}"
echo -e "${GREEN}║    vLLM General:   http://localhost:8003 (GPU 1, shared)        ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  Run benchmarks:                                               ║${NC}"
echo -e "${GREEN}║    cd $INSTALL_DIR                                            ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  A/B comparison (all benchmarks):                              ║${NC}"
echo -e "${GREEN}║    python bench/runner.py --all --ab                            ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  Single benchmark:                                             ║${NC}"
echo -e "${GREEN}║    python bench/runner.py -b humaneval --ab                     ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  View report:                                                  ║${NC}"
echo -e "${GREEN}║    open bench/reports/bench_*.html                             ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
