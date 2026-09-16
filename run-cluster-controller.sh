#!/usr/bin/env bash
# Admin UI that controls the two-host Ray cluster over SSH. No GPUs in this container.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env"
  set +a
fi

IMAGE_NAME="${IMAGE_NAME:-vllm-manager-cluster:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-manager}"
ADMIN_PORT="${ADMIN_PORT:-7080}"
MODELS_DIR="${MODELS_DIR:-/models}"
SSH_KEY="${CLUSTER_SSH_KEY_HOST:-$HOME/.ssh/vllm_manager_ed25519}"
CLUSTER_SSH_USER="${CLUSTER_SSH_USER:?set CLUSTER_SSH_USER in .env}"
CLUSTER_HEAD_HOST="${CLUSTER_HEAD_HOST:?set CLUSTER_HEAD_HOST in .env}"
CLUSTER_WORKER_HOST="${CLUSTER_WORKER_HOST:?set CLUSTER_WORKER_HOST in .env}"
CLUSTER_HEAD_FABRIC="${CLUSTER_HEAD_FABRIC:-}"
CLUSTER_WORKER_FABRIC="${CLUSTER_WORKER_FABRIC:-}"
CLUSTER_HEALTH_URL="${CLUSTER_HEALTH_URL:-http://host.docker.internal:8000}"
CLUSTER_PUBLIC_API="${CLUSTER_PUBLIC_API:-http://127.0.0.1:8000}"

if [ ! -f "$SSH_KEY" ]; then
  echo "missing cluster SSH key: $SSH_KEY" >&2
  exit 1
fi

echo "Building $IMAGE_NAME"
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Containerfile.cluster" "$SCRIPT_DIR"

docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  --add-host=host.docker.internal:host-gateway \
  -p "${ADMIN_PORT}:7080" \
  -v "${MODELS_DIR}:/models:ro" \
  -v "${SSH_KEY}:/keys/vllm_manager_ed25519:ro" \
  -e CLUSTER_ENABLED=true \
  -e CLUSTER_SSH_USER="${CLUSTER_SSH_USER}" \
  -e CLUSTER_SSH_KEY=/keys/vllm_manager_ed25519 \
  -e CLUSTER_HEAD_HOST="${CLUSTER_HEAD_HOST}" \
  -e CLUSTER_WORKER_HOST="${CLUSTER_WORKER_HOST}" \
  -e CLUSTER_HEAD_FABRIC="${CLUSTER_HEAD_FABRIC}" \
  -e CLUSTER_WORKER_FABRIC="${CLUSTER_WORKER_FABRIC}" \
  -e CLUSTER_HEALTH_URL="${CLUSTER_HEALTH_URL}" \
  -e CLUSTER_PUBLIC_API="${CLUSTER_PUBLIC_API}" \
  -e MODELS_DIR=/models \
  "$IMAGE_NAME"

echo "Admin UI: http://$(hostname -I | awk '{print $1}'):${ADMIN_PORT}"
