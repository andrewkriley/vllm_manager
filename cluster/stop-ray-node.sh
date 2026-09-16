#!/usr/bin/env bash
# Stop the Ray container on THIS host. Host Python is not modified.
set -euo pipefail
NAME="${VLLM_CONTAINER:-vllm-ray}"
if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker rm -f "${NAME}"
  echo "removed ${NAME}"
else
  echo "no container named ${NAME}"
fi
