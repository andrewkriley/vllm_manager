#!/usr/bin/env bash
# Build the Ray+vLLM image on THIS GPU host. Host Python is not modified.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${VLLM_IMAGE:-vllm-manager-ray:latest}"
BASE="${VLLM_BASE_IMAGE:-vllm/vllm-openai:latest}"
echo "building ${IMAGE} FROM ${BASE}"
docker build --build-arg "VLLM_BASE_IMAGE=${BASE}" -t "${IMAGE}" -f "${SCRIPT_DIR}/Dockerfile.ray" "${SCRIPT_DIR}"
echo "checking ray CLI in image"
docker run --rm --entrypoint ray "${IMAGE}" --version
