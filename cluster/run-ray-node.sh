#!/usr/bin/env bash
# Start a Ray head or worker in Docker on THIS host. Host Python is not modified.
# Required env: VLLM_HEAD_IP, VLLM_WORKER_IP
set -euo pipefail

ROLE="${1:?usage: run-ray-node.sh head|worker}"
IMAGE="${VLLM_IMAGE:-vllm-manager-ray:latest}"
NAME="${VLLM_CONTAINER:-vllm-ray}"
HEAD_IP="${VLLM_HEAD_IP:?set VLLM_HEAD_IP (fabric IP of the head)}"
WORKER_IP="${VLLM_WORKER_IP:?set VLLM_WORKER_IP (fabric IP of the worker)}"
MODELS="${VLLM_MODELS_DIR:-/models}"
HF_CACHE="${VLLM_HF_CACHE:-${MODELS}/hf-cache}"
RAY_PORT="${RAY_PORT:-6379}"
NUM_GPUS="${NUM_GPUS:-}"
if [ -z "$NUM_GPUS" ]; then
  NUM_GPUS="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
fi
if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" = "0" ]; then
  echo "no GPUs visible via nvidia-smi" >&2
  exit 1
fi

case "${ROLE}" in
  head)
    HOST_IP="${HEAD_IP}"
    RAY_CMD="ray start --block --head --node-ip-address=${HEAD_IP} --port=${RAY_PORT} --num-gpus=${NUM_GPUS}"
    ;;
  worker)
    HOST_IP="${WORKER_IP}"
    RAY_CMD="ray start --block --address=${HEAD_IP}:${RAY_PORT} --node-ip-address=${WORKER_IP} --num-gpus=${NUM_GPUS}"
    ;;
  *)
    echo "usage: $0 head|worker" >&2
    exit 1
    ;;
esac

mkdir -p "${MODELS}" "${HF_CACHE}"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
  docker rm -f "${NAME}" >/dev/null
fi

RUN_ARGS=(
  run -d
  --name "${NAME}"
  --entrypoint /bin/bash
  --network host
  --ipc host
  --gpus all
  --ulimit memlock=-1
  --ulimit stack=67108864
  --ulimit nofile=1048576:1048576
  --shm-size "${SHM_SIZE:-16g}"
  -v "${MODELS}:/models"
  -v "${HF_CACHE}:/root/.cache/huggingface"
  -e "VLLM_HOST_IP=${HOST_IP}"
)

if [ "${CLUSTER_PRIVILEGED:-true}" = "true" ]; then
  RUN_ARGS+=(--privileged)
fi
if [ -d /dev/infiniband ]; then
  RUN_ARGS+=(-v /dev/infiniband:/dev/infiniband)
fi
if [ -n "${NCCL_SOCKET_IFNAME:-}" ]; then
  RUN_ARGS+=(-e "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}" -e "GLOO_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}")
fi
if [ -n "${NCCL_IB_HCA:-}" ]; then
  RUN_ARGS+=(-e "NCCL_IB_HCA=${NCCL_IB_HCA}")
fi
if [ -n "${NCCL_IB_GID_INDEX:-}" ]; then
  RUN_ARGS+=(-e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}")
fi
if [ -n "${NCCL_NET_GDR_LEVEL:-}" ]; then
  RUN_ARGS+=(-e "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}")
fi

RUN_ARGS+=("${IMAGE}" -c "${RAY_CMD}")

echo "starting ${NAME} role=${ROLE} node_ip=${HOST_IP} gpus=${NUM_GPUS} image=${IMAGE}"
exec docker "${RUN_ARGS[@]}"
