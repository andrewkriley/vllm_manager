#!/usr/bin/env bash
# Copy cluster node scripts onto a GPU host (run from a machine that can SSH).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:?usage: install-node.sh user@host [remote_dir]}"
DEST="${2:-vllm_manager/cluster}"
ssh -o BatchMode=yes "$HOST" "mkdir -p ~/${DEST}"
scp -o BatchMode=yes \
  "$SCRIPT_DIR/run-ray-node.sh" \
  "$SCRIPT_DIR/stop-ray-node.sh" \
  "$SCRIPT_DIR/restrict-fabric.sh" \
  "$SCRIPT_DIR/Dockerfile.ray" \
  "$SCRIPT_DIR/build-ray-image.sh" \
  "${HOST}:~/${DEST}/"
ssh -o BatchMode=yes "$HOST" "chmod +x ~/${DEST}/*.sh"
echo "installed scripts to ${HOST}:~/${DEST}"
