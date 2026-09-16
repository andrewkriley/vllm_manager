#!/usr/bin/env bash
# Preflight for two-host Ray + vLLM Manager. Does not print key material.
# Usage: bash deploy/check-deps.sh   (from repo root, .env populated)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

fail=0
ok() { printf '  [ok]  %s\n' "$1"; }
bad() { printf '  [FAIL] %s\n' "$1"; fail=1; }
warn() { printf '  [warn] %s\n' "$1"; }

SSH_USER="${CLUSTER_SSH_USER:?set CLUSTER_SSH_USER}"
HEAD="${CLUSTER_HEAD_HOST:?set CLUSTER_HEAD_HOST}"
WORKER="${CLUSTER_WORKER_HOST:?set CLUSTER_WORKER_HOST}"
KEY="${CLUSTER_SSH_KEY_HOST:-$HOME/.ssh/vllm_manager_ed25519}"
IMAGE="${VLLM_IMAGE:-vllm-manager-ray:latest}"
MODELS="${MODELS_DIR:-/models}"
HEAD_FABRIC="${CLUSTER_HEAD_FABRIC:-}"
WORKER_FABRIC="${CLUSTER_WORKER_FABRIC:-}"

ssh_base() {
  ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 \
    -o LogLevel=ERROR -o StrictHostKeyChecking=accept-new "$@"
}

host_kv() {
  local host="$1"
  ssh_base "${SSH_USER}@${host}" "bash -s" <<EOF
set +e
echo docker=\$(docker info >/dev/null 2>&1 && echo ok || echo FAIL)
echo docker_group=\$(id -nG | grep -qw docker && echo ok || echo FAIL)
echo nvidia_smi=\$(command -v nvidia-smi >/dev/null && echo ok || echo FAIL)
echo gpu_count=\$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
echo models=\$(test -d ${MODELS} && echo ok || echo FAIL)
echo script=\$(test -x "\$HOME/vllm_manager/cluster/run-ray-node.sh" && echo ok || echo FAIL)
echo image=\$(docker image inspect ${IMAGE} >/dev/null 2>&1 && echo ok || echo missing)
echo ib=\$(test -d /dev/infiniband && echo yes || echo no)
echo sudo_iptables=\$(sudo -n iptables -S INPUT >/dev/null 2>&1 && echo ok || echo FAIL)
EOF
}

score() {
  local label="$1" report="$2"
  while IFS= read -r line; do
    case "$line" in
      docker=ok) ok "$label docker daemon" ;;
      docker=FAIL) bad "$label docker daemon" ;;
      docker_group=ok) ok "$label docker group" ;;
      docker_group=FAIL) bad "$label user not in docker group" ;;
      nvidia_smi=ok) ok "$label nvidia-smi" ;;
      nvidia_smi=FAIL) bad "$label nvidia-smi missing" ;;
      gpu_count=0) bad "$label reports 0 GPUs" ;;
      gpu_count=*) ok "$label GPUs: ${line#gpu_count=}" ;;
      models=ok) ok "$label ${MODELS}" ;;
      models=FAIL) bad "$label ${MODELS} missing" ;;
      script=ok) ok "$label node scripts (~/vllm_manager/cluster)" ;;
      script=FAIL) bad "$label install scripts: bash cluster/install-node.sh ${SSH_USER}@host" ;;
      image=ok) ok "$label image ${IMAGE}" ;;
      image=missing) bad "$label image ${IMAGE} missing — Ray is IN THE IMAGE, not apt/pip on the host. Run cluster/build-ray-image.sh there." ;;
      ib=yes) ok "$label /dev/infiniband" ;;
      ib=no) warn "$label no InfiniBand (set NCCL_SOCKET_IFNAME for Ethernet)" ;;
      sudo_iptables=ok) ok "$label passwordless sudo iptables" ;;
      sudo_iptables=FAIL) warn "$label no passwordless sudo iptables — disable CLUSTER_RESTRICT_FABRIC or add sudoers" ;;
    esac
  done <<< "$report"
}

echo "== controller =="
if command -v docker >/dev/null; then ok "docker CLI"; else warn "docker CLI not on this machine (only needed if you build the admin UI here)"; fi
if [ -f "$KEY" ]; then ok "SSH key ${KEY}"; else bad "SSH key missing: ${KEY}"; fi

echo "== head ${HEAD} =="
if ssh_base "${SSH_USER}@${HEAD}" true; then ok "SSH"; else bad "SSH BatchMode to head"; fi
score head "$(host_kv "$HEAD" || true)"

echo "== worker ${WORKER} =="
if ssh_base "${SSH_USER}@${WORKER}" true; then ok "SSH"; else bad "SSH BatchMode to worker"; fi
score worker "$(host_kv "$WORKER" || true)"

if [ -n "$HEAD_FABRIC" ] && [ -n "$WORKER_FABRIC" ]; then
  echo "== fabric =="
  if ssh_base "${SSH_USER}@${HEAD}" "ping -c 1 -W 2 ${WORKER_FABRIC} >/dev/null"; then
    ok "head ping ${WORKER_FABRIC}"
  else
    bad "head cannot ping worker fabric ${WORKER_FABRIC}"
  fi
fi

echo
echo "Ray is not a host package. It must exist inside ${IMAGE} (cluster/Dockerfile.ray)."
if [ "$fail" -ne 0 ]; then
  echo "preflight FAILED"
  exit 1
fi
echo "preflight PASSED"
exit 0
