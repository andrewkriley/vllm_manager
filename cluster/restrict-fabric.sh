#!/usr/bin/env bash
# Restrict Ray GCS / dashboard to the fabric subnet. SSH on the management NIC stays open.
# Requires passwordless sudo for iptables, or run as root.
set -euo pipefail
FABRIC="${FABRIC_CIDR:?set FABRIC_CIDR e.g. 192.168.200.0/24}"
RAY_PORT="${RAY_PORT:-6379}"
DASH_PORT="${RAY_DASHBOARD_PORT:-8265}"
COMMENT="${IPTABLES_COMMENT:-vllm-manager-fabric}"

if sudo iptables -S INPUT | grep -q "$COMMENT"; then
  echo "iptables ${COMMENT} rules already present"
  exit 0
fi
sudo iptables -I INPUT 1 -p tcp --dport "$RAY_PORT" -s "$FABRIC" -j ACCEPT -m comment --comment "$COMMENT"
sudo iptables -I INPUT 2 -p tcp --dport "$RAY_PORT" -j DROP -m comment --comment "$COMMENT"
sudo iptables -I INPUT 3 -p tcp --dport "$DASH_PORT" -s "$FABRIC" -j ACCEPT -m comment --comment "$COMMENT"
sudo iptables -I INPUT 4 -p tcp --dport "$DASH_PORT" -j DROP -m comment --comment "$COMMENT"
echo "applied fabric restriction for tcp/${RAY_PORT} and tcp/${DASH_PORT} (allow ${FABRIC})"
sudo iptables -S INPUT | grep "$COMMENT"
