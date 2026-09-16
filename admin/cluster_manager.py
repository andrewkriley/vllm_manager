"""Two-host Ray cluster control for vLLM.

Starts/stops the existing Docker Ray head/worker (fabric IPs) and runs
`vllm serve` inside the head container. This is not CUDA_VISIBLE_DEVICES
in the admin process.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class ClusterState(str, Enum):
    DISABLED = "disabled"
    STOPPED = "stopped"
    STARTING_RAY = "starting_ray"
    STARTING_VLLM = "starting_vllm"
    RUNNING = "running"
    ERROR = "error"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ClusterSettings:
    enabled: bool = field(default_factory=lambda: _env_bool("CLUSTER_ENABLED"))
    ssh_user: str = field(default_factory=lambda: os.getenv("CLUSTER_SSH_USER", "ubuntu"))
    ssh_key: str = field(default_factory=lambda: os.getenv("CLUSTER_SSH_KEY", "/keys/vllm_manager_ed25519"))
    head_host: str = field(default_factory=lambda: os.getenv("CLUSTER_HEAD_HOST", ""))
    worker_host: str = field(default_factory=lambda: os.getenv("CLUSTER_WORKER_HOST", ""))
    head_fabric: str = field(default_factory=lambda: os.getenv("CLUSTER_HEAD_FABRIC", ""))
    worker_fabric: str = field(default_factory=lambda: os.getenv("CLUSTER_WORKER_FABRIC", ""))
    container: str = field(default_factory=lambda: os.getenv("CLUSTER_CONTAINER", "vllm-ray"))
    run_script: str = field(
        default_factory=lambda: os.getenv(
            "CLUSTER_RUN_SCRIPT", "/home/andreril/aipod/docker/run-cluster.sh"
        )
    )
    stop_script: str = field(
        default_factory=lambda: os.getenv(
            "CLUSTER_STOP_SCRIPT", "/home/andreril/aipod/docker/stop-cluster.sh"
        )
    )
    restrict_script: str = field(
        default_factory=lambda: os.getenv(
            "CLUSTER_RESTRICT_SCRIPT", "/home/andreril/aipod/scripts/restrict-fabric.sh"
        )
    )
    serve_host: str = field(default_factory=lambda: os.getenv("CLUSTER_SERVE_HOST", "0.0.0.0"))
    serve_port: int = field(default_factory=lambda: int(os.getenv("CLUSTER_API_PORT", "8000")))
    health_url: str = field(
        default_factory=lambda: os.getenv("CLUSTER_HEALTH_URL", "http://host.docker.internal:8000")
    )
    public_api: str = field(
        default_factory=lambda: os.getenv("CLUSTER_PUBLIC_API", "http://127.0.0.1:8000")
    )


class ClusterManager:
    def __init__(self, settings: Optional[ClusterSettings] = None) -> None:
        self.settings = settings or ClusterSettings()
        self.state = ClusterState.DISABLED if not self.settings.enabled else ClusterState.STOPPED
        self.logs: list[str] = []
        self.error: Optional[str] = None
        self.model: Optional[str] = None
        self.served_model_name: Optional[str] = None
        self.tensor_parallel_size: int = 4
        self.pipeline_parallel_size: int = 2
        self.gpu_memory_utilization: float = 0.90
        self.cmd: Optional[str] = None
        self._log_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None

    def _append_log(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        self.logs.append(line)
        if len(self.logs) > 400:
            self.logs = self.logs[-400:]

    def _ssh_cmd(self, host: str, remote: str) -> list[str]:
        return [
            "ssh",
            "-i", self.settings.ssh_key,
            "-o", "IdentitiesOnly=yes",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15",
            "-o", "LogLevel=ERROR",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{self.settings.ssh_user}@{host}",
            remote,
        ]

    async def _run(self, cmd: list[str], timeout: float = 120) -> tuple[int, str]:
        logger.info("cluster exec: %s", " ".join(cmd[:8]))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out_b, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, f"timeout after {timeout}s"
        out = (out_b or b"").decode("utf-8", errors="replace")
        return proc.returncode or 0, out

    async def _ssh(self, host: str, remote: str, timeout: float = 120) -> tuple[int, str]:
        return await self._run(self._ssh_cmd(host, remote), timeout=timeout)

    async def _head(self, remote: str, timeout: float = 120) -> tuple[int, str]:
        return await self._ssh(self.settings.head_host, remote, timeout=timeout)

    async def _worker(self, remote: str, timeout: float = 120) -> tuple[int, str]:
        return await self._ssh(self.settings.worker_host, remote, timeout=timeout)

    def public_config(self) -> dict:
        s = self.settings
        return {
            "enabled": s.enabled,
            "head_host": s.head_host,
            "worker_host": s.worker_host,
            "head_fabric": s.head_fabric,
            "worker_fabric": s.worker_fabric,
            "container": s.container,
            "serve_host": s.serve_host,
            "serve_port": s.serve_port,
            "public_api": s.public_api,
            "defaults": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 2,
                "gpu_memory_utilization": 0.90,
            },
        }

    def get_status(self) -> dict:
        return {
            "id": "cluster-ray",
            "kind": "cluster",
            "state": self.state.value,
            "error": self.error,
            "model": self.model,
            "served_model_name": self.served_model_name,
            "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7] if self.state in {
                ClusterState.STARTING_VLLM, ClusterState.RUNNING,
            } else [],
            "port": self.settings.serve_port,
            "public_api": self.settings.public_api,
            "dtype": "auto",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "cmd": self.cmd,
            "cuda_visible_devices": "cluster (fabric Ray, not CUDA_VISIBLE_DEVICES)",
            "logs": self.logs[-100:],
            "head_host": self.settings.head_host,
            "worker_host": self.settings.worker_host,
        }

    async def ray_snapshot(self) -> dict:
        if not self.settings.enabled:
            return {"enabled": False}
        head_ps, head_out = await self._head(
            f"docker ps --filter name=^{self.settings.container}$ --format '{{{{.Status}}}}'"
        )
        worker_ps, worker_out = await self._worker(
            f"docker ps --filter name=^{self.settings.container}$ --format '{{{{.Status}}}}'"
        )
        ray_txt = ""
        if head_ps == 0 and head_out.strip():
            _, ray_txt = await self._head(
                f"docker exec {self.settings.container} ray status"
            )
        return {
            "enabled": True,
            "head_container": head_out.strip() if head_ps == 0 else "",
            "worker_container": worker_out.strip() if worker_ps == 0 else "",
            "ray_status": ray_txt.strip()[-2000:],
        }

    async def ensure_ray(self) -> None:
        self.state = ClusterState.STARTING_RAY
        self.error = None
        snap = await self.ray_snapshot()
        if snap.get("head_container") and snap.get("worker_container") and (
            "2 node" in (snap.get("ray_status") or "") or "8.0" in (snap.get("ray_status") or "")
        ):
            self._append_log("Ray already up on both hosts")
            return
        self._append_log("starting Ray head")
        code, out = await self._head(f"bash {shlex.quote(self.settings.run_script)} head", timeout=90)
        self._append_log(out[-1500:])
        if code != 0:
            self.state = ClusterState.ERROR
            self.error = f"Ray head failed: {out[-500:]}"
            raise RuntimeError(self.error)

        self._append_log("starting Ray worker")
        code, out = await self._worker(f"bash {shlex.quote(self.settings.run_script)} worker", timeout=90)
        self._append_log(out[-1500:])
        if code != 0:
            self.state = ClusterState.ERROR
            self.error = f"Ray worker failed: {out[-500:]}"
            raise RuntimeError(self.error)

        code, out = await self._head(f"bash {shlex.quote(self.settings.restrict_script)}", timeout=30)
        self._append_log(out[-500:] or "fabric iptables ok")

        for _ in range(15):
            _, status = await self._head(f"docker exec {self.settings.container} ray status")
            if "2 node" in status or "8.0/8.0 GPU" in status or "8.0 GPU" in status:
                self._append_log("Ray cluster has 2 nodes")
                return
            await asyncio.sleep(2)
        self._append_log("Ray started; waiting for full GPU report")

    async def start_serve(
        self,
        model: str,
        tensor_parallel_size: int = 4,
        pipeline_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.90,
        served_model_name: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
        trust_remote_code: bool = True,
    ) -> None:
        if not self.settings.enabled:
            raise RuntimeError("Cluster mode is disabled")
        if self.state in {ClusterState.STARTING_RAY, ClusterState.STARTING_VLLM, ClusterState.RUNNING}:
            raise RuntimeError(f"Cluster is already {self.state.value}")

        self.logs = []
        self.model = model
        self.served_model_name = served_model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization

        await self.ensure_ray()

        model_path = model if model.startswith("/") else f"/models/{model}"
        serve = [
            "vllm serve",
            shlex.quote(model_path),
            "--host", shlex.quote(self.settings.serve_host),
            "--port", str(self.settings.serve_port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--pipeline-parallel-size", str(pipeline_parallel_size),
            "--distributed-executor-backend ray",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
        ]
        if served_model_name:
            serve.extend(["--served-model-name", shlex.quote(served_model_name)])
        if trust_remote_code:
            serve.append("--trust-remote-code")
        if extra_args:
            serve.extend(shlex.quote(a) for a in extra_args)
        inner = "ulimit -n 1048576; exec " + " ".join(serve) + " > /tmp/vllm-serve.log 2>&1"
        self.cmd = inner
        remote = (
            f"docker exec {self.settings.container} bash -lc "
            f"{shlex.quote('pkill -f /usr/local/bin/vllm serve || true')}; "
            f"docker exec -d {self.settings.container} bash -lc {shlex.quote(inner)}"
        )
        self.state = ClusterState.STARTING_VLLM
        self._append_log("starting vllm serve on head (fabric Ray)")
        code, out = await self._head(remote, timeout=30)
        self._append_log(out[-800:])
        if code != 0:
            self.state = ClusterState.ERROR
            self.error = f"vllm serve launch failed: {out[-500:]}"
            raise RuntimeError(self.error)

        self._log_task = asyncio.create_task(self._poll_logs())
        self._health_task = asyncio.create_task(self._poll_health())

    async def _poll_logs(self) -> None:
        while self.state in {ClusterState.STARTING_VLLM, ClusterState.RUNNING}:
            try:
                _, out = await self._head(
                    f"docker exec {self.settings.container} bash -lc "
                    f"'tail -n 60 /tmp/vllm-serve.log 2>/dev/null || true'"
                )
                for line in out.splitlines():
                    if line and (not self.logs or line != self.logs[-1]):
                        self._append_log(line)
            except Exception as exc:
                logger.warning("cluster log poll failed: %s", exc)
            await asyncio.sleep(3)

    async def _poll_health(self) -> None:
        url = self.settings.health_url.rstrip("/") + "/health"
        async with httpx.AsyncClient() as client:
            while self.state == ClusterState.STARTING_VLLM:
                await asyncio.sleep(4)
                try:
                    resp = await client.get(url, timeout=3)
                    if resp.status_code == 200:
                        self.state = ClusterState.RUNNING
                        self._append_log(f"vLLM healthy at {self.settings.public_api}")
                        return
                except httpx.RequestError:
                    pass
                _, alive = await self._head(
                    f"docker exec {self.settings.container} bash -lc "
                    "\"pgrep -f 'vllm serve' >/dev/null && echo ALIVE || echo DEAD\""
                )
                if "DEAD" in alive and self.logs:
                    # Allow a few cycles before the process exists
                    if any("RuntimeError" in x or "Traceback" in x for x in self.logs[-30:]):
                        self.state = ClusterState.ERROR
                        self.error = "vllm serve process exited; see logs"
                        self._append_log(self.error)
                        return

    async def stop_serve(self) -> None:
        self._append_log("stopping vllm serve")
        await self._head(
            f"docker exec {self.settings.container} bash -lc "
            f"'pkill -f /usr/local/bin/vllm serve || pkill -f \"vllm serve\" || true'"
        )
        self.model = None
        self.cmd = None
        self.state = ClusterState.STOPPED if self.settings.enabled else ClusterState.DISABLED
        if self._log_task:
            self._log_task.cancel()
        if self._health_task:
            self._health_task.cancel()

    async def stop_ray(self) -> None:
        await self.stop_serve()
        self._append_log("stopping Ray worker")
        await self._worker(f"bash {shlex.quote(self.settings.stop_script)}", timeout=60)
        self._append_log("stopping Ray head")
        await self._head(f"bash {shlex.quote(self.settings.stop_script)}", timeout=60)
        self.state = ClusterState.STOPPED if self.settings.enabled else ClusterState.DISABLED
        self._append_log("Ray cluster stopped")
