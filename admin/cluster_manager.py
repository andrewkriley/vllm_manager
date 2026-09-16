"""Two-host Ray cluster control for vLLM.

Starts/stops the existing Docker Ray head/worker (fabric IPs) and runs
`vllm serve` inside the head container. This is not CUDA_VISIBLE_DEVICES
in the admin process.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import time
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
    scripts_dir: str = field(
        default_factory=lambda: os.getenv("CLUSTER_SCRIPTS_DIR", "$HOME/vllm_manager/cluster")
    )
    vllm_image: str = field(
        default_factory=lambda: os.getenv("VLLM_IMAGE", "vllm-manager-ray:latest")
    )
    models_dir: str = field(default_factory=lambda: os.getenv("MODELS_DIR", "/models"))
    nccl_socket_ifname: str = field(default_factory=lambda: os.getenv("NCCL_SOCKET_IFNAME", ""))
    nccl_ib_hca: str = field(default_factory=lambda: os.getenv("NCCL_IB_HCA", ""))
    nccl_ib_gid_index: str = field(default_factory=lambda: os.getenv("NCCL_IB_GID_INDEX", ""))
    nccl_net_gdr_level: str = field(default_factory=lambda: os.getenv("NCCL_NET_GDR_LEVEL", ""))
    fabric_cidr: str = field(default_factory=lambda: os.getenv("FABRIC_CIDR", ""))
    restrict_fabric: bool = field(default_factory=lambda: _env_bool("CLUSTER_RESTRICT_FABRIC", True))
    num_gpus: str = field(default_factory=lambda: os.getenv("NUM_GPUS", ""))
    run_script: str = field(default_factory=lambda: os.getenv("CLUSTER_RUN_SCRIPT", ""))
    stop_script: str = field(default_factory=lambda: os.getenv("CLUSTER_STOP_SCRIPT", ""))
    restrict_script: str = field(default_factory=lambda: os.getenv("CLUSTER_RESTRICT_SCRIPT", ""))
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
        self.gpu_ids: list[int] = []
        self.cmd: Optional[str] = None
        self._log_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._gpu_cache: list[dict] = []
        self._gpu_cache_at: float = 0.0

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

    def _resolved_script(self, filename: str, override: str) -> str:
        if override:
            return override
        return f"{self.settings.scripts_dir.rstrip('/')}/{filename}"

    def _bash_script(self, filename: str, override: str) -> str:
        path = self._resolved_script(filename, override)
        if path.startswith("$") or path.startswith("~"):
            return f"bash {path}"
        return f"bash {shlex.quote(path)}"

    def _node_exports(self, extra: Optional[dict[str, str]] = None) -> str:
        s = self.settings
        vals = {
            "VLLM_HEAD_IP": s.head_fabric,
            "VLLM_WORKER_IP": s.worker_fabric,
            "VLLM_IMAGE": s.vllm_image,
            "VLLM_CONTAINER": s.container,
            "VLLM_MODELS_DIR": s.models_dir,
            "NCCL_SOCKET_IFNAME": s.nccl_socket_ifname,
            "NCCL_IB_HCA": s.nccl_ib_hca,
            "NCCL_IB_GID_INDEX": s.nccl_ib_gid_index,
            "NCCL_NET_GDR_LEVEL": s.nccl_net_gdr_level,
            "FABRIC_CIDR": s.fabric_cidr,
            "NUM_GPUS": s.num_gpus,
        }
        if extra:
            vals.update(extra)
        return " ".join(f"{k}={shlex.quote(str(v))}" for k, v in vals.items() if v)

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
            "scripts_dir": s.scripts_dir,
            "vllm_image": s.vllm_image,
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
            "gpu_ids": self.gpu_ids if self.state in {
                ClusterState.STARTING_RAY, ClusterState.STARTING_VLLM, ClusterState.RUNNING,
            } else [],
            "port": self.settings.serve_port,
            "public_api": self.settings.public_api,
            "dtype": "auto",
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "cmd": self.cmd,
            "cuda_visible_devices": ",".join(str(i) for i in self.gpu_ids) if self.gpu_ids else "cluster (all visible Ray GPUs)",
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

    async def _vllm_is_healthy(self) -> bool:
        port = self.settings.serve_port
        code, out = await self._head(
            f"curl -sS -m 3 -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health"
        )
        if code == 0 and out.strip() == "200":
            return True
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(self.settings.health_url.rstrip("/") + "/health")
            return resp.status_code == 200
        except Exception:
            return False

    async def _adopt_running(self) -> None:
        self.state = ClusterState.RUNNING
        self.error = None
        models_url = self.settings.health_url.rstrip("/") + "/v1/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                models = await client.get(models_url)
            data = (models.json() or {}).get("data") or []
            if data:
                self.model = data[0].get("id")
                self.served_model_name = data[0].get("id")
        except Exception:
            _, out = await self._head(
                "curl -sS -m 5 http://127.0.0.1:8000/v1/models"
            )
            try:
                data = (json.loads(out) or {}).get("data") or []
                if data:
                    self.model = data[0].get("id")
                    self.served_model_name = data[0].get("id")
            except Exception:
                pass
        inventory = await self.list_gpus()
        if inventory and not self.gpu_ids:
            self.gpu_ids = [g["index"] for g in inventory]
        self._gpu_cache_at = 0.0
        self._append_log(f"recovered running cluster model={self.model}")

    async def recover(self) -> None:
        """Adopt a cluster that is already serving after the controller restarts."""
        if not self.settings.enabled:
            return
        if self.state == ClusterState.RUNNING:
            return
        if await self._vllm_is_healthy():
            await self._adopt_running()
            return
        snap = await self.ray_snapshot()
        if snap.get("head_container") or snap.get("worker_container"):
            self._append_log(
                "Ray containers are up; vLLM health was not ready "
                f"head={snap.get('head_container')!r} worker={snap.get('worker_container')!r}"
            )

    def _cuda_extra(self, local_ids: Optional[list[int]], role: str) -> Optional[dict[str, str]]:
        if not local_ids:
            return None
        total = sum(1 for g in self._gpu_cache if g.get("role") == role)
        if total and sorted(local_ids) == list(range(total)):
            return None
        return {
            "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in local_ids),
            "NUM_GPUS": str(len(local_ids)),
        }

    async def _split_gpu_ids(self, gpu_ids: list[int]) -> tuple[list[int], list[int]]:
        inventory = self._gpu_cache or await self.list_gpus()
        by_index = {g["index"]: g for g in inventory}
        missing = [i for i in gpu_ids if i not in by_index]
        if missing:
            raise RuntimeError(f"Unknown GPU ids: {missing}")
        head: list[int] = []
        worker: list[int] = []
        for i in gpu_ids:
            g = by_index[i]
            local = int(g.get("local_index", i))
            if g.get("role") == "worker":
                worker.append(local)
            else:
                head.append(local)
        return sorted(head), sorted(worker)

    async def ensure_ray(
        self,
        head_devs: Optional[list[int]] = None,
        worker_devs: Optional[list[int]] = None,
    ) -> None:
        self.state = ClusterState.STARTING_RAY
        self.error = None
        if head_devs is not None or worker_devs is not None:
            await self.list_gpus()
        head_extra = self._cuda_extra(head_devs, "head") if head_devs else None
        worker_extra = self._cuda_extra(worker_devs, "worker") if worker_devs else None
        pin_devices = bool(head_extra or worker_extra)
        snap = await self.ray_snapshot()
        ray_up = bool(
            snap.get("head_container")
            and (worker_devs == [] or snap.get("worker_container"))
            and (
                "2 node" in (snap.get("ray_status") or "")
                or "1 node" in (snap.get("ray_status") or "")
                or "GPU" in (snap.get("ray_status") or "")
            )
        )
        if ray_up and not pin_devices:
            self._append_log("Ray already up; GPU selection uses the full node(s)")
            return

        run = self._bash_script("run-ray-node.sh", self.settings.run_script)
        stop = self._bash_script("stop-ray-node.sh", self.settings.stop_script)
        if pin_devices and (snap.get("head_container") or snap.get("worker_container")):
            self._append_log("restarting Ray so CUDA_VISIBLE_DEVICES matches the GPU selection")
            await self._worker(stop, timeout=60)
            await self._head(stop, timeout=60)

        self._append_log("starting Ray head")
        env = self._node_exports(head_extra)
        code, out = await self._head(f"{env} {run} head", timeout=90)
        self._append_log(out[-1500:])
        if code != 0:
            self.state = ClusterState.ERROR
            self.error = f"Ray head failed: {out[-500:]}"
            raise RuntimeError(self.error)

        if worker_devs == []:
            self._append_log("no worker GPUs selected; single-node Ray")
        else:
            self._append_log("starting Ray worker")
            env = self._node_exports(worker_extra)
            code, out = await self._worker(f"{env} {run} worker", timeout=90)
            self._append_log(out[-1500:])
            if code != 0:
                self.state = ClusterState.ERROR
                self.error = f"Ray worker failed: {out[-500:]}"
                raise RuntimeError(self.error)

        if self.settings.restrict_fabric:
            restrict = self._bash_script("restrict-fabric.sh", self.settings.restrict_script)
            env = self._node_exports()
            code, out = await self._head(f"{env} {restrict}", timeout=30)
            self._append_log(out[-500:] or "fabric iptables ok")
        else:
            self._append_log("skipping fabric iptables (CLUSTER_RESTRICT_FABRIC=false)")

        want_worker = worker_devs != []
        for _ in range(15):
            _, status = await self._head(f"docker exec {self.settings.container} ray status")
            if want_worker and ("2 node" in status or "8.0/8.0 GPU" in status or "8.0 GPU" in status):
                self._append_log("Ray cluster has 2 nodes")
                return
            if not want_worker and ("1 node" in status or "Active:" in status):
                self._append_log("Ray head is up")
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
        gpu_ids: Optional[list[int]] = None,
    ) -> None:
        if not self.settings.enabled:
            raise RuntimeError("Cluster mode is disabled")
        if await self._vllm_is_healthy():
            await self._adopt_running()
            raise RuntimeError("Cluster is already running")
        if self.state in {ClusterState.STARTING_RAY, ClusterState.STARTING_VLLM, ClusterState.RUNNING}:
            raise RuntimeError(f"Cluster is already {self.state.value}")

        self.logs = []
        self.model = model
        self.served_model_name = served_model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.gpu_ids = list(gpu_ids or [])

        if not self.settings.head_fabric or not self.settings.worker_fabric:
            raise RuntimeError("CLUSTER_HEAD_FABRIC and CLUSTER_WORKER_FABRIC must be set")

        head_devs = worker_devs = None
        if gpu_ids:
            head_devs, worker_devs = await self._split_gpu_ids(gpu_ids)
            if not head_devs:
                raise RuntimeError("Select at least one GPU on the Ray head; vLLM serve runs there")
        await self.ensure_ray(head_devs=head_devs, worker_devs=worker_devs)

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
            f"{shlex.quote('pkill -f vllm.serve || true')}; "
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
        self.gpu_ids = []
        self.state = ClusterState.STOPPED if self.settings.enabled else ClusterState.DISABLED
        if self._log_task:
            self._log_task.cancel()
        if self._health_task:
            self._health_task.cancel()

    async def stop_ray(self) -> None:
        await self.stop_serve()
        stop = self._bash_script("stop-ray-node.sh", self.settings.stop_script)
        self._append_log("stopping Ray worker")
        await self._worker(stop, timeout=60)
        self._append_log("stopping Ray head")
        await self._head(stop, timeout=60)
        self.state = ClusterState.STOPPED if self.settings.enabled else ClusterState.DISABLED
        self._append_log("Ray cluster stopped")

    async def preflight(self) -> dict:
        """SSH both hosts and report Ray/SSH/Docker/GPU/image readiness."""
        checks: list[dict] = []

        def add(host: str, name: str, ok: bool, detail: str = "") -> None:
            checks.append({"host": host, "name": name, "ok": ok, "detail": detail})

        s = self.settings
        if not s.enabled:
            add("controller", "CLUSTER_ENABLED", False, "set CLUSTER_ENABLED=true")
            return {"ok": False, "checks": checks}
        add("controller", "CLUSTER_ENABLED", True)
        add("controller", "head_host", bool(s.head_host), s.head_host)
        add("controller", "worker_host", bool(s.worker_host), s.worker_host)
        add("controller", "head_fabric", bool(s.head_fabric), s.head_fabric)
        add("controller", "worker_fabric", bool(s.worker_fabric), s.worker_fabric)
        add("controller", "ssh_key", os.path.exists(s.ssh_key), s.ssh_key)

        probe = (
            "set +e; "
            "echo docker=$(docker info >/dev/null 2>&1 && echo ok || echo FAIL); "
            "echo gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '); "
            f"echo models=$(test -d {shlex.quote(s.models_dir)} && echo ok || echo FAIL); "
            "echo script=$(test -x $HOME/vllm_manager/cluster/run-ray-node.sh && echo ok || echo FAIL); "
            f"echo image=$(docker image inspect {shlex.quote(s.vllm_image)} >/dev/null 2>&1 && echo ok || echo missing); "
            "echo ib=$(test -d /dev/infiniband && echo yes || echo no)"
        )
        for label, host in (("head", s.head_host), ("worker", s.worker_host)):
            if not host:
                add(label, "ssh", False, "host not configured")
                continue
            code, out = await self._ssh(host, probe, timeout=25)
            add(label, "ssh", code == 0, out[-200:] if code != 0 else "")
            fields = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
            add(label, "docker", fields.get("docker") == "ok", "")
            gpus = fields.get("gpus", "0")
            add(label, "gpus", gpus.isdigit() and int(gpus) > 0, gpus)
            add(label, "models", fields.get("models") == "ok", s.models_dir)
            add(
                label,
                "node_scripts",
                fields.get("script") == "ok",
                "$HOME/vllm_manager/cluster/run-ray-node.sh",
            )
            add(
                label,
                "ray_image",
                fields.get("image") == "ok",
                "Ray must be in the Docker image, not apt/pip on the host" if fields.get("image") != "ok" else s.vllm_image,
            )
            add(label, "infiniband", True, fields.get("ib", "unknown"))

        if s.head_fabric and s.worker_fabric and s.head_host:
            code, _ = await self._head(f"ping -c 1 -W 2 {shlex.quote(s.worker_fabric)} >/dev/null")
            add("fabric", "head_to_worker_ping", code == 0, s.worker_fabric)

        ok = all(c["ok"] for c in checks if c["name"] != "infiniband")
        return {"ok": ok, "checks": checks}

    def _gpu_in_use(self, index: int) -> bool:
        if self.state not in {
            ClusterState.STARTING_RAY,
            ClusterState.STARTING_VLLM,
            ClusterState.RUNNING,
        }:
            return False
        if self.gpu_ids:
            return index in self.gpu_ids
        return True

    async def list_gpus(self) -> list[dict]:
        """Read GPU inventory over SSH. The controller has no pynvml/GPUs."""
        if not self.settings.enabled:
            return []
        now = time.monotonic()
        if not (self._gpu_cache and now - self._gpu_cache_at < 2.0):
            query = (
                "nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,compute_cap "
                "--format=csv,noheader,nounits"
            )
            gpus: list[dict] = []
            offset = 0
            for role, host in (("head", self.settings.head_host), ("worker", self.settings.worker_host)):
                if not host:
                    continue
                code, out = await self._ssh(host, query, timeout=15)
                parsed = 0
                if code == 0:
                    for line in out.splitlines():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) < 5 or not parts[0].isdigit():
                            continue
                        local_idx = int(parts[0])
                        try:
                            total, used, free = (int(float(parts[i])) for i in (2, 3, 4))
                        except ValueError:
                            continue
                        gpus.append({
                            "index": offset + local_idx,
                            "local_index": local_idx,
                            "name": parts[1],
                            "memory_total_mb": total,
                            "memory_used_mb": used,
                            "memory_free_mb": free,
                            "compute_capability": parts[5] if len(parts) > 5 else "unknown",
                            "source": "cluster",
                            "host": host,
                            "role": role,
                        })
                        parsed += 1
                if parsed == 0:
                    logger.warning("cluster GPU probe failed on %s (%s): %s", role, host, out[-200:])
                offset += parsed if parsed else 4
            self._gpu_cache = gpus
            self._gpu_cache_at = time.monotonic()
        annotated = []
        for g in self._gpu_cache:
            in_use = self._gpu_in_use(g["index"])
            annotated.append({
                **g,
                "in_use": in_use,
                "used_by": "cluster-ray" if in_use else None,
                "selectable": not in_use,
            })
        return annotated

    async def sync_model_to_worker(self, model_name: str) -> None:
        """Copy a model directory from this controller's /models mount to the worker.

        Uses the cluster SSH key in this container. Do not rsync from the head
        host — that host's default key is not authorized for BatchMode to the worker.
        """
        if not self.settings.enabled or not self.settings.worker_host:
            return
        src = f"{self.settings.models_dir.rstrip('/')}/{model_name}"
        if not os.path.isdir(src):
            raise RuntimeError(f"model directory missing on controller: {src}")
        user = self.settings.ssh_user
        worker = self.settings.worker_host
        mkdir_code, mkdir_out = await self._worker(f"mkdir -p {shlex.quote(src)}", timeout=30)
        if mkdir_code != 0:
            raise RuntimeError(f"could not create {src} on worker: {mkdir_out[-300:]}")
        rsync_ssh = (
            f"ssh -i {shlex.quote(self.settings.ssh_key)} "
            "-o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 "
            "-o LogLevel=ERROR -o StrictHostKeyChecking=accept-new"
        )
        dest = f"{user}@{worker}:{src}/"
        cmd = [
            "rsync", "-a", "--info=stats2",
            "-e", rsync_ssh,
            src + "/",
            dest,
        ]
        self._append_log(f"syncing {model_name} to worker {worker}")
        code, out = await self._run(cmd, timeout=86400)
        if code != 0:
            raise RuntimeError(f"rsync to worker failed: {out[-500:]}")
        self._append_log(f"synced {model_name} to worker")
