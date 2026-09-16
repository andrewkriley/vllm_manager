# TODO

## High Priority

- [x] HuggingFace token support (UI field + `HF_TOKEN`/`HF_API` env, never logged)
- [ ] Concurrent model downloads (currently limited to one at a time)
- [ ] Auto-restart instances on crash (configurable)
- [ ] Persist instance configurations across container restarts

## Features

- [ ] Model deletion from the UI
- [ ] Instance resource monitoring (GPU utilization, memory per instance)
- [ ] Configurable vLLM arguments per instance (additional CLI flags)
- [ ] Model search from HuggingFace Hub in the UI
- [ ] Instance naming (custom names instead of instance-1, instance-2)
- [ ] API key / basic auth for the admin UI
- [x] docker-compose.yml / podman-compose.yml
- [x] Two-host Ray cluster mode (SSH + Docker Ray, not host pip)
- [x] Cluster dependency preflight (`deploy/check-deps.sh`, `GET /api/cluster/preflight`)
- [x] Cluster GPU selector shortcuts + auto PP=2 when both hosts are selected
- [x] Worker rsync from the controller using the cluster SSH key (`rsync` in `Containerfile.cluster`)
- [x] Gated Hugging Face probe (readable 401/403, not cache-miss)
- [x] Health check endpoint for the admin container itself
- [ ] Cluster stop without bouncing Ray can leak placement groups / actor handles; next Start may hit `ActorHandleNotFoundError` across Ray jobs. Workaround: `"ray": true` on stop, or recreate the Ray containers.
- [ ] HTTPS support for admin UI

## UI Improvements

- [x] Dark/light theme toggle
- [x] Log filtering and search
- [x] Log download/export
- [x] GPU utilization charts over time
- [x] Mobile-responsive layout improvements
- [x] Toast notifications instead of alert() dialogs
- [x] Confirmation dialog before stopping instances

## Technical Debt

- [ ] Add unit tests for vllm_manager.py
- [ ] Add integration tests for API endpoints
- [x] Health check endpoint for the admin container itself
- [ ] Structured JSON logging
- [ ] Rate limiting on download endpoint
