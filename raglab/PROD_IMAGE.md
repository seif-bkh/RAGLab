# RAGLab production image — build once, run anywhere (incl. offline)

`Dockerfile.prod` builds the compose stack (one service: `raglab`) as a
**multi-stage, self-contained image**. Build it (or let CI build it), then
run it on any Docker host — including machines that never touch the internet.

## What "offline" means here — read this section first

The container **starts and serves with zero internet**: no pip, no apt, no
model downloads, no tokenizer fetch. CI proves it on every image build with
`docker run --network none` + a smoke suite + the image's own HEALTHCHECK.

The one thing that is *by design* not offline: the RAG pipeline's provider
calls. `/ingest`, `/search` and `/answer` call the NVIDIA embeddings endpoint
and the xKiro answer endpoint over HTTPS — that is the lab's architecture
(hosted models), not an image limitation. In a banking deployment that
egress normally goes through your gateway/proxy allowlist (`HTTPS_PROXY` is
honored — the clients use stdlib HTTPS). A fully air-gapped outbound network
would need a local-model provider, which this repo does not ship.

What the image bakes in to make the rest hermetic:

| Concern | Dev image | `Dockerfile.prod` |
|---|---|---|
| pip at runtime | already no | no (venv copied from the `deps` stage) |
| `cl100k_base` tokenizer (tiktoken CDN) | downloaded on first chunking — falls back to an estimator when blocked, **silently changing chunk boundaries vs CI** (AGENTS.md §6) | **baked at build time** (`TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache`), load asserted in the offline smoke |
| Keys / `.env` | never baked (`.dockerignore`) | never baked — passed at run time |
| Runtime user | root | **non-root** (uid 10001) |
| Provenance | — | OCI labels + `--provenance/--sbom` attestations on published builds |
| Health | none | `HEALTHCHECK` via `raglab/container_healthcheck.py` (200 *or* 401 = alive — the API is token-gated) |

## The three ways to get and run the image

### A) CI publishes it (recommended)

Push a tag matching `v*` — the workflow
(`.github/workflows/docker-image.yml`) builds, offline-smokes and publishes:

```
ghcr.io/seif-bkh/raglab-service:<version>     # + :latest
GitHub Release <tag> → raglab-service-<version>.tar.gz (Linux/macOS targets)
                       raglab-service-<version>.zip    (Windows targets)
                       raglab-service-<version>.sha256  (one file, both artifacts)
```

Manual alternative: *Actions → RAGLab docker image → Run workflow* with
`push=true` publishes without a release tag (assets land in a release named
`image-<version>`). **Version policy:** the tag mirrors `SERVICE_VERSION` in
`raglab/service.py` (bump it → push → `git tag v<x.y.z>` → push the tag;
the workflow warns when they diverge).

First-time one-time step: after the first GHCR push the package is **private**
— flip it to public once (repo → Packages → *raglab-service* → Package
settings → Change visibility), or run it behind an authenticated pull.

Run it:

```bash
docker pull ghcr.io/seif-bkh/raglab-service:1.2.5
docker run -d --name raglab -p 8000:8000 --env-file raglab/.env \
  -e RAGLAB_SERVICE_TOKEN=your-long-random-token \
  -v raglab-index:/app/raglab/chroma_db \
  -v raglab-embed-cache:/app/raglab/caches \
  -v raglab-documents:/app/raglab/documents \
  ghcr.io/seif-bkh/raglab-service:1.2.5
# API docs: http://localhost:8000/docs
```

### B) Export a portable archive and carry it to an offline machine

On any machine with Docker (needs internet once, for the build). The exporters
emit **both packagings of the same docker-save tar** — `docker load` is
cross-platform, so only the wrapper differs per target:

| File | Target machines | Load with |
|---|---|---|
| `raglab-service-<v>.tar.gz` | Linux / macOS | `docker/load-image.sh <file>` (or `gunzip -c … \| docker load`) |
| `raglab-service-<v>.zip` | Windows | `.\docker\load-image.ps1 <file>` / `load-image.bat` (or extract, `docker load -i`) |
| `raglab-service-<v>.sha256` | both | one sidecar; the load helpers verify it automatically |

```bash
docker/save-image.sh            # Linux/macOS build host — produces .tar.gz (+ .zip when 'zip' is installed)
docker\save-image.ps1           # Windows build host (Docker Desktop) — produces .zip (+ .tar.gz when bsdtar is present)
```

Then on the target machine (no internet at any step):

```bash
# Linux/macOS target:
docker/load-image.sh raglab-service_1.2.5.tar.gz     # verifies sha256, docker load, prints the run command
```

```powershell
# Windows target:
.\docker\load-image.ps1 .\raglab-service_1.2.5.zip   # verifies sha256, docker load, prints the run command
```

The helpers print the exact `docker run` line; the reference after load is
always `ghcr.io/seif-bkh/raglab-service:<version>`.

### C) Compose, image-only (no build, no dev Dockerfile)

```bash
cp raglab/.env.example raglab/.env       # keys + RAGLAB_SERVICE_TOKEN
RAGLAB_VERSION=1.2.5 docker compose -f docker-compose.prod.yml up -d
# air-gapped (image came from a tarball): RAGLAB_PULL_POLICY=never
```

`docker-compose.prod.yml` never builds; `pull_policy: missing` (default)
only reaches a registry when the image is absent.

## Operating notes

- **Same env contract as the dev stack** — `RAGLAB_ALLOW_PROFILE_SWITCH`
  (default `1`; pin `0` for a frozen deployment), `RAGLAB_CORS_ORIGINS`,
  `RAGLAB_DATA_DIRS` (default `/app/docs`, the corpus baked into the image;
  pushed documents always join via `/app/raglab/documents`), and every key
  from `raglab/.env.example`.
- **State = the three volumes** (`chroma_db/`, `caches/`, `documents/`) —
  recreate the container freely, the index survives. Changing the embedding
  provider/model or chunking creates a *new* profile collection; a stale
  fingerprint is refused with `409 retrieval_refused` → `/ingest?reset=true`
  (same semantics as everywhere else).
- **Endpoint freeze applies** (`AGENTS.md` §2.7): the image's HTTP surface is
  exactly `CONTRACT.md`. A refusal is HTTP 200 with `status=refused`; while an
  ingest job runs, data endpoints answer `409 ingest_in_progress`.
- **Security**: set `RAGLAB_SERVICE_TOKEN` (the banking app sends it as
  `X-Service-Token`), keep the container behind your gateway, and keep the
  standing disclaimer in mind — this lab is *not production-hardened for a
  banking service; supervised pilot standing* (`README.md`, `SERVICE.md`).
- Bind-mounting instead of named volumes? The runtime user is uid **10001**
  (`chown 10001:10001` the host dirs first).
- Size: a few hundred MB (slim base + chromadb/onnxruntime wheels).

## Verifying a running container

```bash
docker ps                            # STATUS column must read (healthy)
curl -s localhost:8000/health -H "X-Service-Token: $TOKEN" | jq .index.count
python raglab/local_front.py --status --base-url http://localhost:8000
python raglab/local_front.py --smoke  --base-url http://localhost:8000
```
