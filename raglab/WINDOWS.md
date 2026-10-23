# RAGLab on Windows (no Docker)

Run the RAGLab HTTP service (`service.py`) and the REST console (`local_front.py`)
natively on Windows — same port, same endpoints, same corpus, same state as the
Docker/compose stack, without installing Docker. The server is the piece the
banking application links against; `local_front.py` is an optional client used
to operate and verify it.

Everything here is additive: no endpoint, schema or behavior changes (the HTTP
contract stays exactly as documented in `CONTRACT.md`).

## What you need

- Windows 10/11, 64-bit
- Python 3.11+ from [python.org](https://www.python.org/downloads/) — during
  install, tick **"Add python.exe to PATH"** (that also installs the `py`
  launcher the setup script uses). Every dependency ships prebuilt Windows
  wheels (chromadb, tiktoken, uvicorn, …) — **no compiler, no WSL needed.**
- This repo cloned (e.g. `C:\RAGLab`)

## Quick start

Open **PowerShell** in the `raglab` folder (or use the `.bat` twins from CMD /
double-click):

```powershell
cd C:\RAGLab\raglab
.\setup.ps1            # one-time: finds Python, venv, installs deps, creates .env
```

Edit `raglab\.env` that was just created — paste `NVIDIA_API_KEY` and
`XKIRO_API_KEY` (the two keys the supported pipeline needs). Then, window 1:

```powershell
.\run_server.ps1       # the service -> http://localhost:8000/docs  (Ctrl+C to stop)
```

Window 2:

```powershell
.\run_front.ps1 --ingest     # first time only: embed the corpus (needs NVIDIA_API_KEY)
.\run_front.ps1              # the console: status, keys, ingest, search, answer, chat
.\run_front.ps1 --status     # one-shot doctor report
.\run_front.ps1 --smoke      # endpoint smoke suite (state-aware; the full contract)
.\run_front.ps1 --ask "What is Murabaha?"
```

Interactive API docs (Swagger UI): **http://localhost:8000/docs** — same as Docker.

## The scripts

| Script | What it is | Docker equivalent |
|---|---|---|
| `setup.ps1` / `setup.bat` | One-time setup: Python → venv (`raglab\.venv`) → `requirements-service.txt` → `raglab\.env` | the image build (`Dockerfile`) |
| `run_server.ps1` / `run_server.bat` | Runs `uvicorn service:app` on `0.0.0.0:8000` with the compose env contract | `docker compose up` |
| `run_front.ps1` / `run_front.bat` | `local_front.py` against the service (UTF-8 console, token auto-picked from `.env`) | running the front against the compose stack |

`run_server.ps1` parameters (all optional):

| Parameter | Default | Meaning |
|---|---|---|
| `-ListenHost` | `0.0.0.0` | Bind address. Use `127.0.0.1` if only this machine may call the service |
| `-Port` | `8000` | Listening port |
| `-Token` | env / `.env` | `RAGLAB_SERVICE_TOKEN` — every client must send it as `X-Service-Token` |
| `-AllowProfileSwitch` | `1` | `0` freezes the profile (blocks `POST /profile`) |
| `-CorsOrigins` | service default (`*`) | Comma-separated allowed browser origins |
| `-DataDirs` | `..\docs + data\` | Corpus dirs, if you keep documents elsewhere |
| `-WithHarness` (setup only) | off | Also install `requirements-harness.txt` (eval tooling, not needed to serve) |

Any profile knob from `SERVICE.md` (`RAGLAB_EMBEDDING_PROVIDER`,
`RAGLAB_TOP_K`, `RAGLAB_CHUNKING_MODE`, …) can also be set in `raglab\.env` —
`config.py` loads it exactly like compose's `env_file`.

## Docker ↔ Windows mapping

| Docker / compose | On Windows with these scripts |
|---|---|
| `docker build` (pip layer) | `setup.ps1` |
| `env_file: raglab/.env` | `config.py` loads `raglab\.env` itself; `-Token` etc. on `run_server.ps1` |
| port mapping `8000:8000` | `-Port 8000` (already the default) |
| volume `raglab-index` | folder `raglab\chroma_db\` — persists by itself, survives restarts |
| volume `raglab-embed-cache` | `embeddings_cache_app_*.json` next to the code (re-ingests stay cheap) |
| volume `raglab-documents` | folder `raglab\documents\` |
| `restart: unless-stopped` | re-run the script, or register it in Task Scheduler (below) |
| `docker compose logs` | the console the server runs in |

All of those folders are already in `.gitignore` — nothing stateful gets committed.

## Linking the banking application (the part that matters)

The service on Windows is **byte-for-byte the same HTTP surface** as the
container — the frozen contract in `CONTRACT.md` applies unchanged:

1. **Base URL**: `http://<windows-host>:8000`. The banking app calls it like
   any microservice (`POST /answer`, `POST /search`, `GET /health`, …).
2. **Protect it**: start with a token and give the same value to the banking
   app, which sends it on every request as the `X-Service-Token` header:
   ```powershell
   .\run_server.ps1 -Token 'paste-a-long-random-string'
   ```
   `.\run_front.ps1` picks the token up from `raglab\.env` (or `-Token`)
   automatically. Without a token, anyone who can reach the port can call
   `/ingest`, `/profile` and `/keys`.
3. **Bind address**: `0.0.0.0` (default) accepts connections from other
   machines — Windows Firewall prompts on first run; allow only the networks
   the banking app needs. Same-machine-only: `-ListenHost 127.0.0.1`.
4. **CORS**: if a browser-based banking UI calls the service directly, pass
   `-CorsOrigins "https://bank.local,https://gateway.bank.local"` instead of
   the `*` default. Server-to-server calls don't use CORS.
5. **Auth is a shared secret, not user auth** — same standing as documented in
   `SERVICE.md`: keep the service behind your gateway/auth sidecar for a real
   deployment, and note the lab's standing disclaimer: *not production-ready
   for a banking service; suitable for a supervised pilot with approved
   nonconfidential documents.*
6. **One writer**: run **one** server process per index folder. Two servers
   sharing `raglab\chroma_db\` will fight over the ChromaDB lock; while an
   ingest runs, `/search` and `/answer` return `409 ingest_in_progress` (poll
   `GET /ingest/status`) — this is the contract, unchanged from Docker.
7. **Refusals are HTTP 200** (`status=refused` + `reason`) — test your error
   handling against `--smoke`, not just against happy-path answers.

## Autostart with Windows (optional)

To start the server at logon (server machines: use "whether user is logged on
or not" in the UI):

```
schtasks /create /tn "RAGLab service" /sc onlogon /rl highest ^
  /tr "powershell -NoProfile -ExecutionPolicy Bypass -File C:\RAGLab\raglab\run_server.ps1 -Token YOURTOKEN"
```

For a true Windows service (auto-restart, no console), wrap the same
`run_server.ps1` with a service manager such as NSSM or `sc.exe` + a watchdog
— the service itself is a plain console process and needs no changes.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `scripts disabled on this system` / execution-policy error | Use the `.bat` files (they bypass the policy), or `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Python 3.11+ not found` | Install from python.org with **Add python.exe to PATH**; reopen PowerShell |
| Port already in use | `netstat -ano | findstr :8000`, or run with `-Port 8001` + `.\run_front.ps1 -BaseUrl http://localhost:8001` |
| Windows Firewall prompt on first start | Expected when binding `0.0.0.0` — allow Private networks for LAN access |
| Arabic/French text shows as `?????` in the console | The scripts already force UTF-8; use **Windows Terminal** (or PowerShell 7) for a font that renders Arabic |
| First question is slow | Normal: tiktoken downloads its tokenizer on first use; first xKiro `/answer` does the live free-price check |
| Provider calls fail through a corporate proxy | Set `HTTPS_PROXY` before starting: `$env:HTTPS_PROXY="http://proxy:3128"` in the server window |
| Accidentally started two servers | Keep one; the second one errors on the ChromaDB lock — Ctrl+C it |

## Uninstall / reset

Delete, any time (all gitignored): `raglab\.venv`, `raglab\chroma_db\`,
`raglab\documents\`, the `*_cache*.json` files, and `raglab\.env`.
`.\setup.ps1` rebuilds everything from scratch.
