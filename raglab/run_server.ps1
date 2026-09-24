#Requires -Version 5.1
<#
.SYNOPSIS
  Runs the RAGLab HTTP service on Windows - no Docker.

.DESCRIPTION
  The local equivalent of "docker compose up": same env contract, same port,
  same corpus, same endpoints (raglab\CONTRACT.md is the frozen HTTP contract
  the banking application integrates against).

  Keys come from raglab\.env (config.py loads it = compose's env_file). State
  lives in plain folders (compose's volumes): chroma_db\ (the index),
  documents\ (pushed docs) and the *_cache*.json files next to the code.

  Ctrl+C stops the server. Logs go to this console.

.EXAMPLE
  .\run_server.ps1
  .\run_server.ps1 -Token 'long-random-string' -CorsOrigins 'https://gateway.bank.local'
  .\run_server.ps1 -ListenHost 127.0.0.1          # only this machine may call it
#>
[CmdletBinding()]
param(
    # Address to bind. 0.0.0.0 = reachable by other machines (like compose's
    # published port; Windows Firewall will prompt on first run). 127.0.0.1 =
    # this machine only.
    [string]$ListenHost = "0.0.0.0",

    [int]$Port = 8000,

    # Shared secret every client must send as X-Service-Token on EVERY request
    # (the banking app, local_front.py, curl). Resolved in this order:
    # -Token > environment > raglab\.env. Strongly recommended when the
    # banking app links in - without it /ingest, /profile and /keys are open
    # to anyone who can reach the port.
    [string]$Token,

    # compose ships "1" (the console/front and integrations need POST /profile).
    [ValidateSet("0", "1")]
    [string]$AllowProfileSwitch = "1",

    # Comma-separated allowed browser origins (RAGLAB_CORS_ORIGINS). Default:
    # the service's own default ("*"). Restrict it for the banking app.
    [string]$CorsOrigins,

    # Comma-separated corpus dirs (RAGLAB_DATA_DIRS). Default: ..\docs + data\
    # = the repo's docs\ folder, same corpus as the image's /app/docs.
    [string]$DataDirs
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot
$RepoRoot   = Split-Path -Parent $ProjectDir

# venv python (repo-root first, then raglab\.venv - setup.ps1's search order)
$python = $null
foreach ($candidate in @("$RepoRoot\.venv\Scripts\python.exe", "$ProjectDir\.venv\Scripts\python.exe")) {
    if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) {
    throw "no virtualenv found - run .\setup.ps1 first (or setup.bat from CMD)."
}

function Read-EnvValue([string]$Name) {
    $envFile = Join-Path $ProjectDir ".env"
    if (-not (Test-Path $envFile)) { return $null }
    foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
        if ($line -match ("^\s*" + [regex]::Escape($Name) + "\s*=\s*(.+?)\s*$")) {
            return $Matches[1].Trim('"').Trim("'")
        }
    }
    return $null
}

# The service reads raglab\.env itself (config.py), so provider keys need no
# export here. The token is surfaced explicitly because clients must send it
# back on every request as X-Service-Token.
$resolved = $Token
if (-not $resolved) { $resolved = $env:RAGLAB_SERVICE_TOKEN }
if (-not $resolved) { $resolved = $env:RAGLAB_TOKEN }
if (-not $resolved) { $resolved = Read-EnvValue "RAGLAB_SERVICE_TOKEN" }
if (-not $resolved) { $resolved = Read-EnvValue "RAGLAB_TOKEN" }
if ($resolved) { $env:RAGLAB_SERVICE_TOKEN = $resolved }

$env:RAGLAB_ALLOW_PROFILE_SWITCH = $AllowProfileSwitch
if ($CorsOrigins) { $env:RAGLAB_CORS_ORIGINS = $CorsOrigins }
if ($DataDirs)    { $env:RAGLAB_DATA_DIRS = $DataDirs }

$masked = if ($resolved) { $resolved.Substring(0, [Math]::Min(8, $resolved.Length)) } else { "" }
$corpus = if ($DataDirs) { $DataDirs } else { "..\docs + data\  (the repo's docs\ folder = the image's /app/docs)" }

Write-Host ""
Write-Host "  RAGLab service - Windows, no Docker -> http://localhost:$Port/docs"
Write-Host "  listen    : ${ListenHost}:$Port"
Write-Host "  corpus    : $corpus"
Write-Host "  state dirs: $ProjectDir\chroma_db , $ProjectDir\documents , caches next to the code"
Write-Host "  switching : POST /profile $(if ($AllowProfileSwitch -eq '1') { 'ENABLED (the front and integrations need it)' } else { 'disabled' })"
if ($resolved) {
    Write-Host "  token     : RAGLAB_SERVICE_TOKEN set ($masked...)  - clients send it as X-Service-Token"
} elseif ($AllowProfileSwitch -eq "1") {
    Write-Host "  token     : NOT SET - anyone who can reach this port can call /profile, /ingest, /keys."
    Write-Host "              set one and hand it to the banking app:"
    Write-Host "              .\run_server.ps1 -Token 'long-random-string'   (or RAGLAB_SERVICE_TOKEN=... in raglab\.env)"
} else {
    Write-Host "  token     : NOT SET (open - local/dev default)"
}
if (-not (Test-Path (Join-Path $ProjectDir ".env"))) {
    Write-Host "  note      : raglab\.env not found - run .\setup.ps1, or set keys later via the front (menu 3) / POST /keys."
}
Write-Host "  stop with Ctrl+C"
Write-Host ""

Push-Location $ProjectDir
try {
    & $python -m uvicorn service:app --host $ListenHost --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
