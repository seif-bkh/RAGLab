#Requires -Version 5.1
<#
.SYNOPSIS
  RAGLab on Windows, no Docker: one-time setup.

.DESCRIPTION
  The Windows equivalent of the Dockerfile's pip layer + "./local.sh setup":
    * finds a Python 3.11+ interpreter (py launcher or PATH)
    * creates a virtualenv (raglab\.venv; an existing repo-root .venv wins,
      same precedence as local.sh)
    * installs requirements-service.txt (the server) -- the only deps the
      service and local_front.py need
    * creates raglab\.env from .env.example when missing

  Run once, then:
    .\run_server.ps1   window 1: the HTTP service -> http://localhost:8000/docs
    .\run_front.ps1    window 2: the console over REST
  From CMD instead of PowerShell: setup.bat / run_server.bat / run_front.bat.

  Full guide: raglab\WINDOWS.md
#>
[CmdletBinding()]
param(
    # Also install requirements-harness.txt (the evaluation tooling; NOT
    # needed to run the service or the front).
    [switch]$WithHarness
)

$ErrorActionPreference = "Stop"
$ProjectDir = $PSScriptRoot                       # ...\raglab
$RepoRoot   = Split-Path -Parent $ProjectDir      # the clone root

function Write-Step([string]$Text) { Write-Host "[setup] $Text" }

# -- 1. find a Python >= 3.11 (the version the Docker image and CI use) --------
$found = $null
foreach ($candidate in @(@("py", @("-3.11")), @("py", @("-3")),
                         @("python", @()), @("python3", @()))) {
    $exe, $argList = $candidate[0], $candidate[1]
    try {
        $version = & $exe @argList -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')" 2>$null
    } catch { continue }
    if ($LASTEXITCODE -eq 0 -and "$version" -match "^3\.(\d+)$" -and [int]$Matches[1] -ge 11) {
        $found = @{ Exe = $exe; Args = $argList; Version = "$version".Trim() }
        break
    }
}
if (-not $found) {
    throw "Python 3.11+ not found. Install it from https://www.python.org/downloads/ " +
          "(tick 'Add python.exe to PATH' so the py launcher is available), then re-run .\setup.ps1"
}
$pythonExe  = $found.Exe
$pythonArgs = @($found.Args)
Write-Step "Python $($found.Version) via: $pythonExe $($pythonArgs -join ' ')"

# -- 2. virtualenv: an existing one wins (repo-root first, like local.sh); -----
# --    a fresh one is created at raglab\.venv (matches the README flow).  -----
$venv = $null
foreach ($candidate in @("$RepoRoot\.venv", "$ProjectDir\.venv")) {
    if (Test-Path (Join-Path $candidate "Scripts\python.exe")) { $venv = $candidate; break }
}
if (-not $venv) {
    $venv = Join-Path $ProjectDir ".venv"
    Write-Step "creating virtualenv: $venv"
    & $pythonExe @pythonArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed (exit $LASTEXITCODE)" }
}
$venvPython = Join-Path $venv "Scripts\python.exe"
Write-Step "virtualenv: $venv"

# -- 3. dependencies (pinned versions; all ship prebuilt Windows x64 wheels) ---
Write-Step "installing requirements-service.txt (first run downloads a few hundred MB)..."
& $venvPython -m pip install --upgrade pip -q
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed (exit $LASTEXITCODE)" }
& $venvPython -m pip install -r (Join-Path $ProjectDir "requirements-service.txt")
if ($LASTEXITCODE -ne 0) { throw "pip install failed (exit $LASTEXITCODE)" }
if ($WithHarness) {
    Write-Step "installing requirements-harness.txt (-WithHarness)..."
    & $venvPython -m pip install -r (Join-Path $ProjectDir "requirements-harness.txt")
    if ($LASTEXITCODE -ne 0) { throw "pip install (harness) failed (exit $LASTEXITCODE)" }
}

# -- 4. raglab\.env (the compose env_file equivalent; loaded by config.py) -----
if (Test-Path (Join-Path $ProjectDir ".env")) {
    Write-Step "raglab\.env exists - leaving it alone"
} else {
    Copy-Item (Join-Path $ProjectDir ".env.example") (Join-Path $ProjectDir ".env")
    Write-Step "created raglab\.env - paste NVIDIA_API_KEY and XKIRO_API_KEY into it (can also be done later from the front, menu 3)"
}

Write-Host ""
Write-Step "done. Next:"
Write-Host "    1. edit $ProjectDir\.env     (NVIDIA_API_KEY, XKIRO_API_KEY)"
Write-Host "    2. .\run_server.ps1          the service  -> http://localhost:8000/docs"
Write-Host "    3. .\run_front.ps1           the console  (in a second window)"
Write-Host ""
Write-Step "first run on the service: .\run_front.ps1 --ingest   (embeds the corpus; needs NVIDIA_API_KEY)"
