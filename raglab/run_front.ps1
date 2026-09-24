#Requires -Version 5.1
<#
.SYNOPSIS
  The RAGLab console (local_front.py) against the Windows-run service.

.DESCRIPTION
  local_front.py is a pure-stdlib HTTP client - the app.py console experience
  driven over REST (menus, keys, ingest, chat, evaluate, ...). The service must
  be running (run_server.ps1) - everything else is handled here:

    * forces a UTF-8 console so the Arabic/French corpus text renders correctly
    * prefers the venv python, falls back to any python on PATH (stdlib only)
    * hands the service token to the front (resolved like run_server.ps1:
      -Token > environment > raglab\.env) so a protected service "just works"

  Every extra argument is passed straight through to local_front.py.

.EXAMPLE
  .\run_front.ps1                      the menu console (app.py parity, over REST)
  .\run_front.ps1 --status             doctor report, no prompts
  .\run_front.ps1 --ingest             build the index (needs NVIDIA_API_KEY)
  .\run_front.ps1 --smoke              endpoint smoke suite
  .\run_front.ps1 --ask "What is Murabaha?"
#>
[CmdletBinding()]
param(
    # Service base URL (default: env RAGLAB_SERVICE_URL or http://localhost:8000).
    [string]$BaseUrl,

    # X-Service-Token the service requires (if run_server.ps1 was started with one).
    [string]$Token,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$FrontArgs
)

$ProjectDir = $PSScriptRoot
$RepoRoot   = Split-Path -Parent $ProjectDir

# UTF-8 console: the corpus is Arabic/French/English and is printed verbatim.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

if (-not $BaseUrl) {
    $BaseUrl = if ($env:RAGLAB_SERVICE_URL) { $env:RAGLAB_SERVICE_URL } else { "http://localhost:8000" }
}

# Any Python works (stdlib only); prefer the venv for consistency.
$python = $null
foreach ($candidate in @("$RepoRoot\.venv\Scripts\python.exe", "$ProjectDir\.venv\Scripts\python.exe")) {
    if (Test-Path $candidate) { $python = $candidate; break }
}
if (-not $python) { $python = "python" }

# Token: the front must echo the service's shared secret on every request.
if (-not $Token) { $Token = $env:RAGLAB_SERVICE_TOKEN }
if (-not $Token) { $Token = $env:RAGLAB_TOKEN }
if (-not $Token) {
    $envFile = Join-Path $ProjectDir ".env"
    if (Test-Path $envFile) {
        foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
            if ($line -match "^\s*RAGLAB_SERVICE_TOKEN\s*=\s*(.+?)\s*$" -or
                $line -match "^\s*RAGLAB_TOKEN\s*=\s*(.+?)\s*$") {
                $Token = $Matches[1].Trim('"').Trim("'")
                break
            }
        }
    }
}
if ($Token) { $env:RAGLAB_SERVICE_TOKEN = $Token }

# Windows PowerShell 5.1 native-call quoting: re-quote args containing spaces.
$quoted = foreach ($arg in @($FrontArgs)) {
    if ($arg -match '[\s"]') { '"' + ($arg -replace '"', '\"') + '"' } else { $arg }
}

& $python (Join-Path $ProjectDir "local_front.py") --base-url $BaseUrl @quoted
exit $LASTEXITCODE
