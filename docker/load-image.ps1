#Requires -Version 5.1
<#
.SYNOPSIS
  Verify + docker load a RAGLab image export on Windows, then print the exact
  run command. Handles .zip (double-click-friendly), .tar.gz (bsdtar) and
  plain .tar. Needs zero downloads: everything happens locally.

.EXAMPLE
  .\docker\load-image.ps1 .\dist\raglab-service_1.2.5.zip
  .\docker\load-image.ps1 .\dist\raglab-service_1.2.5.tar.gz
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$File
)

$ErrorActionPreference = "Stop"
$File = (Resolve-Path $File).Path

# integrity check when the sidecar travels with the artifact
# (one raglab-service_<v>.sha256 covers every artifact of that version)
$name = Split-Path $File -Leaf
$stem = $name -replace '\.tar\.gz$', '' -replace '\.zip$', '' -replace '\.tar$', ''
$shaFile = Join-Path (Split-Path $File -Parent) "$stem.sha256"
if (Test-Path $shaFile) {
    $entry = Get-Content $shaFile | Where-Object { $_ -match "\s$([regex]::Escape($name))$" } | Select-Object -First 1
    if ($entry) {
        $expected = ($entry -split "\s+")[0].ToLower()
        $actual   = (Get-FileHash $File -Algorithm SHA256).Hash.ToLower()
        if ($expected -ne $actual) { throw "checksum mismatch!`n  expected: $expected`n  actual:   $actual" }
        Write-Host "[load-image] sha256 verified"
    } else {
        Write-Host "[load-image] $name not listed in $stem.sha256 - skipping integrity check"
    }
} else {
    Write-Host "[load-image] no $stem.sha256 next to the file - skipping integrity check"
}

$out = $null
if ($File -like "*.zip") {
    $tmp = Join-Path $env:TEMP ("raglab-load-" + [guid]::NewGuid().ToString("N"))
    Expand-Archive -LiteralPath $File -DestinationPath $tmp
    $tar = Get-ChildItem $tmp -Filter *.tar | Select-Object -First 1
    if (-not $tar) { Remove-Item -Recurse -Force $tmp; throw "no .tar inside $File" }
    $out = docker load -i $tar.FullName
    Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
} elseif ($File -like "*.tar.gz") {
    $out = tar -xOf $File | docker load        # bsdtar -O streams to stdout
} else {
    $out = docker load -i $File
}
if ($LASTEXITCODE -ne 0) { throw "docker load failed (exit $LASTEXITCODE)`n$out" }
$out | ForEach-Object { Write-Host $_ }

$ref = ($out | Select-String "Loaded image: (.+)$" | Select-Object -First 1)
$ref = if ($ref) { $ref.Matches.Groups[1].Value } else { $null }
if (-not $ref) {
    $refId = ($out | Select-String "Loaded image ID: (.+)$" | Select-Object -First 1)
    $ref = if ($refId) { $refId.Matches.Groups[1].Value } else { $null }
}
if (-not $ref) { Write-Host "[load-image] loaded, but could not read the image reference"; exit 0 }

Write-Host ""
Write-Host "[load-image] ready: $ref"
Write-Host "Run it (state persists in the three named volumes):"
Write-Host ""
Write-Host "  docker run -d --name raglab -p 8000:8000 ``"
Write-Host "    -e NVIDIA_API_KEY=`$env:NVIDIA_API_KEY -e XKIRO_API_KEY=`$env:XKIRO_API_KEY ``"
Write-Host "    -e RAGLAB_SERVICE_TOKEN=your-long-random-token ``"
Write-Host "    -v raglab-index:/app/raglab/chroma_db ``"
Write-Host "    -v raglab-embed-cache:/app/raglab/caches ``"
Write-Host "    -v raglab-documents:/app/raglab/documents ``"
Write-Host "    $ref"
Write-Host ""
Write-Host "or with compose:  `$env:RAGLAB_VERSION='$($ref -replace '.*:', '')'; `$env:RAGLAB_PULL_POLICY='never'; docker compose -f docker-compose.prod.yml up -d"
Write-Host "API docs once it is up: http://localhost:8000/docs"
