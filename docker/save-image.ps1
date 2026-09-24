#Requires -Version 5.1
<#
.SYNOPSIS
  Build the RAGLab production image (Dockerfile.prod, multi-stage) and export
  it as a portable tarball - the artifact you hand to a machine that never
  touches the internet (docker load + docker run). Guide: raglab\PROD_IMAGE.md.

.EXAMPLE
  .\docker\save-image.ps1            # version = SERVICE_VERSION from raglab\service.py
  .\docker\save-image.ps1 -Version 1.2.6
#>
[CmdletBinding()]
param([string]$Version)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    if (-not $Version) {
        $match = Select-String -Path "raglab\service.py" -Pattern '^SERVICE_VERSION = "([^"]+)"' |
                 Select-Object -First 1
        if ($match) { $Version = $match.Matches.Groups[1].Value }
    }
    if (-not $Version) { throw "no -Version given and SERVICE_VERSION not found in raglab\service.py" }
    $image  = "ghcr.io/seif-bkh/raglab-service"
    $vcsRef = (git rev-parse --short HEAD 2>$null); if (-not $vcsRef) { $vcsRef = "unknown" }

    Write-Host "[save-image] building ${image}:$Version (+ :latest) from Dockerfile.prod ..."
    docker build -f Dockerfile.prod `
        --build-arg "SERVICE_VERSION=$Version" --build-arg "VCS_REF=$vcsRef" `
        -t "${image}:$Version" -t "${image}:latest" .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed (exit $LASTEXITCODE)" }

    New-Item -ItemType Directory -Force -Path dist | Out-Null
    $tar   = "dist\raglab-service_$Version.tar"
    $gzout = "$tar.gz"
    Write-Host "[save-image] exporting $gzout ..."
    docker save "${image}:$Version" -o $tar
    if ($LASTEXITCODE -ne 0) { throw "docker save failed (exit $LASTEXITCODE)" }
    tar -czf $gzout -C dist "raglab-service_$Version.tar"   # bsdtar ships with Windows 10/11
    if ($LASTEXITCODE -ne 0) { throw "gzip step failed (exit $LASTEXITCODE) - plain $tar is kept and loadable as-is" }
    Remove-Item $tar -ErrorAction SilentlyContinue
    $hash = (Get-FileHash $gzout -Algorithm SHA256).Hash.ToLower()
    "$hash  raglab-service_$Version.tar.gz" | Out-File -Encoding ascii "$gzout.sha256"
    $size = "{0:N1} MB" -f ((Get-Item $gzout).Length / 1MB)

    Write-Host ""
    Write-Host "[save-image] done: $gzout ($size)"
    Write-Host "Ship the .tar.gz (+ .sha256) to the target machine, then:"
    Write-Host "  tar -xzf raglab-service_$Version.tar.gz -O | docker load"
    Write-Host "  docker run -d --name raglab -p 8000:8000 ``"
    Write-Host "    -e NVIDIA_API_KEY=... -e XKIRO_API_KEY=... ``"
    Write-Host "    -e RAGLAB_SERVICE_TOKEN=your-long-random-token ``"
    Write-Host "    -v raglab-index:/app/raglab/chroma_db ``"
    Write-Host "    ${image}:$Version"
    Write-Host "or with compose (uses the loaded image, never pulls):"
    Write-Host "  `$env:RAGLAB_VERSION='$Version'; `$env:RAGLAB_PULL_POLICY='never'; docker compose -f docker-compose.prod.yml up -d"
} finally {
    Pop-Location
}
