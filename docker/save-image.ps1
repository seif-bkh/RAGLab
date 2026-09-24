#Requires -Version 5.1
<#
.SYNOPSIS
  Build the RAGLab production image (Dockerfile.prod, multi-stage) and export
  it as portable artifacts for BOTH target platforms:
    dist\raglab-service_<v>.zip      -> Windows machines (the native one)
    dist\raglab-service_<v>.tar.gz   -> Linux/macOS machines (when bsdtar is present)
  The image inside is the SAME docker-save tar in both files (docker load is
  cross-platform) - only the packaging differs. Guide: raglab\PROD_IMAGE.md.

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
    $base = "raglab-service_$Version"
    $tar  = "dist\$base.tar"
    $zip  = "dist\$base.zip"
    $gz   = "dist\$base.tar.gz"

    Write-Host "[save-image] exporting ${image}:$Version ..."
    docker save "${image}:$Version" -o $tar
    if ($LASTEXITCODE -ne 0) { throw "docker save failed (exit $LASTEXITCODE)" }

    # Windows-native artifact: .zip (Compress-Archive, no external tools needed)
    Compress-Archive -LiteralPath $tar -DestinationPath $zip -Force
    $artifacts = @($zip)

    # Linux/macOS artifact: .tar.gz (only with bsdtar, shipping since Win10 1803)
    if (Get-Command tar -ErrorAction SilentlyContinue) {
        tar -czf $gz -C dist "$base.tar" 2>$null
        if ($LASTEXITCODE -eq 0) { $artifacts += $gz } else { Remove-Item $gz -ErrorAction SilentlyContinue }
    } else {
        Write-Host "[save-image] NOTE: tar.exe not found - .zip only (Linux targets can be served by CI or docker/save-image.sh)"
    }
    Remove-Item $tar -ErrorAction SilentlyContinue

    # one sidecar covering every artifact produced
    $lines = foreach ($a in $artifacts) { "{0}  {1}" -f (Get-FileHash $a -Algorithm SHA256).Hash.ToLower(), (Split-Path $a -Leaf) }
    $shaFile = "dist\$base.sha256"
    $lines | Out-File -Encoding ascii $shaFile
    $artifacts += $shaFile

    Write-Host ""
    Write-Host "[save-image] done:"
    foreach ($a in $artifacts) { Write-Host ("  {0} ({1:N1} MB)" -f $a, ((Get-Item $a).Length / 1MB)) }
    Write-Host ""
    Write-Host "Ship the files to the target machine, then load+run with the helpers:"
    Write-Host "  Windows:      .\docker\load-image.ps1 $zip   (or load-image.bat)"
    if (Test-Path $gz) { Write-Host "  Linux/macOS:  docker/load-image.sh $gz" }
    Write-Host "Image reference after load: ${image}:$Version"
} finally {
    Pop-Location
}
