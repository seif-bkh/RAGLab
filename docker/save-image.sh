#!/usr/bin/env bash
# save-image.sh — build the RAGLab production image (Dockerfile.prod, multi-stage)
# and export it as portable artifacts for BOTH target platforms:
#   dist/raglab-service_<v>.tar.gz   -> Linux/macOS machines (gunzip -c | docker load)
#   dist/raglab-service_<v>.zip      -> Windows machines (double-click extract -> docker load)
#   dist/raglab-service_<v>.<fmt>.sha256
# The image inside is the SAME docker-save tar in both files (docker load is
# cross-platform) — only the packaging differs. Guide: raglab/PROD_IMAGE.md.
#
#   docker/save-image.sh            # version = SERVICE_VERSION from raglab/service.py
#   docker/save-image.sh 1.2.6      # explicit version/tag
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

VERSION="${1:-$(sed -n 's/^SERVICE_VERSION = "\([^"]*\)".*/\1/p' raglab/service.py | head -1)}"
[ -n "$VERSION" ] || { echo "no version given and SERVICE_VERSION not found in raglab/service.py"; exit 1; }
IMAGE="ghcr.io/seif-bkh/raglab-service"
VCS_REF="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "[save-image] building $IMAGE:$VERSION (+ :latest) from Dockerfile.prod ..."
docker build -f Dockerfile.prod \
    --build-arg SERVICE_VERSION="$VERSION" --build-arg VCS_REF="$VCS_REF" \
    -t "$IMAGE:$VERSION" -t "$IMAGE:latest" .

mkdir -p dist
BASE="raglab-service_${VERSION}"
TAR="dist/${BASE}.tar"
GZ="dist/${BASE}.tar.gz"
ZIP="dist/${BASE}.zip"

echo "[save-image] exporting $IMAGE:$VERSION ..."
docker save "$IMAGE:$VERSION" -o "$TAR"
gzip -c -1 "$TAR" > "$GZ"
if command -v zip >/dev/null 2>&1; then
    ( cd dist && zip -q -1 "${BASE}.zip" "${BASE}.tar" && rm -f "${BASE}.tar" )
    ( cd dist && sha256sum "${BASE}.tar.gz" "${BASE}.zip" > "${BASE}.sha256" )
    ARTIFACTS="$GZ $ZIP"
else
    echo "[save-image] NOTE: 'zip' not installed here — shipping the .tar.gz only"
    echo "             (CI and docker\\save-image.ps1 produce the .zip for Windows targets)"
    rm -f "$TAR"
    ( cd dist && sha256sum "${BASE}.tar.gz" > "${BASE}.sha256" )
    ARTIFACTS="$GZ"
fi

echo
echo "[save-image] done:"
for f in $ARTIFACTS "dist/${BASE}.sha256"; do echo "  $f ($(du -h "$f" | cut -f1))"; done
cat <<EOF

Ship the files to the target machine, then load+run with the helpers:
  Linux/macOS:  docker/load-image.sh $GZ
  Windows:      .\\docker\\load-image.ps1 $ZIP      (or load-image.bat)

Manual equivalent (Linux):  sha256sum -c ${BASE}.sha256 && gunzip -c ${BASE}.tar.gz | docker load
Manual equivalent (Windows): expand the .zip, then: docker load -i ${BASE}.tar
Image reference after load: $IMAGE:$VERSION
EOF
