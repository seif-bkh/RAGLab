#!/usr/bin/env bash
# save-image.sh — build the RAGLab production image (Dockerfile.prod, multi-stage)
# and export it as a portable tarball: the artifact you hand to a machine that
# never touches the internet (docker load + docker run). Guide: raglab/PROD_IMAGE.md.
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
OUT="dist/raglab-service_${VERSION}.tar.gz"
echo "[save-image] exporting $OUT ..."
docker save "$IMAGE:$VERSION" | gzip -1 > "$OUT"
( cd dist && sha256sum "raglab-service_${VERSION}.tar.gz" > "raglab-service_${VERSION}.tar.gz.sha256" )
SIZE="$(du -h "$OUT" | cut -f1)"

cat <<EOF

[save-image] done: $OUT ($SIZE)
Ship the .tar.gz (+ .sha256 next to it) to the target machine, then:

  sha256sum -c raglab-service_${VERSION}.tar.gz.sha256   # integrity check
  gunzip -c raglab-service_${VERSION}.tar.gz | docker load
  docker run -d --name raglab -p 8000:8000 \\
    -e NVIDIA_API_KEY=... -e XKIRO_API_KEY=... \\
    -e RAGLAB_SERVICE_TOKEN=your-long-random-token \\
    -v raglab-index:/app/raglab/chroma_db \\
    -v raglab-embed-cache:/app/raglab/caches \\
    -v raglab-documents:/app/raglab/documents \\
    $IMAGE:$VERSION

or with compose (uses the loaded image, never pulls):
  RAGLAB_VERSION=$VERSION RAGLAB_PULL_POLICY=never \\
    docker compose -f docker-compose.prod.yml up -d
EOF
