#!/usr/bin/env bash
# load-image.sh — verify + docker load a RAGLab image export on Linux/macOS,
# then print the exact run command. Handles .tar.gz, .zip and plain .tar.
# Needs zero downloads: everything happens locally.
#
#   docker/load-image.sh dist/raglab-service_1.2.5.tar.gz
#   docker/load-image.sh dist/raglab-service_1.2.5.zip
set -euo pipefail

FILE="${1:-}"
[ -n "$FILE" ] || { echo "usage: docker/load-image.sh <raglab-service_V.tar.gz|.zip|.tar>"; exit 1; }
[ -f "$FILE" ] || { echo "no such file: $FILE"; exit 1; }

# integrity check when the sidecar travels with the artifact
# (one raglab-service_<v>.sha256 covers every artifact of that version)
NAME="$(basename "$FILE")"; DIR="$(dirname "$FILE")"
STEM="${NAME%.tar.gz}"; STEM="${STEM%.zip}"; STEM="${STEM%.tar}"
SHA="$DIR/$STEM.sha256"
if [ -f "$SHA" ]; then
    ( cd "$DIR" && grep "  $NAME\$" "$(basename "$SHA")" | sha256sum -c - )
else
    echo "[load-image] no $STEM.sha256 next to the file - skipping integrity check"
fi

case "$FILE" in
    *.tar.gz)
        OUT="$(gunzip -c "$FILE" | docker load)"
        ;;
    *.zip)
        command -v unzip >/dev/null 2>&1 || { echo "'unzip' not installed (apt install unzip / dnf install unzip)"; exit 1; }
        TMP="$(mktemp -d)"
        trap 'rm -rf "$TMP"' EXIT
        unzip -q "$FILE" -d "$TMP"
        OUT="$(docker load -i "$TMP"/*.tar)"
        ;;
    *)
        OUT="$(docker load -i "$FILE")"
        ;;
esac

echo "$OUT"
REF="$(printf '%s\n' "$OUT" | sed -n 's/^Loaded image: //p' | head -1)"
[ -n "$REF" ] || REF="$(printf '%s\n' "$OUT" | sed -n 's/^Loaded image ID: //p' | head -1)"
[ -n "$REF" ] || { echo "[load-image] loaded, but could not read the image reference"; exit 0; }

cat <<EOF

[load-image] ready: $REF
Run it (state persists in the three named volumes):

  docker run -d --name raglab -p 8000:8000 \\
    -e NVIDIA_API_KEY=\$NVIDIA_API_KEY -e XKIRO_API_KEY=\$XKIRO_API_KEY \\
    -e RAGLAB_SERVICE_TOKEN=your-long-random-token \\
    -v raglab-index:/app/raglab/chroma_db \\
    -v raglab-embed-cache:/app/raglab/caches \\
    -v raglab-documents:/app/raglab/documents \\
    $REF

or with compose:  RAGLAB_VERSION="${REF##*:}" RAGLAB_PULL_POLICY=never \\
    docker compose -f docker-compose.prod.yml up -d
API docs once it is up: http://localhost:8000/docs
EOF
