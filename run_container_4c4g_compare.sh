#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-whisper-rust-ort:4c4g}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.container}"
CPUSET="${CPUSET:-0-3}"

docker build -t "${IMAGE}" -f "${ROOT_DIR}/${DOCKERFILE}" "${ROOT_DIR}"

docker run --rm \
  --cpuset-cpus "${CPUSET}" \
  --memory 4g \
  --memory-swap 4g \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  bash -lc "scripts/run_container_4c4g_inner.sh"
