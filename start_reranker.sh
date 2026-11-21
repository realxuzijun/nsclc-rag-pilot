#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/reranker_runtime"
binary="$(cat runtime.path)"
weights=bge-reranker-v2-m3-Q8_0.gguf
printf '%s  %s\n' a43c7c9b11a4c1517e5bf95151960e1621d1b72f7a493364b01e386cf1aaa1d3 "$weights" | sha256sum -c
exec "$binary" --host 127.0.0.1 --port "${RERANK_PORT:-11544}" --reranking \
  --model "$weights" --ctx-size 4096 --parallel 1 --n-gpu-layers 0 --threads 8

