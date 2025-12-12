#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the FSDP2 GSM8K example on a 4-GPU node:
  - GPU0: vLLM server
  - GPUs1-3: FSDP2 training (torchrun, 3 ranks)

Usage:
  examples/fsdp2_training/run_example.sh [both|server|train]

Environment overrides (optional):
  MODEL=Qwen/Qwen2.5-7B-Instruct
  VLLM_HOST=127.0.0.1
  VLLM_PORT=8000
  VLLM_MAX_NUM_SEQS=32

  TRAIN_LIMIT=256
  TRAIN_STEPS=50
  TRAIN_CONCURRENCY=4
  TRAIN_BATCH_SIZE=1
  TRAIN_GROUP_SIZE=8
  TRAIN_LOG_LEVEL=INFO
  TRAIN_LOGGER=print

  UV_CACHE_DIR=/path/to/writable/cache
EOF
}

mode="${1:-both}"
case "$mode" in
  both|server|train) ;;
  -h|--help|help) usage; exit 0 ;;
  *) echo "unknown mode: $mode" >&2; usage; exit 2 ;;
esac

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$root"

export PYTHONPATH="${PYTHONPATH:-.}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.uv_cache}"
mkdir -p "$UV_CACHE_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"

TRAIN_LIMIT="${TRAIN_LIMIT:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-50}"
TRAIN_CONCURRENCY="${TRAIN_CONCURRENCY:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-8}"
TRAIN_LOG_LEVEL="${TRAIN_LOG_LEVEL:-INFO}"
TRAIN_LOGGER="${TRAIN_LOGGER:-print}"

server_pid=""

start_server() {
  echo "[run_example] starting vLLM on GPU0 (port=$VLLM_PORT) ..."
  CUDA_VISIBLE_DEVICES=0 uv run python -m ludic.inference.vllm_server \
    --model "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" &
  server_pid="$!"

  # Wait for /health
  echo "[run_example] waiting for vLLM /health ..."
  for _ in $(seq 1 120); do
    if curl -fsS "http://${VLLM_HOST}:${VLLM_PORT}/health" >/dev/null 2>&1; then
      echo "[run_example] vLLM is healthy."
      return 0
    fi
    sleep 1
  done
  echo "[run_example] vLLM did not become healthy in time." >&2
  return 1
}

stop_server() {
  if [[ -n "$server_pid" ]]; then
    echo "[run_example] stopping vLLM (pid=$server_pid) ..."
    kill "$server_pid" >/dev/null 2>&1 || true
    wait "$server_pid" >/dev/null 2>&1 || true
  fi
}

run_train() {
  echo "[run_example] starting training on GPUs 1,2,3 ..."
  CUDA_VISIBLE_DEVICES=1,2,3 uv run torchrun --nproc_per_node=3 \
    examples/fsdp2_training/train_gsm8k_fsdp2.py \
      --model "$MODEL" \
      --vllm-host "$VLLM_HOST" \
      --vllm-port "$VLLM_PORT" \
      --limit "$TRAIN_LIMIT" \
      --train-steps "$TRAIN_STEPS" \
      --concurrency "$TRAIN_CONCURRENCY" \
      --batch-size "$TRAIN_BATCH_SIZE" \
      --group-size "$TRAIN_GROUP_SIZE" \
      --log-level "$TRAIN_LOG_LEVEL" \
      --logger "$TRAIN_LOGGER"
}

case "$mode" in
  server)
    start_server
    wait "$server_pid"
    ;;
  train)
    run_train
    ;;
  both)
    trap stop_server EXIT INT TERM
    start_server
    run_train
    ;;
esac

