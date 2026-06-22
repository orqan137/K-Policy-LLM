#!/usr/bin/env bash
set -euo pipefail

PROJECT="/DATA/castle/projects/make_llm"
SERVER_BIN="$PROJECT/source/llama.cpp/build/bin/llama-server"
MODEL="$PROJECT/results/04_exports/llama_cpp/v0_0_9_14p5b_pt_v2_cpt_sft_dpo_1epoch_mb8_2026-06-15_1228/make_llm-v0_0_9-14p5b-cpt-sft-dpo-q4_k_m.gguf"
LOG_DIR="$PROJECT/results/run_logs/llama_cpp"
LOG="$LOG_DIR/v0_0_9_14p5b_pt_v2_cpt_sft_dpo_q4_k_m_8622_2026-06-15.log"
PID_FILE="$LOG_DIR/v0_0_9_14p5b_pt_v2_cpt_sft_dpo_q4_k_m_8622_2026-06-15.pid"
HOST="${LLAMA_CPP_HOST:-127.0.0.1}"
PORT="${LLAMA_CPP_PORT:-8622}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$SERVER_BIN" ]]; then
  echo "missing llama-server: $SERVER_BIN" >&2
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "missing GGUF model: $MODEL" >&2
  exit 1
fi

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "llama-server already running: pid $(cat "$PID_FILE")"
  echo "log: $LOG"
  exit 0
fi

if ss -ltn "( sport = :$PORT )" | tail -n +2 | grep -q .; then
  echo "port $PORT is already in use" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
setsid "$SERVER_BIN" \
  --model "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --ctx-size 4096 \
  --gpu-layers all \
  --device CUDA0 \
  --threads 16 \
  --batch-size 512 \
  --ubatch-size 128 \
  --flash-attn auto \
  --parallel 1 \
  --no-webui \
  > "$LOG" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "started llama-server pid $(cat "$PID_FILE") on ${HOST}:${PORT}"
echo "model: $MODEL"
echo "log: $LOG"
