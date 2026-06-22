#!/usr/bin/env bash
set -euo pipefail

PROJECT="/DATA/castle/projects/make_llm"
PID_FILE="$PROJECT/results/run_logs/llama_cpp/v0_0_9_14p5b_pt_v2_cpt_sft_dpo_q4_k_m_8622_2026-06-15.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "pid file not found: $PID_FILE"
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "stopped llama-server pid $PID"
else
  echo "llama-server pid $PID is not running"
fi
