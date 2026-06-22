#!/usr/bin/env bash
set -euo pipefail

PROJECT="/DATA/castle/projects/make_llm"
PYTHON="/DATA/castle/envs/make_llm/.venv/bin/python"
MEMORY_GUARD="/DATA/castle/projects/workspace/scripts/memory_guard.sh"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y-%m-%d_%H%M)}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT}/results/02_training_runs/v0_0_9_base/checkpoint}"
TOKENIZER="${TOKENIZER:-${PROJECT}/data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model}"
POSTTRAIN_LAUNCHER="${POSTTRAIN_LAUNCHER:-${PROJECT}/scripts/run_v0_0_9_targeted_posttrain_cpt_sft_dpo_grpo.sh}"

TRAIN_GPU="${TRAIN_GPU:-0}"
EVAL_GPUS_CSV="${EVAL_GPUS_CSV:-1,2,3}"
MIN_TRAIN_GPU_FREE_GIB="${MIN_TRAIN_GPU_FREE_GIB:-42}"
MIN_EVAL_GPU_FREE_GIB="${MIN_EVAL_GPU_FREE_GIB:-12}"
MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB:-96}"

RUN_BASE_PUBLIC_EVAL="${RUN_BASE_PUBLIC_EVAL:-1}"
RUN_BASE_PUBLIC_EVAL_ASYNC="${RUN_BASE_PUBLIC_EVAL_ASYNC:-1}"
RUN_BASE_TARGET_EVAL="${RUN_BASE_TARGET_EVAL:-1}"
RUN_POSTTRAIN="${RUN_POSTTRAIN:-1}"
RUN_FINAL_PUBLIC_EVAL="${RUN_FINAL_PUBLIC_EVAL:-1}"
RUN_FINAL_TARGET_EVAL="${RUN_FINAL_TARGET_EVAL:-0}"

PUBLIC_EVAL_BENCHMARKS_CSV="${PUBLIC_EVAL_BENCHMARKS_CSV:-kmmlu_pro,kobalt_700,click}"
PUBLIC_EVAL_TIER="${PUBLIC_EVAL_TIER:-full_public_eval}"
PUBLIC_EVAL_MAX_EXAMPLES="${PUBLIC_EVAL_MAX_EXAMPLES:-}"
PUBLIC_EVAL_PAIR_BATCH_SIZE="${PUBLIC_EVAL_PAIR_BATCH_SIZE:-16}"
PUBLIC_EVAL_MAX_LENGTH="${PUBLIC_EVAL_MAX_LENGTH:-4096}"

TARGET_EVAL_DIR="${TARGET_EVAL_DIR:-${PROJECT}/data/05_datasets/posttrain/v0_0_9_posttrain_targeted_opendataloader_improvement_2026-06-15_0130/target_eval}"
TARGET_EVAL_LIMIT_PER_TASK="${TARGET_EVAL_LIMIT_PER_TASK:-}"
TARGET_EVAL_MAX_NEW_TOKENS="${TARGET_EVAL_MAX_NEW_TOKENS:-420}"

LOG_DIR="${PROJECT}/results/run_logs"
MANIFEST_DIR="${PROJECT}/results/run_manifests"
PIPE_LOG="${PIPE_LOG:-${LOG_DIR}/v0_0_9_posttrain_with_public_evals_${RUN_STAMP}.log}"
LAUNCH_MANIFEST="${LAUNCH_MANIFEST:-${MANIFEST_DIR}/v0_0_9_posttrain_with_public_evals_${RUN_STAMP}.launch.json}"
SUMMARY_PATH="${SUMMARY_PATH:-${PROJECT}/results/02_training_runs/v0_0_9_posttrain_with_public_evals_${RUN_STAMP}.summary.json}"
POSTTRAIN_PIPE_LOG="${POSTTRAIN_PIPE_LOG:-${LOG_DIR}/v0_0_9_targeted_posttrain_cpt_sft_dpo_grpo_${RUN_STAMP}.log}"
POSTTRAIN_GUARD_LOG="${POSTTRAIN_GUARD_LOG:-${LOG_DIR}/v0_0_9_targeted_posttrain_cpt_sft_dpo_grpo_${RUN_STAMP}.memory_guard.log}"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}" "$(dirname "${SUMMARY_PATH}")"
exec > >(tee -a "${PIPE_LOG}") 2>&1

cd "${PROJECT}"

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "${csv}"
}

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi -i "${gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

require_gpu_free() {
  local gpu="$1"
  local min_gib="$2"
  local free_mib
  free_mib="$(gpu_free_mib "${gpu}")"
  local min_mib=$((min_gib * 1024))
  echo "gpu=${gpu} free_mib=${free_mib} min_mib=${min_mib}"
  if [ "${free_mib}" -lt "${min_mib}" ]; then
    echo "GPU ${gpu} free memory is too low for this run." >&2
    exit 4
  fi
}

require_path() {
  local path="$1"
  if [ ! -e "${path}" ]; then
    echo "missing required path: ${path}" >&2
    exit 1
  fi
}

rel_or_abs() {
  local path="$1"
  case "${path}" in
    "${PROJECT}"/*) printf '%s' "${path#${PROJECT}/}" ;;
    *) printf '%s' "${path}" ;;
  esac
}

run_public_eval_set() {
  local stage="$1"
  local checkpoint="$2"
  local eval_tier="$3"
  local max_examples="$4"
  local run_alias="$5"
  local checkpoint_alias="$6"

  local benchmarks=()
  local eval_gpus=()
  split_csv "${PUBLIC_EVAL_BENCHMARKS_CSV}" benchmarks
  split_csv "${EVAL_GPUS_CSV}" eval_gpus
  if [ "${#eval_gpus[@]}" -eq 0 ]; then
    echo "no eval GPUs configured" >&2
    exit 1
  fi

  echo "public_eval stage=${stage} checkpoint=${checkpoint} tier=${eval_tier} benchmarks=${PUBLIC_EVAL_BENCHMARKS_CSV}"
  local pids=()
  local idx=0
  for benchmark in "${benchmarks[@]}"; do
    local gpu="${eval_gpus[$((idx % ${#eval_gpus[@]}))]}"
    require_gpu_free "${gpu}" "${MIN_EVAL_GPU_FREE_GIB}"
    local eval_guard_log="${LOG_DIR}/v0_0_9_${stage}_${benchmark}_${eval_tier}_${RUN_STAMP}.memory_guard.log"
    local args=(
      "${PYTHON}" -u scripts/evaluate_public_benchmarks.py
      --checkpoint-dir "${checkpoint}"
      --tokenizer "${TOKENIZER}"
      --benchmark "${benchmark}"
      --run-alias "${run_alias}"
      --checkpoint-alias "${checkpoint_alias}"
      --eval-tier "${eval_tier}_${stage}_${benchmark}"
      --device cuda:0
      --max-length "${PUBLIC_EVAL_MAX_LENGTH}"
      --dtype bf16
      --pair-batch-size "${PUBLIC_EVAL_PAIR_BATCH_SIZE}"
    )
    if [ -n "${max_examples}" ]; then
      args+=(--max-examples-per-benchmark "${max_examples}")
    fi
    echo "public_eval launch benchmark=${benchmark} gpu=${gpu} log=${eval_guard_log}"
    (
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export CUDA_DEVICE_ORDER=PCI_BUS_ID
      export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
      export MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB}"
      export POLL_SECONDS="${POLL_SECONDS:-10}"
      export BREACH_COUNT="${BREACH_COUNT:-3}"
      export LOG_PATH="${eval_guard_log}"
      "${MEMORY_GUARD}" wrap -- "${args[@]}"
    ) &
    pids+=("$!")
    idx=$((idx + 1))
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [ "${failed}" -ne 0 ]; then
    echo "one or more public eval jobs failed for stage=${stage}" >&2
    exit 5
  fi
  echo "public_eval stage=${stage} status=completed"
}

run_target_eval_set() {
  local stage="$1"
  local checkpoint="$2"
  local out_dir="${PROJECT}/results/03_evals/target_posttrain_tasks/v0_0_9_${stage}_target_eval_${RUN_STAMP}"
  local guard_log="${LOG_DIR}/v0_0_9_${stage}_target_eval_${RUN_STAMP}.memory_guard.log"
  local args=(
    "${PYTHON}" -u scripts/evaluate_target_posttrain_tasks.py
    --target-eval-dir "${TARGET_EVAL_DIR}"
    --tokenizer "${TOKENIZER}"
    --checkpoint-dir "${checkpoint}"
    --output-dir "${out_dir}"
    --device cuda:0
    --dtype bf16
    --max-input-tokens 4096
    --max-new-tokens "${TARGET_EVAL_MAX_NEW_TOKENS}"
    --temperature 0.0
    --repetition-penalty 1.05
  )
  if [ -n "${TARGET_EVAL_LIMIT_PER_TASK}" ]; then
    args+=(--limit-per-task "${TARGET_EVAL_LIMIT_PER_TASK}")
  fi
  echo "target_eval stage=${stage} checkpoint=${checkpoint} out_dir=${out_dir}"
  (
    export CUDA_VISIBLE_DEVICES="${TRAIN_GPU}"
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    export MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB}"
    export POLL_SECONDS="${POLL_SECONDS:-10}"
    export BREACH_COUNT="${BREACH_COUNT:-3}"
    export LOG_PATH="${guard_log}"
    "${MEMORY_GUARD}" wrap -- "${args[@]}"
  )
  echo "target_eval stage=${stage} status=completed"
}

read_pipeline_checkpoint() {
  local key="$1"
  local summary="$2"
  KEY="${key}" SUMMARY="${summary}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

summary = Path(os.environ["SUMMARY"])
key = os.environ["KEY"]
if not summary.exists():
    raise SystemExit(f"missing pipeline summary: {summary}")
data = json.loads(summary.read_text(encoding="utf-8"))
value = data.get(key)
if not value:
    raise SystemExit(f"missing {key} in {summary}")
print(value)
PY
}

echo "run_stamp=${RUN_STAMP}"
echo "base_checkpoint=${BASE_CHECKPOINT}"
echo "train_gpu=${TRAIN_GPU}"
echo "eval_gpus=${EVAL_GPUS_CSV}"
echo "pipe_log=${PIPE_LOG}"
echo "summary_path=${SUMMARY_PATH}"
echo
echo "== free -h =="
free -h
echo
echo "== nvidia-smi =="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu --format=csv,noheader
echo

for path in "${PYTHON}" "${MEMORY_GUARD}" "${POSTTRAIN_LAUNCHER}" "${BASE_CHECKPOINT}" "${TOKENIZER}" "${TARGET_EVAL_DIR}"; do
  require_path "${path}"
done
if [[ "${BASE_CHECKPOINT}" != *"v0_0_9"* && "${BASE_CHECKPOINT}" != *"v0.0.9"* ]]; then
  echo "BASE_CHECKPOINT does not look like v0.0.9 lineage: ${BASE_CHECKPOINT}" >&2
  exit 3
fi
require_gpu_free "${TRAIN_GPU}" "${MIN_TRAIN_GPU_FREE_GIB}"

cat > "${LAUNCH_MANIFEST}" <<JSON
{
  "run_stamp": "${RUN_STAMP}",
  "stage": "v0.0.9_posttrain_with_public_evals",
  "start_time_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "project": "${PROJECT}",
  "canonical_env": "${PYTHON}",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "tokenizer": "${TOKENIZER}",
  "train_gpu": "${TRAIN_GPU}",
  "eval_gpus_csv": "${EVAL_GPUS_CSV}",
  "min_train_gpu_free_gib": ${MIN_TRAIN_GPU_FREE_GIB},
  "min_eval_gpu_free_gib": ${MIN_EVAL_GPU_FREE_GIB},
  "min_mem_available_gib": ${MIN_MEM_AVAILABLE_GIB},
  "public_eval_benchmarks_csv": "${PUBLIC_EVAL_BENCHMARKS_CSV}",
  "public_eval_tier": "${PUBLIC_EVAL_TIER}",
  "public_eval_max_examples": "${PUBLIC_EVAL_MAX_EXAMPLES}",
  "target_eval_dir": "${TARGET_EVAL_DIR}",
  "posttrain_launcher": "${POSTTRAIN_LAUNCHER}",
  "pipeline_log": "${PIPE_LOG}",
  "launch_manifest": "${LAUNCH_MANIFEST}",
  "summary_path": "${SUMMARY_PATH}"
}
JSON

BASE_RUN_ALIAS="v0_0_9_base"
BASE_CHECKPOINT_ALIAS="checkpoint_step_00236925_imported"

if [ "${RUN_BASE_PUBLIC_EVAL}" = "1" ]; then
  if [ "${RUN_BASE_PUBLIC_EVAL_ASYNC}" = "1" ]; then
    echo "public_eval stage=base status=started_async"
    run_public_eval_set "base" "${BASE_CHECKPOINT}" "${PUBLIC_EVAL_TIER}" "${PUBLIC_EVAL_MAX_EXAMPLES}" "${BASE_RUN_ALIAS}" "${BASE_CHECKPOINT_ALIAS}" &
    BASE_PUBLIC_EVAL_PID="$!"
  else
    run_public_eval_set "base" "${BASE_CHECKPOINT}" "${PUBLIC_EVAL_TIER}" "${PUBLIC_EVAL_MAX_EXAMPLES}" "${BASE_RUN_ALIAS}" "${BASE_CHECKPOINT_ALIAS}"
    BASE_PUBLIC_EVAL_PID=""
  fi
else
  echo "public_eval stage=base status=skipped"
  BASE_PUBLIC_EVAL_PID=""
fi

if [ "${RUN_BASE_TARGET_EVAL}" = "1" ]; then
  run_target_eval_set "base" "${BASE_CHECKPOINT}"
else
  echo "target_eval stage=base status=skipped"
fi

POSTTRAIN_SUMMARY="${PROJECT}/results/02_training_runs/v0_0_9_targeted_posttrain_pipeline_${RUN_STAMP}.summary.json"
FINAL_CHECKPOINT=""
SFT_CHECKPOINT=""
DPO_CHECKPOINT=""
if [ "${RUN_POSTTRAIN}" = "1" ]; then
  echo "posttrain status=started"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" \
  MIN_GPU_FREE_GIB="${MIN_TRAIN_GPU_FREE_GIB}" \
  MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB}" \
  RUN_STAMP="${RUN_STAMP}" \
  BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
  TOKENIZER="${TOKENIZER}" \
  PIPE_LOG="${POSTTRAIN_PIPE_LOG}" \
  GUARD_LOG="${POSTTRAIN_GUARD_LOG}" \
  "${POSTTRAIN_LAUNCHER}"
  echo "posttrain status=completed summary=${POSTTRAIN_SUMMARY}"
  SFT_CHECKPOINT="$(read_pipeline_checkpoint sft_checkpoint "${POSTTRAIN_SUMMARY}")"
  DPO_CHECKPOINT="$(read_pipeline_checkpoint dpo_checkpoint "${POSTTRAIN_SUMMARY}")"
  FINAL_CHECKPOINT="$(read_pipeline_checkpoint final_checkpoint "${POSTTRAIN_SUMMARY}")"
else
  echo "posttrain status=skipped"
fi

if [ -n "${BASE_PUBLIC_EVAL_PID}" ]; then
  echo "public_eval stage=base status=waiting_async pid=${BASE_PUBLIC_EVAL_PID}"
  wait "${BASE_PUBLIC_EVAL_PID}"
  echo "public_eval stage=base status=completed_async"
fi

if [ -n "${FINAL_CHECKPOINT}" ] && [ "${RUN_FINAL_PUBLIC_EVAL}" = "1" ]; then
  run_public_eval_set "final" "${FINAL_CHECKPOINT}" "${PUBLIC_EVAL_TIER}" "${PUBLIC_EVAL_MAX_EXAMPLES}" "v0_0_9_targeted_posttrain" "$(basename "${FINAL_CHECKPOINT}")"
else
  echo "public_eval stage=final status=skipped"
fi

if [ -n "${FINAL_CHECKPOINT}" ] && [ "${RUN_FINAL_TARGET_EVAL}" = "1" ]; then
  run_target_eval_set "final" "${FINAL_CHECKPOINT}"
else
  echo "target_eval stage=final status=skipped"
fi

cat > "${SUMMARY_PATH}" <<JSON
{
  "run_stamp": "${RUN_STAMP}",
  "completed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "sft_checkpoint": "${SFT_CHECKPOINT}",
  "dpo_checkpoint": "${DPO_CHECKPOINT}",
  "final_checkpoint": "${FINAL_CHECKPOINT}",
  "posttrain_summary": "${POSTTRAIN_SUMMARY}",
  "tokenizer": "${TOKENIZER}",
  "public_eval_benchmarks_csv": "${PUBLIC_EVAL_BENCHMARKS_CSV}",
  "public_eval_tier": "${PUBLIC_EVAL_TIER}",
  "target_eval_dir": "${TARGET_EVAL_DIR}",
  "pipeline_log": "${PIPE_LOG}",
  "launch_manifest": "${LAUNCH_MANIFEST}",
  "posttrain_pipe_log": "${POSTTRAIN_PIPE_LOG}",
  "posttrain_guard_log": "${POSTTRAIN_GUARD_LOG}"
}
JSON
echo "summary_path=${SUMMARY_PATH}"
