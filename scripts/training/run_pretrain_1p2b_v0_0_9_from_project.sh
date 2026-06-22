#!/usr/bin/env bash
set -euo pipefail

CASTLE_ROOT="/DATA/castle"
CODE_PROJECT="${CODE_PROJECT:-${CASTLE_ROOT}/projects/make_llm}"
ASSET_PROJECT="${ASSET_PROJECT:-${CASTLE_ROOT}/project/projects/make_llm}"
PYTHON="${PYTHON:-/DATA/castle/envs/make_llm/.venv/bin/python}"
cd "${CODE_PROJECT}"

require_under_castle() {
  local label="$1"
  local raw_path="$2"
  local resolved
  resolved="$(realpath -sm "${raw_path}")"
  case "${resolved}" in
    "${CASTLE_ROOT}"|"${CASTLE_ROOT}"/*) ;;
    *)
      echo "${label} must stay under ${CASTLE_ROOT}: ${raw_path} -> ${resolved}" >&2
      exit 6
      ;;
  esac
}

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${TARGET_CUDA_VISIBLE_DEVICES:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRAIN_CONFIG="${TRAIN_CONFIG:-${CODE_PROJECT}/configs/03_train/pretrain_1p2b_v0_0_9_80k_from_project_h200_b12_accum1_lc512_safe.yaml}"
DATASET_DIR="${DATASET_DIR:-${ASSET_PROJECT}/data/05_datasets/pretrain/v0_0_9_80k_mapreduce_ratio_fineweb2_15shards_2026-06-09_023724_fast24_1p25}"
MODEL_CONFIG="${MODEL_CONFIG:-${ASSET_PROJECT}/configs/02_model/make_llm_1.2b_v0_0_9_80k.yaml}"
TOKENIZER="${TOKENIZER:-${ASSET_PROJECT}/data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model}"
RESUME_FROM="${RESUME_FROM:-}"

CACHE_ROOT="${CACHE_ROOT:-${CODE_PROJECT}/results/cache/v0_0_9_from_project}"
TMP_ROOT="${TMP_ROOT:-${CODE_PROJECT}/results/tmp/v0_0_9_from_project}"
config_defaults="$("${PYTHON}" - "${TRAIN_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(cfg["max_steps"]))
print(int(cfg.get("save_every_steps") or 0))
PY
)"
CONFIG_MAX_STEPS="$(printf '%s\n' "${config_defaults}" | sed -n '1p')"
CONFIG_SAVE_EVERY_STEPS="$(printf '%s\n' "${config_defaults}" | sed -n '2p')"
MAX_STEPS="${MAX_STEPS:-${CONFIG_MAX_STEPS}}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-${CONFIG_SAVE_EVERY_STEPS}}"
SAVE_TOKEN_MILESTONES="${SAVE_TOKEN_MILESTONES:-0.5b,1b,1.5b,2b,2.5b,3b,3.5b,4b,4.5b,5b,5.5b,6b,6.5b,7b,7.5b,8b,8.5b,9b,9.5b,10b,10.5b,11b,11.5b,12b,12.5b,13b,13.5b,14b,14.5b,15b,15.5b,16b,16.5b,17b,17.5b,18b,18.5b,19b,19.5b,20b,20.5b,21b,21.5b,22b,22.5b,23b,23.5b,24b,24.5b,25b,25.5b,26b,26.5b,27b,27.5b,28b,28.5b,29b,29.5b,30b,30.5b,31b}"
MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-12}"
STOP_AFTER_SECONDS="${STOP_AFTER_SECONDS:-}"
ENABLE_PUBLIC_EVAL="${ENABLE_PUBLIC_EVAL:-1}"
EVAL_PUBLIC_ASYNC_ON_SAVE="${EVAL_PUBLIC_ASYNC_ON_SAVE:-1}"
EVAL_PUBLIC_TOKEN_MILESTONES_ONLY="${EVAL_PUBLIC_TOKEN_MILESTONES_ONLY:-1}"
EVAL_MAX_EXAMPLES_PER_BENCHMARK="${EVAL_MAX_EXAMPLES_PER_BENCHMARK:-50}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
if [ "${EVAL_DEVICE}" = "cpu" ]; then
  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-}"
else
  EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
fi
EVAL_PAIR_BATCH_SIZE="${EVAL_PAIR_BATCH_SIZE:-1}"
PUBLIC_EVAL_MONITOR_METRIC="${PUBLIC_EVAL_MONITOR_METRIC:-overall_avg}"
BEST_CHECKPOINT_NAME="${BEST_CHECKPOINT_NAME:-checkpoint_best_public_macro}"

for guarded in \
  "${CODE_PROJECT}" "${ASSET_PROJECT}" "${PYTHON}" "${CACHE_ROOT}" "${TMP_ROOT}" \
  "${TRAIN_CONFIG}" "${DATASET_DIR}" "${MODEL_CONFIG}" "${TOKENIZER}"; do
  require_under_castle "path" "${guarded}"
done
if [ -n "${RESUME_FROM}" ]; then
  require_under_castle "RESUME_FROM" "${RESUME_FROM}"
  if [ ! -f "${RESUME_FROM}/training_state.pt" ]; then
    echo "RESUME_FROM must be a full trainer checkpoint containing training_state.pt: ${RESUME_FROM}" >&2
    exit 7
  fi
fi

mkdir -p "${CACHE_ROOT}/xdg" "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${TMP_ROOT}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export HF_HOME="${CACHE_ROOT}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${CACHE_ROOT}/torch"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TMPDIR="${TMP_ROOT}"

RESUME_ARGS=()
if [ -n "${RESUME_FROM}" ]; then
  RESUME_ARGS=(--resume-from "${RESUME_FROM}" --resume-strict-data-state)
fi

EVAL_ARGS=()
if [ "${ENABLE_PUBLIC_EVAL}" = "1" ]; then
  EVAL_ARGS=(
    --eval-public-on-save
    --eval-benchmark arc_challenge
    --eval-benchmark arc_easy
    --eval-benchmark boolq
    --eval-benchmark click
    --eval-benchmark commonsenseqa
    --eval-benchmark hellaswag
    --eval-benchmark kmmlu_pro
    --eval-benchmark kobalt_700
    --eval-benchmark mmlu
    --eval-benchmark mmlu_pro
    --eval-benchmark openbookqa
    --eval-benchmark truthfulqa_mc1
    --eval-benchmark winogrande
    --eval-max-examples-per-benchmark "${EVAL_MAX_EXAMPLES_PER_BENCHMARK}"
    --eval-cuda-visible-devices "${EVAL_CUDA_VISIBLE_DEVICES}"
    --eval-device "${EVAL_DEVICE}"
    --eval-pair-batch-size "${EVAL_PAIR_BATCH_SIZE}"
    --public-eval-monitor-metric "${PUBLIC_EVAL_MONITOR_METRIC}"
    --best-checkpoint-name "${BEST_CHECKPOINT_NAME}"
  )
  if [ "${EVAL_PUBLIC_TOKEN_MILESTONES_ONLY}" = "1" ]; then
    EVAL_ARGS+=(--eval-public-on-token-milestones-only)
  fi
  if [ "${EVAL_PUBLIC_ASYNC_ON_SAVE}" = "1" ]; then
    EVAL_ARGS+=(--eval-public-async-on-save)
  fi
fi

STOP_ARGS=()
if [ -n "${STOP_AFTER_SECONDS}" ]; then
  STOP_ARGS=(--stop-after-seconds "${STOP_AFTER_SECONDS}")
fi

"${PYTHON}" -u "${CODE_PROJECT}/scripts/train_tiny_pretrain.py" \
  --train-config "${TRAIN_CONFIG}" \
  --dataset "${DATASET_DIR}" \
  --tokenizer "${TOKENIZER}" \
  --model-config "${MODEL_CONFIG}" \
  --device cuda:0 \
  --prefer-shards \
  --max-steps "${MAX_STEPS}" \
  --save-every-steps "${SAVE_EVERY_STEPS}" \
  --save-token-milestones "${SAVE_TOKEN_MILESTONES}" \
  --max-checkpoints-to-keep "${MAX_CHECKPOINTS_TO_KEEP}" \
  "${RESUME_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${STOP_ARGS[@]}"
