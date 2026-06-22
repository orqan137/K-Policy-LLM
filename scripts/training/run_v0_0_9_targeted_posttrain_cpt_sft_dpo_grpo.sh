#!/usr/bin/env bash
set -euo pipefail

PROJECT="/DATA/castle/projects/make_llm"
PYTHON="/DATA/castle/envs/make_llm/.venv/bin/python"
MEMORY_GUARD="/DATA/castle/projects/workspace/scripts/memory_guard.sh"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y-%m-%d_%H%M)}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"
TOKENIZER="${TOKENIZER:-${PROJECT}/data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model}"
MODEL_CONFIG="${MODEL_CONFIG:-${PROJECT}/configs/02_model/make_llm_1.2b_v0_0_9_80k.yaml}"
POSTTRAIN_PACK="${POSTTRAIN_PACK:-${PROJECT}/data/05_datasets/posttrain/v0_0_9_posttrain_targeted_opendataloader_improvement_2026-06-15_0130}"
CPT_DATASET="${CPT_DATASET:-${PROJECT}/data/05_datasets/cpt/v0_0_9_targeted_clean_cpt_80k_2026-06-15_0134/combined_2048_shuffle}"
CPT_CONFIG="${CPT_CONFIG:-${PROJECT}/configs/03_train/pretrain_1p2b_v0_0_9_targeted_clean_cpt.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT}/results/02_training_runs}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT}/results/03_evals/target_posttrain_tasks}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
MIN_GPU_FREE_GIB="${MIN_GPU_FREE_GIB:-42}"

CPT_MAX_STEPS="${CPT_MAX_STEPS:-4408}"
CPT_SAVE_EVERY_STEPS="${CPT_SAVE_EVERY_STEPS:-1104}"
CPT_MAX_CHECKPOINTS_TO_KEEP="${CPT_MAX_CHECKPOINTS_TO_KEEP:-4}"

SFT_SEQ_LEN="${SFT_SEQ_LEN:-4096}"
SFT_MAX_STEPS="${SFT_MAX_STEPS:-6240}"
SFT_EVAL_EVERY_STEPS="${SFT_EVAL_EVERY_STEPS:-520}"
SFT_SAVE_EVERY_STEPS="${SFT_SAVE_EVERY_STEPS:-1040}"
SFT_MAX_EVAL_BATCHES="${SFT_MAX_EVAL_BATCHES:-64}"
SFT_MAX_TRAIN_EXAMPLES="${SFT_MAX_TRAIN_EXAMPLES:-}"
SFT_MAX_EVAL_EXAMPLES="${SFT_MAX_EVAL_EXAMPLES:-}"

DPO_SEQ_LEN="${DPO_SEQ_LEN:-3072}"
DPO_TRAIN="${DPO_TRAIN:-${POSTTRAIN_PACK}/dpo/train.jsonl}"
DPO_VAL="${DPO_VAL:-${POSTTRAIN_PACK}/dpo/val.jsonl}"
DPO_MAX_STEPS="${DPO_MAX_STEPS:-$(wc -l < "${DPO_TRAIN}")}"
DPO_MAX_TRAIN_EXAMPLES="${DPO_MAX_TRAIN_EXAMPLES:-}"
DPO_MAX_EVAL_EXAMPLES="${DPO_MAX_EVAL_EXAMPLES:-}"
DPO_MAX_EVAL_BATCHES="${DPO_MAX_EVAL_BATCHES:-64}"

RUN_GRPO_NUMERIC="${RUN_GRPO_NUMERIC:-1}"
GRPO_TRAIN="${GRPO_TRAIN:-${POSTTRAIN_PACK}/grpo_numeric/train.jsonl}"
GRPO_VAL="${GRPO_VAL:-${POSTTRAIN_PACK}/grpo_numeric/val.jsonl}"
GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-$(wc -l < "${GRPO_TRAIN}")}"
GRPO_MAX_TRAIN_EXAMPLES="${GRPO_MAX_TRAIN_EXAMPLES:-}"
GRPO_MAX_EVAL_EXAMPLES="${GRPO_MAX_EVAL_EXAMPLES:-256}"
GRPO_NUM_GENERATIONS="${GRPO_NUM_GENERATIONS:-2}"
GRPO_PER_DEVICE_BATCH="${GRPO_PER_DEVICE_BATCH:-${GRPO_NUM_GENERATIONS}}"

RUN_TARGET_EVAL="${RUN_TARGET_EVAL:-1}"
TARGET_EVAL_DIR="${TARGET_EVAL_DIR:-${POSTTRAIN_PACK}/target_eval}"
TARGET_EVAL_LIMIT_PER_TASK="${TARGET_EVAL_LIMIT_PER_TASK:-}"
TARGET_EVAL_MAX_NEW_TOKENS="${TARGET_EVAL_MAX_NEW_TOKENS:-420}"

CPT_PREFIX="${CPT_PREFIX:-pretrain_1p2b_v0_0_9_targeted_clean_cpt}"
SFT_PREFIX="${SFT_PREFIX:-v0_0_9_targeted_cpt_sft}"
DPO_PREFIX="${DPO_PREFIX:-v0_0_9_targeted_cpt_sft_dpo}"
GRPO_PREFIX="${GRPO_PREFIX:-v0_0_9_targeted_cpt_sft_dpo_grpo_numeric}"

LOG_DIR="${PROJECT}/results/run_logs"
MANIFEST_DIR="${PROJECT}/results/run_manifests"
PIPE_LOG="${PIPE_LOG:-${LOG_DIR}/v0_0_9_targeted_posttrain_cpt_sft_dpo_grpo_${RUN_STAMP}.log}"
GUARD_LOG="${GUARD_LOG:-${LOG_DIR}/v0_0_9_targeted_posttrain_cpt_sft_dpo_grpo_${RUN_STAMP}.memory_guard.log}"
LAUNCH_MANIFEST="${LAUNCH_MANIFEST:-${MANIFEST_DIR}/v0_0_9_targeted_posttrain_cpt_sft_dpo_grpo_${RUN_STAMP}.launch.json}"

mkdir -p "${LOG_DIR}" "${MANIFEST_DIR}" "${OUTPUT_ROOT}" "${EVAL_ROOT}"
exec > >(tee -a "${PIPE_LOG}") 2>&1

cd "${PROJECT}"

export CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER PYTORCH_CUDA_ALLOC_CONF
export MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB:-64}"
export POLL_SECONDS="${POLL_SECONDS:-10}"
export BREACH_COUNT="${BREACH_COUNT:-3}"
export LOG_PATH="${GUARD_LOG}"

echo "run_stamp=${RUN_STAMP}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "base_checkpoint=${BASE_CHECKPOINT:-<unset>}"
echo "posttrain_pack=${POSTTRAIN_PACK}"
echo "cpt_dataset=${CPT_DATASET}"
echo "target_eval_dir=${TARGET_EVAL_DIR}"
echo "pipe_log=${PIPE_LOG}"
echo "guard_log=${GUARD_LOG}"
echo
echo "== free -h =="
free -h
echo
echo "== nvidia-smi =="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo
FIRST_VISIBLE_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
if [[ "${FIRST_VISIBLE_GPU}" =~ ^[0-9]+$ ]]; then
  GPU_FREE_MIB="$(nvidia-smi -i "${FIRST_VISIBLE_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  MIN_GPU_FREE_MIB=$((MIN_GPU_FREE_GIB * 1024))
  echo "selected_gpu=${FIRST_VISIBLE_GPU} gpu_free_mib=${GPU_FREE_MIB} min_gpu_free_mib=${MIN_GPU_FREE_MIB}"
  if [ "${GPU_FREE_MIB}" -lt "${MIN_GPU_FREE_MIB}" ]; then
    echo "selected GPU does not have enough free memory for the default v0.0.9 full posttrain settings." >&2
    echo "Use a freer 48GB-class GPU, or explicitly lower SFT_SEQ_LEN/DPO_SEQ_LEN and MIN_GPU_FREE_GIB for a constrained ablation." >&2
    exit 4
  fi
else
  echo "could not parse numeric CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}; skipping GPU free-memory gate" >&2
fi
echo
echo "== df -h /DATA/castle =="
df -h /DATA/castle
echo

if [ -z "${BASE_CHECKPOINT}" ]; then
  echo "missing BASE_CHECKPOINT. Provide a v0.0.9 base checkpoint path; this launcher will not initialize from v0.0.8 or another version." >&2
  exit 2
fi

for path in \
  "${PYTHON}" \
  "${MEMORY_GUARD}" \
  "${BASE_CHECKPOINT}" \
  "${TOKENIZER}" \
  "${MODEL_CONFIG}" \
  "${POSTTRAIN_PACK}/manifest.json" \
  "${CPT_DATASET}/input_ids.pt" \
  "${CPT_DATASET}/manifest.json" \
  "${CPT_CONFIG}" \
  "${POSTTRAIN_PACK}/sft/train.jsonl" \
  "${POSTTRAIN_PACK}/sft/val.jsonl" \
  "${DPO_TRAIN}" \
  "${DPO_VAL}" \
  "${GRPO_TRAIN}" \
  "${GRPO_VAL}" \
  "${TARGET_EVAL_DIR}"; do
  if [ ! -e "${path}" ]; then
    echo "missing required path: ${path}" >&2
    exit 1
  fi
done

if [[ "${BASE_CHECKPOINT}" != *"v0_0_9"* && "${BASE_CHECKPOINT}" != *"v0.0.9"* ]]; then
  echo "BASE_CHECKPOINT does not look like v0.0.9 lineage: ${BASE_CHECKPOINT}" >&2
  echo "Refusing to run; use a v0.0.9 checkpoint or explicitly rename/document a non-official ablation outside this launcher." >&2
  exit 3
fi

cat > "${LAUNCH_MANIFEST}" <<JSON
{
  "run_stamp": "${RUN_STAMP}",
  "stage": "v0.0.9_targeted_posttrain",
  "project": "${PROJECT}",
  "canonical_env": "${PYTHON}",
  "start_time_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "tokenizer": "${TOKENIZER}",
  "model_config": "${MODEL_CONFIG}",
  "posttrain_pack": "${POSTTRAIN_PACK}",
  "cpt_dataset": "${CPT_DATASET}",
  "target_eval_dir": "${TARGET_EVAL_DIR}",
  "pipeline_log": "${PIPE_LOG}",
  "memory_guard_log": "${GUARD_LOG}",
  "output_root": "${OUTPUT_ROOT}",
  "eval_root": "${EVAL_ROOT}",
  "cuda_visible_devices": "${CUDA_VISIBLE_DEVICES}",
  "scope": "/DATA/castle only",
  "method": "CPT -> SFT -> DPO -> optional GRPO numeric; target_eval after trainable instruction stages"
}
JSON

find_latest_checkpoint() {
  local prefix="$1"
  OUTPUT_ROOT="${OUTPUT_ROOT}" RUN_PREFIX="${prefix}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
prefix = os.environ["RUN_PREFIX"] + "_"
candidates = sorted(root.glob(prefix + "*"), key=lambda path: path.stat().st_mtime, reverse=True)
for run_dir in candidates:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("best_eval_checkpoint", "checkpoint"):
        checkpoint = summary.get(key)
        if checkpoint and Path(checkpoint).exists():
            print(checkpoint)
            raise SystemExit(0)
raise SystemExit(f"could not find checkpoint for prefix={prefix!r}")
PY
}

run_target_eval() {
  local stage="$1"
  local checkpoint="$2"
  if [ "${RUN_TARGET_EVAL}" != "1" ]; then
    return 0
  fi
  local out_dir="${EVAL_ROOT}/v0_0_9_${stage}_target_eval_${RUN_STAMP}"
  local extra_args=()
  if [ -n "${TARGET_EVAL_LIMIT_PER_TASK}" ]; then
    extra_args+=(--limit-per-task "${TARGET_EVAL_LIMIT_PER_TASK}")
  fi
  echo "stage=${stage} target_eval=started checkpoint=${checkpoint}"
  "${MEMORY_GUARD}" wrap -- \
    "${PYTHON}" -u scripts/evaluate_target_posttrain_tasks.py \
      --target-eval-dir "${TARGET_EVAL_DIR}" \
      --tokenizer "${TOKENIZER}" \
      --checkpoint-dir "${checkpoint}" \
      --output-dir "${out_dir}" \
      --device cuda:0 \
      --dtype bf16 \
      --max-input-tokens 4096 \
      --max-new-tokens "${TARGET_EVAL_MAX_NEW_TOKENS}" \
      --temperature 0.0 \
      --repetition-penalty 1.05 \
      "${extra_args[@]}"
  echo "stage=${stage} target_eval=completed output_dir=${out_dir}"
}

echo "stage=cpt status=started"
"${MEMORY_GUARD}" wrap -- \
  "${PYTHON}" -u scripts/train_tiny_pretrain.py \
    --train-config "${CPT_CONFIG}" \
    --dataset "${CPT_DATASET}" \
    --tokenizer "${TOKENIZER}" \
    --model-config "${MODEL_CONFIG}" \
    --resume-from "${BASE_CHECKPOINT}" \
    --resume-model-only \
    --device cuda:0 \
    --max-steps "${CPT_MAX_STEPS}" \
    --save-every-steps "${CPT_SAVE_EVERY_STEPS}" \
    --max-checkpoints-to-keep "${CPT_MAX_CHECKPOINTS_TO_KEEP}"

CPT_CHECKPOINT="$(find_latest_checkpoint "${CPT_PREFIX}")"
echo "stage=cpt status=completed checkpoint=${CPT_CHECKPOINT}"

SFT_EXTRA_ARGS=()
if [ -n "${SFT_MAX_TRAIN_EXAMPLES}" ]; then
  SFT_EXTRA_ARGS+=(--max-train-examples "${SFT_MAX_TRAIN_EXAMPLES}")
fi
if [ -n "${SFT_MAX_EVAL_EXAMPLES}" ]; then
  SFT_EXTRA_ARGS+=(--max-eval-examples "${SFT_MAX_EVAL_EXAMPLES}")
fi

echo "stage=sft status=started"
"${MEMORY_GUARD}" wrap -- \
  "${PYTHON}" -u scripts/train_tiny_reasoning_sft.py \
    --train-jsonl "${POSTTRAIN_PACK}/sft/train.jsonl" \
    --eval-jsonl "${POSTTRAIN_PACK}/sft/val.jsonl" \
    --tokenizer "${TOKENIZER}" \
    --model-config "${MODEL_CONFIG}" \
    --base-checkpoint "${CPT_CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda:0 \
    --seq-len "${SFT_SEQ_LEN}" \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 1e-5 \
    --weight-decay 0.01 \
    --max-steps "${SFT_MAX_STEPS}" \
    --max-eval-batches "${SFT_MAX_EVAL_BATCHES}" \
    --eval-every-steps "${SFT_EVAL_EVERY_STEPS}" \
    --save-every-steps "${SFT_SAVE_EVERY_STEPS}" \
    --max-checkpoints-to-keep 4 \
    --best-checkpoint-name checkpoint_best_eval \
    --best-min-delta 0.003 \
    --gradient-checkpointing \
    --run-id-prefix "${SFT_PREFIX}" \
    --run-type v0_0_9_targeted_cpt_sft \
    --seed 42 \
    "${SFT_EXTRA_ARGS[@]}"

SFT_CHECKPOINT="$(find_latest_checkpoint "${SFT_PREFIX}")"
echo "stage=sft status=completed checkpoint=${SFT_CHECKPOINT}"
run_target_eval "sft" "${SFT_CHECKPOINT}"

DPO_EXTRA_ARGS=()
if [ -n "${DPO_MAX_TRAIN_EXAMPLES}" ]; then
  DPO_EXTRA_ARGS+=(--max-train-examples "${DPO_MAX_TRAIN_EXAMPLES}")
fi
if [ -n "${DPO_MAX_EVAL_EXAMPLES}" ]; then
  DPO_EXTRA_ARGS+=(--max-eval-examples "${DPO_MAX_EVAL_EXAMPLES}")
fi

echo "stage=dpo status=started"
"${MEMORY_GUARD}" wrap -- \
  "${PYTHON}" -u scripts/train_reasoning_dpo.py \
    --train-jsonl "${DPO_TRAIN}" \
    --eval-jsonl "${DPO_VAL}" \
    --tokenizer "${TOKENIZER}" \
    --policy-checkpoint "${SFT_CHECKPOINT}" \
    --reference-checkpoint "${SFT_CHECKPOINT}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda:0 \
    --seq-len "${DPO_SEQ_LEN}" \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-6 \
    --weight-decay 0.0 \
    --beta 0.1 \
    --max-steps "${DPO_MAX_STEPS}" \
    --max-eval-batches "${DPO_MAX_EVAL_BATCHES}" \
    --gradient-checkpointing \
    --run-id-prefix "${DPO_PREFIX}" \
    --run-type v0_0_9_targeted_cpt_sft_dpo \
    --seed 42 \
    "${DPO_EXTRA_ARGS[@]}"

DPO_CHECKPOINT="$(find_latest_checkpoint "${DPO_PREFIX}")"
echo "stage=dpo status=completed checkpoint=${DPO_CHECKPOINT}"
run_target_eval "dpo" "${DPO_CHECKPOINT}"

FINAL_CHECKPOINT="${DPO_CHECKPOINT}"
if [ "${RUN_GRPO_NUMERIC}" = "1" ]; then
  GRPO_EXTRA_ARGS=()
  if [ -n "${GRPO_MAX_TRAIN_EXAMPLES}" ]; then
    GRPO_EXTRA_ARGS+=(--max-train-examples "${GRPO_MAX_TRAIN_EXAMPLES}")
  fi
  echo "stage=grpo_numeric status=started"
  "${MEMORY_GUARD}" wrap -- \
    "${PYTHON}" -u scripts/train_grpo_numeric_trl.py \
      --train-jsonl "${GRPO_TRAIN}" \
      --eval-jsonl "${GRPO_VAL}" \
      --tokenizer "${TOKENIZER}" \
      --policy-checkpoint "${DPO_CHECKPOINT}" \
      --output-root "${OUTPUT_ROOT}" \
      --run-id-prefix "${GRPO_PREFIX}" \
      --run-type v0_0_9_targeted_cpt_sft_dpo_grpo_numeric_probe \
      --max-eval-examples "${GRPO_MAX_EVAL_EXAMPLES}" \
      --max-steps "${GRPO_MAX_STEPS}" \
      --per-device-train-batch-size "${GRPO_PER_DEVICE_BATCH}" \
      --per-device-eval-batch-size "${GRPO_PER_DEVICE_BATCH}" \
      --gradient-accumulation-steps 1 \
      --num-generations "${GRPO_NUM_GENERATIONS}" \
      --max-prompt-length 512 \
      --max-completion-length 64 \
      --learning-rate 2e-6 \
      --weight-decay 0.0 \
      --beta 0.03 \
      --epsilon 0.2 \
      --loss-type grpo \
      --temperature 0.8 \
      --top-p 0.9 \
      --top-k 50 \
      --repetition-penalty 1.08 \
      --logging-steps 1 \
      --save-steps 0 \
      --bf16 \
      --gradient-checkpointing \
      --seed 42 \
      "${GRPO_EXTRA_ARGS[@]}"
  GRPO_CHECKPOINT="$(find_latest_checkpoint "${GRPO_PREFIX}")"
  FINAL_CHECKPOINT="${GRPO_CHECKPOINT}"
  echo "stage=grpo_numeric status=completed checkpoint=${GRPO_CHECKPOINT}"
  run_target_eval "grpo_numeric" "${GRPO_CHECKPOINT}"
else
  echo "stage=grpo_numeric status=skipped RUN_GRPO_NUMERIC=${RUN_GRPO_NUMERIC}"
fi

PIPELINE_SUMMARY="${OUTPUT_ROOT}/v0_0_9_targeted_posttrain_pipeline_${RUN_STAMP}.summary.json"
cat > "${PIPELINE_SUMMARY}" <<JSON
{
  "run_stamp": "${RUN_STAMP}",
  "base_checkpoint": "${BASE_CHECKPOINT}",
  "cpt_checkpoint": "${CPT_CHECKPOINT}",
  "sft_checkpoint": "${SFT_CHECKPOINT}",
  "dpo_checkpoint": "${DPO_CHECKPOINT}",
  "final_checkpoint": "${FINAL_CHECKPOINT}",
  "posttrain_pack": "${POSTTRAIN_PACK}",
  "cpt_dataset": "${CPT_DATASET}",
  "target_eval_dir": "${TARGET_EVAL_DIR}",
  "pipeline_log": "${PIPE_LOG}",
  "memory_guard_log": "${GUARD_LOG}",
  "launch_manifest": "${LAUNCH_MANIFEST}",
  "completed_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
echo "pipeline_summary=${PIPELINE_SUMMARY}"
