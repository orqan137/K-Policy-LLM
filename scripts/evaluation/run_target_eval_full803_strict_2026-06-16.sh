#!/usr/bin/env bash
set -euo pipefail

PROJECT="/DATA/castle/projects/make_llm"
PY="/DATA/castle/envs/make_llm/.venv/bin/python"
EVAL_SCRIPT="${PROJECT}/scripts/evaluate_target_posttrain_tasks_strict.py"
TARGET_EVAL="/DATA/castle/data/posttrain_v6_15b_v17/projects/make_llm/data/05_datasets/posttrain/v009_general_policy_sft_mega_v17_refusal_phrase_cleanup_2026-06-15_1630/target_eval"
TOKENIZER="/DATA/castle/data/posttrain_v6_15b_v17/projects/make_llm/data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model"
RESULT_ROOT="${PROJECT}/results/03_evals/target_posttrain_tasks"
RUN_ID="${RUN_ID:-$(date -u +%Y-%m-%d_%H%M)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${PROJECT}/results/cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT}/results/cache/huggingface/transformers"
export TORCH_HOME="${PROJECT}/results/cache/torch"

mkdir -p "${RESULT_ROOT}" "${PROJECT}/results/cache/huggingface" "${PROJECT}/results/cache/torch"

run_stage() {
  local label="$1"
  local checkpoint="$2"
  local output_dir="$3"

  echo "===== ${label} start $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "checkpoint=${checkpoint}"
  echo "output_dir=${output_dir}"

  "${PY}" -u "${EVAL_SCRIPT}" \
    --target-eval-dir "${TARGET_EVAL}" \
    --tokenizer "${TOKENIZER}" \
    --checkpoint-dir "${checkpoint}" \
    --output-dir "${output_dir}" \
    --device cuda:0 \
    --dtype bf16 \
    --batch-size 16 \
    --max-input-tokens 4096 \
    --max-new-tokens 220 \
    --temperature 0.0 \
    --repetition-penalty 1.05 \
    --seed 42

  echo "===== ${label} end $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
}

run_stage \
  "사전학습 베이스라인" \
  "${PROJECT}/results/02_training_runs/pretrain_1p2b_v0_0_9_80k_from_project_h200_b14_accum1_lc512_2026-06-09_150042/checkpoint_tokens_015b" \
  "${RESULT_ROOT}/v0_0_9_15b_pretrain_target_eval_full803_strict_${RUN_ID}"

run_stage \
  "추가학습(CPT)" \
  "${PROJECT}/results/02_training_runs/pretrain_1p2b_v0_0_9_15b_pt_v12_cpt_initweights_2026-06-15_145901/checkpoint_final" \
  "${RESULT_ROOT}/v0_0_9_15b_cpt_target_eval_full803_strict_${RUN_ID}"

run_stage \
  "지도학습(SFT)" \
  "${PROJECT}/results/02_training_runs/v0_0_9_15b_pt_v6_v17_cpt_barrage_sft_4ep_lr5e6_mb32_accum2_gpu0_2026-06-16_004553/checkpoint_final" \
  "${RESULT_ROOT}/v0_0_9_15b_sft_target_eval_full803_strict_${RUN_ID}"

run_stage \
  "선호학습(DPO)" \
  "${PROJECT}/results/02_training_runs/v0_0_9_15b_pt_v6_v17_full_sft_failure_repair_dpo10k_gpu1_seq512_mb8_cont500_from_chunk100_2026-06-16_031507/checkpoint_final" \
  "${RESULT_ROOT}/v0_0_9_15b_dpo_target_eval_full803_strict_${RUN_ID}"

