#!/usr/bin/env bash
set -euo pipefail

CASTLE_ROOT="${CASTLE_ROOT:-/DATA/castle}"
PROJECT="${PROJECT:-${CASTLE_ROOT}/projects/make_llm}"
PYTHON="${PYTHON:-${CASTLE_ROOT}/envs/make_llm/.venv/bin/python}"

DEFAULT_RUN_DIR="${PROJECT}/results/02_training_runs/pretrain_1p2b_v0_0_9_80k_from_project_h200_b14_accum1_lc512_2026-06-09_150042"
DEFAULT_CHECKPOINT_DIR="${DEFAULT_RUN_DIR}/checkpoint_tokens_013500m"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${1:-${DEFAULT_CHECKPOINT_DIR}}}"
RUN_ALIAS="${RUN_ALIAS:-kfri-dmpmaker}"
CHECKPOINT_ALIAS="${CHECKPOINT_ALIAS:-$(basename "${CHECKPOINT_DIR}")}"
TARGET_NAME="${TARGET_NAME:-kfri-dmpmaker}"

PROJECT_TOKENIZER="${PROJECT}/data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model"
LEGACY_TOKENIZER="${CASTLE_ROOT}/project/projects/make_llm/data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model"
if [ -z "${TOKENIZER:-}" ]; then
  if [ -f "${PROJECT_TOKENIZER}" ]; then
    TOKENIZER="${PROJECT_TOKENIZER}"
  else
    TOKENIZER="${LEGACY_TOKENIZER}"
  fi
fi

OFFICIAL_BENCHMARK_ROOT="${OFFICIAL_BENCHMARK_ROOT:-${PROJECT}/results/03_evals/official_hf_downloads/extended_mcq_en_ko_2026-06-10_124209/normalized}"
LIGHT_BENCHMARK_ROOT="${LIGHT_BENCHMARK_ROOT:-${PROJECT}/data/05_datasets/eval/public_benchmarks/normalized}"
PROMPTS="${PROMPTS:-${PROJECT}/configs/04_eval/v0_0_9_korean_politics_qualitative_prompts_2026-06-09.jsonl}"

STAMP="${STAMP:-$(date -u +%Y-%m-%d_%H%M%S)}"
SUITE_OUT="${SUITE_OUT:-${PROJECT}/results/03_evals/kfri_dmpmaker_official/${CHECKPOINT_ALIAS}/${STAMP}}"

RUN_PRECHECK="${RUN_PRECHECK:-1}"
RUN_PUBLIC_LIGHT="${RUN_PUBLIC_LIGHT:-1}"
RUN_NEXT_TOKEN="${RUN_NEXT_TOKEN:-1}"
RUN_CHOICE_TEXT="${RUN_CHOICE_TEXT:-1}"
RUN_PERMUTATION="${RUN_PERMUTATION:-1}"
RUN_QUALITATIVE="${RUN_QUALITATIVE:-1}"
DRY_RUN="${DRY_RUN:-0}"

EVAL_CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-bf16}"
NEXT_TOKEN_BATCH_SIZE="${NEXT_TOKEN_BATCH_SIZE:-8}"
CHOICE_TEXT_BATCH_SIZE="${CHOICE_TEXT_BATCH_SIZE:-8}"
CHOICE_TEXT_MAX_EXAMPLES="${CHOICE_TEXT_MAX_EXAMPLES:-5000}"
PUBLIC_LIGHT_MAX_EXAMPLES="${PUBLIC_LIGHT_MAX_EXAMPLES:-50}"
PUBLIC_LIGHT_PAIR_BATCH_SIZE="${PUBLIC_LIGHT_PAIR_BATCH_SIZE:-1}"
QUAL_MAX_NEW_TOKENS="${QUAL_MAX_NEW_TOKENS:-320}"
MIN_MEM_AVAILABLE_GIB="${MIN_MEM_AVAILABLE_GIB:-64}"
MIN_GPU_FREE_MIB="${MIN_GPU_FREE_MIB:-20000}"
ALLOW_LOW_MEMORY="${ALLOW_LOW_MEMORY:-0}"
ALLOW_LOW_GPU_MEMORY="${ALLOW_LOW_GPU_MEMORY:-0}"

NEXT_TOKEN_BENCHMARKS="${NEXT_TOKEN_BENCHMARKS:-arc_challenge arc_easy boolq click commonsenseqa haerae_bench_1p1_mcq hellaswag kmmlu kmmlu_hard kmmlu_pro kobalt_700 kobest_boolq kobest_copa kobest_hellaswag kobest_sentineg kobest_wic mmlu mmlu_pro openbookqa winogrande}"
CHOICE_TEXT_BENCHMARKS="${CHOICE_TEXT_BENCHMARKS:-mmlu mmlu_pro kmmlu kmmlu_hard kmmlu_pro boolq click}"
PERMUTATION_BENCHMARKS="${PERMUTATION_BENCHMARKS:-mmlu mmlu_pro kmmlu kmmlu_hard kmmlu_pro boolq click}"
PERMUTATION_SEEDS="${PERMUTATION_SEEDS:-101 202 303}"
PUBLIC_LIGHT_BENCHMARKS="${PUBLIC_LIGHT_BENCHMARKS:-arc_challenge arc_easy boolq click commonsenseqa hellaswag kmmlu_pro kobalt_700 mmlu mmlu_pro openbookqa truthfulqa_mc1 winogrande}"

fail() {
  printf 'kfri-dmpmaker official eval not ready: %s\n' "$*" >&2
  exit 1
}

require_under_castle() {
  local label="$1"
  local raw_path="$2"
  local resolved
  resolved="$(realpath -sm "${raw_path}")"
  case "${resolved}" in
    "${CASTLE_ROOT}"|"${CASTLE_ROOT}"/*) ;;
    *) fail "${label} must stay under ${CASTLE_ROOT}: ${raw_path} -> ${resolved}" ;;
  esac
}

mem_available_gib() {
  awk '/MemAvailable:/ {printf "%.0f\n", $2 / 1024 / 1024}' /proc/meminfo
}

gpu_free_mib() {
  local gpu_index="${EVAL_CUDA_VISIBLE_DEVICES%%,*}"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F',' -v wanted="${gpu_index}" '$1 ~ wanted {gsub(/ /, "", $2); print $2; exit}'
}

precheck_resources() {
  local mem_gib
  mem_gib="$(mem_available_gib)"
  if [ "${mem_gib}" -lt "${MIN_MEM_AVAILABLE_GIB}" ] && [ "${ALLOW_LOW_MEMORY}" != "1" ]; then
    fail "MemAvailable ${mem_gib}GiB is below MIN_MEM_AVAILABLE_GIB=${MIN_MEM_AVAILABLE_GIB}; set ALLOW_LOW_MEMORY=1 only after documenting the launch."
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local free_mib
    free_mib="$(gpu_free_mib || true)"
    if [ -n "${free_mib}" ] && [ "${free_mib}" -lt "${MIN_GPU_FREE_MIB}" ] && [ "${ALLOW_LOW_GPU_MEMORY}" != "1" ]; then
      fail "GPU free memory ${free_mib}MiB is below MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB}; set ALLOW_LOW_GPU_MEMORY=1 only after documenting the launch."
    fi
  fi
}

append_command() {
  printf '%q ' "$@" >>"${SUITE_OUT}/commands.sh"
  printf '\n' >>"${SUITE_OUT}/commands.sh"
}

run_step() {
  local name="$1"
  shift
  local log_path="${SUITE_OUT}/logs/${name}.log"
  append_command "$@"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[dry-run] %s\n' "$*" | tee -a "${log_path}"
    return 0
  fi
  printf '[%s] start %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" | tee -a "${log_path}"
  env CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "$@" >>"${log_path}" 2>&1
  printf '[%s] done %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" | tee -a "${log_path}"
}

run_step_stdout_to_file() {
  local name="$1"
  local stdout_path="$2"
  shift 2
  local log_path="${SUITE_OUT}/logs/${name}.stderr.log"
  append_command "$@"
  if [ "${DRY_RUN}" = "1" ]; then
    printf '[dry-run] %s\n' "$*" | tee -a "${log_path}"
    return 0
  fi
  printf '[%s] start %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" | tee -a "${log_path}"
  env CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}" CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "$@" >"${stdout_path}" 2>>"${log_path}"
  printf '[%s] done %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" | tee -a "${log_path}"
}

read_words() {
  local raw="$1"
  read -r -a WORDS_OUT <<<"${raw}"
}

main() {
  require_under_castle "PROJECT" "${PROJECT}"
  require_under_castle "PYTHON" "${PYTHON}"
  require_under_castle "CHECKPOINT_DIR" "${CHECKPOINT_DIR}"
  require_under_castle "TOKENIZER" "${TOKENIZER}"
  require_under_castle "OFFICIAL_BENCHMARK_ROOT" "${OFFICIAL_BENCHMARK_ROOT}"
  require_under_castle "LIGHT_BENCHMARK_ROOT" "${LIGHT_BENCHMARK_ROOT}"
  require_under_castle "PROMPTS" "${PROMPTS}"
  require_under_castle "SUITE_OUT" "${SUITE_OUT}"

  [ -x "${PYTHON}" ] || fail "missing executable Python env: ${PYTHON}"
  [ -d "${CHECKPOINT_DIR}" ] || fail "missing checkpoint dir: ${CHECKPOINT_DIR}"
  [ -f "${CHECKPOINT_DIR}/config.json" ] || fail "checkpoint is missing config.json: ${CHECKPOINT_DIR}"
  [ -f "${TOKENIZER}" ] || fail "missing tokenizer: ${TOKENIZER}"
  [ -d "${OFFICIAL_BENCHMARK_ROOT}" ] || fail "missing official benchmark root: ${OFFICIAL_BENCHMARK_ROOT}"
  [ -d "${LIGHT_BENCHMARK_ROOT}" ] || fail "missing light benchmark root: ${LIGHT_BENCHMARK_ROOT}"
  [ -s "${PROMPTS}" ] || fail "missing prompts: ${PROMPTS}"

  mkdir -p "${SUITE_OUT}/logs"
  : >"${SUITE_OUT}/commands.sh"
  chmod +x "${SUITE_OUT}/commands.sh"

  if [ "${RUN_PRECHECK}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
    precheck_resources
  fi

  if [ "${RUN_PUBLIC_LIGHT}" = "1" ]; then
    read_words "${PUBLIC_LIGHT_BENCHMARKS}"
    public_args=(
      "${PYTHON}" "${PROJECT}/scripts/evaluate_public_benchmarks.py"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --tokenizer "${TOKENIZER}"
      --benchmark-root "${LIGHT_BENCHMARK_ROOT}"
      --output-root "${SUITE_OUT}/public_benchmarks"
      --registry-path "${SUITE_OUT}/checkpoint_registry.jsonl"
      --run-alias "${RUN_ALIAS}"
      --checkpoint-alias "${CHECKPOINT_ALIAS}"
      --eval-tier "kfri_dmpmaker_public_light"
      --max-examples-per-benchmark "${PUBLIC_LIGHT_MAX_EXAMPLES}"
      --device "${DEVICE}"
      --dtype "${DTYPE}"
      --pair-batch-size "${PUBLIC_LIGHT_PAIR_BATCH_SIZE}"
    )
    for benchmark in "${WORDS_OUT[@]}"; do
      public_args+=(--benchmark "${benchmark}")
    done
    run_step "public_light" "${public_args[@]}"
  fi

  if [ "${RUN_NEXT_TOKEN}" = "1" ]; then
    read_words "${NEXT_TOKEN_BENCHMARKS}"
    next_args=(
      "${PYTHON}" "${PROJECT}/scripts/evaluate_official_hf_next_token_mcq.py"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --tokenizer "${TOKENIZER}"
      --benchmark-root "${OFFICIAL_BENCHMARK_ROOT}"
      --output-root "${SUITE_OUT}/official_hf_scores_fast"
      --run-alias "${RUN_ALIAS}"
      --checkpoint-alias "${CHECKPOINT_ALIAS}"
      --eval-tier "kfri_dmpmaker_official_next_token"
      --device "${DEVICE}"
      --dtype "${DTYPE}"
      --batch-size "${NEXT_TOKEN_BATCH_SIZE}"
    )
    for benchmark in "${WORDS_OUT[@]}"; do
      next_args+=(--benchmark "${benchmark}")
    done
    run_step "official_next_token" "${next_args[@]}"
  fi

  if [ "${RUN_CHOICE_TEXT}" = "1" ]; then
    read_words "${CHOICE_TEXT_BENCHMARKS}"
    choice_args=(
      "${PYTHON}" "${PROJECT}/scripts/evaluate_official_hf_choice_text_mcq.py"
      --checkpoint-dir "${CHECKPOINT_DIR}"
      --tokenizer "${TOKENIZER}"
      --benchmark-root "${OFFICIAL_BENCHMARK_ROOT}"
      --output-root "${SUITE_OUT}/official_hf_choice_text"
      --run-alias "${RUN_ALIAS}"
      --checkpoint-alias "${CHECKPOINT_ALIAS}"
      --eval-tier "kfri_dmpmaker_choice_text"
      --device "${DEVICE}"
      --dtype "${DTYPE}"
      --batch-size "${CHOICE_TEXT_BATCH_SIZE}"
      --max-examples-per-benchmark "${CHOICE_TEXT_MAX_EXAMPLES}"
    )
    for benchmark in "${WORDS_OUT[@]}"; do
      choice_args+=(--benchmark "${benchmark}")
    done
    run_step "choice_text" "${choice_args[@]}"
  fi

  if [ "${RUN_PERMUTATION}" = "1" ]; then
    permuted_root="${SUITE_OUT}/permuted_inputs"
    read_words "${PERMUTATION_BENCHMARKS}"
    build_perm_args=(
      "${PYTHON}" "${PROJECT}/scripts/build_permuted_mcq_benchmarks.py"
      --benchmark-root "${OFFICIAL_BENCHMARK_ROOT}"
      --output-root "${permuted_root}"
    )
    for benchmark in "${WORDS_OUT[@]}"; do
      build_perm_args+=(--benchmark "${benchmark}")
    done
    read_words "${PERMUTATION_SEEDS}"
    for seed in "${WORDS_OUT[@]}"; do
      build_perm_args+=(--seed "${seed}")
    done
    run_step "build_permuted_inputs" "${build_perm_args[@]}"

    read_words "${PERMUTATION_SEEDS}"
    seeds=("${WORDS_OUT[@]}")
    read_words "${PERMUTATION_BENCHMARKS}"
    perm_benchmarks=("${WORDS_OUT[@]}")
    for seed in "${seeds[@]}"; do
      seed_dir="${permuted_root}/seed_$(printf '%04d' "${seed}")"
      perm_args=(
        "${PYTHON}" "${PROJECT}/scripts/evaluate_official_hf_next_token_mcq.py"
        --checkpoint-dir "${CHECKPOINT_DIR}"
        --tokenizer "${TOKENIZER}"
        --benchmark-root "${seed_dir}"
        --output-root "${SUITE_OUT}/official_hf_scores_fast"
        --run-alias "${RUN_ALIAS}"
        --checkpoint-alias "${CHECKPOINT_ALIAS}_permutation_seed${seed}"
        --eval-tier "kfri_dmpmaker_permutation_seed${seed}"
        --device "${DEVICE}"
        --dtype "${DTYPE}"
        --batch-size "${NEXT_TOKEN_BATCH_SIZE}"
      )
      for benchmark in "${perm_benchmarks[@]}"; do
        perm_args+=(--benchmark "${benchmark}")
      done
      run_step "permutation_seed${seed}" "${perm_args[@]}"
    done
  fi

  if [ "${RUN_QUALITATIVE}" = "1" ]; then
    mkdir -p "${SUITE_OUT}/qualitative"
    run_step_stdout_to_file "qualitative_samples" "${SUITE_OUT}/qualitative/summary.json" \
      "${PYTHON}" "${PROJECT}/scripts/generate_checkpoint_samples.py" \
      --checkpoint-dir "${CHECKPOINT_DIR}" \
      --tokenizer "${TOKENIZER}" \
      --prompts "${PROMPTS}" \
      --output "${SUITE_OUT}/qualitative/samples.jsonl" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --max-new-tokens "${QUAL_MAX_NEW_TOKENS}" \
      --temperature 0.0
  fi

  "${PYTHON}" "${PROJECT}/scripts/summarize_official_eval_suite.py" \
    --suite-dir "${SUITE_OUT}" \
    --target-name "${TARGET_NAME}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --tokenizer "${TOKENIZER}" \
    --run-alias "${RUN_ALIAS}" \
    --checkpoint-alias "${CHECKPOINT_ALIAS}" \
    --launch-decision "$(if [ "${DRY_RUN}" = "1" ]; then printf 'dry_run'; else printf 'executed'; fi)" \
    --note "Generated by scripts/evaluate_kfri_dmpmaker_official_suite.sh"

  printf '%s\n' "${SUITE_OUT}"
}

main "$@"
