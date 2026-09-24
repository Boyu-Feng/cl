#!/usr/bin/env bash
set -euo pipefail

# Run from any directory:
#   bash general-knowledge/scripts/train_SFT.sh
#
# Optional overrides, for example:
#   NUM_PROCESSES=1 OUTPUT_DIR=models/iter1-test \
#     bash general-knowledge/scripts/train_SFT.sh
#
# Second SEAL round:
#   MODEL_NAME=models/iter1 \
#   TRAIN_FILE=general-knowledge/data/synthetic_data/EM_SFT/sft_best1of5_iter1.jsonl \
#   OUTPUT_DIR=models/iter2 \
#     bash general-knowledge/scripts/train_SFT.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Activate the requested Conda environment unless it is already active.
CONDA_ENV_NAME="${CONDA_ENV_NAME:-seal_env}"
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
    elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
        CONDA_BASE="${HOME}/miniconda3"
    elif [[ -x "${HOME}/anaconda3/bin/conda" ]]; then
        CONDA_BASE="${HOME}/anaconda3"
    else
        echo "ERROR: conda was not found. Activate ${CONDA_ENV_NAME} before running this script." >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
fi

# Use the locally stored Qwen3 checkpoint by default.
DEFAULT_MODEL_PATH="${PROJECT_ROOT}/../delta-Mem/model/Qwen3-4B-Instruct-2507"
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_PATH}}"
TRAIN_FILE="${TRAIN_FILE:-general-knowledge/data/synthetic_data/EM_SFT/sft_best1of5_iter0.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-models/iter1}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-5}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-3e-4}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
LOG_STEPS="${LOG_STEPS:-1}"
# On a single node, Accelerate treats port 0 as automatic free-port selection.
# Set MAIN_PROCESS_PORT to a specific port to override this behavior.
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
    echo "ERROR: training data not found: ${PROJECT_ROOT}/${TRAIN_FILE}" >&2
    exit 1
fi

if [[ ! -f "${MODEL_NAME}/config.json" ]] && [[ ! -f "${MODEL_NAME}" ]]; then
    echo "ERROR: base model not found: ${MODEL_NAME}" >&2
    exit 1
fi

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if [[ ! "${GPU_COUNT}" =~ ^[0-9]+$ ]] || (( GPU_COUNT < 1 )); then
    echo "ERROR: PyTorch cannot see an NVIDIA GPU. Check nvidia-smi, the driver, and the CUDA-enabled PyTorch installation." >&2
    exit 1
fi

if (( GPU_COUNT >= 2 )); then
    DEFAULT_NUM_PROCESSES=2
else
    DEFAULT_NUM_PROCESSES=1
fi
NUM_PROCESSES="${NUM_PROCESSES:-${DEFAULT_NUM_PROCESSES}}"
if [[ ! "${NUM_PROCESSES}" =~ ^[0-9]+$ ]] || (( NUM_PROCESSES < 1 || NUM_PROCESSES > GPU_COUNT )); then
    echo "ERROR: NUM_PROCESSES=${NUM_PROCESSES}, but PyTorch sees ${GPU_COUNT} GPU(s)." >&2
    exit 1
fi

if [[ -d "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]] \
   && [[ "${ALLOW_OVERWRITE:-0}" != "1" ]]; then
    echo "ERROR: output directory is not empty: ${PROJECT_ROOT}/${OUTPUT_DIR}" >&2
    echo "Choose another OUTPUT_DIR, or set ALLOW_OVERWRITE=1 if overwriting is intentional." >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}" logs

LAUNCH_ARGS=(
    --num_processes "${NUM_PROCESSES}"
    --num_machines 1
    --main_process_port "${MAIN_PROCESS_PORT}"
    --dynamo_backend no
    --mixed_precision bf16
)

# Use the repository's ZeRO-3 configuration when multiple GPUs and DeepSpeed
# are available. Otherwise Accelerate uses ordinary single-GPU/DDP training.
if (( NUM_PROCESSES > 1 )) && python -c 'import deepspeed' >/dev/null 2>&1; then
    LAUNCH_ARGS+=(
        --deepspeed_config_file general-knowledge/src/EM/config/deepspeed_stage3.json
    )
fi

CMD=(
    accelerate launch
    "${LAUNCH_ARGS[@]}"
    general-knowledge/src/EM/train_SFT.py
    --train_file "${TRAIN_FILE}"
    --model_name_or_path "${MODEL_NAME}"
    --output_dir "${OUTPUT_DIR}"
    --per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --gradient_accumulation_steps "${GRAD_ACC}"
    --num_train_epochs "${EPOCHS}"
    --learning_rate "${LR}"
    --lora_rank "${LORA_RANK}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
    --lora_target_modules "${LORA_TARGET_MODULES}"
    --logging_steps "${LOG_STEPS}"
)

echo "SEAL outer-loop SFT"
echo "  project:       ${PROJECT_ROOT}"
echo "  environment:   ${CONDA_DEFAULT_ENV:-unknown}"
echo "  model:         ${MODEL_NAME}"
echo "  training data: ${TRAIN_FILE}"
echo "  output:        ${OUTPUT_DIR}"
echo "  GPUs:          ${NUM_PROCESSES}/${GPU_COUNT}"
echo "  launch port:   ${MAIN_PROCESS_PORT} (0 = auto)"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"

echo "Training finished. Merged model saved to ${PROJECT_ROOT}/${OUTPUT_DIR}"
