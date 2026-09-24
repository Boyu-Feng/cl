#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname -- "$(readlink -f -- "$0")")/.."
RUN_ROOT="${RUN_ROOT:-$PWD/model/qasper_tsw_remote}"
LOG_FILE="${LOG_FILE:-/tmp/qasper_tsw_remote.nohup.log}"
GPU_LIST="${GPU_LIST:-0,1,2,3,5}"
GPU_COUNT="${GPU_COUNT:-5}"

mkdir -p "$RUN_ROOT/logs"
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "ERROR: CUDA GPUs are not visible on this node; refusing to start training." >&2
  echo "Requested CUDA_VISIBLE_DEVICES=$GPU_LIST" >&2
  exit 2
fi
VISIBLE_GPU_COUNT=$(nvidia-smi -L | wc -l)
if (( GPU_COUNT > VISIBLE_GPU_COUNT )); then
  echo "ERROR: requested $GPU_COUNT processes but only $VISIBLE_GPU_COUNT GPUs are visible." >&2
  exit 2
fi
IFS=, read -r -a REQUESTED_GPUS <<< "$GPU_LIST"
if (( ${#REQUESTED_GPUS[@]} != GPU_COUNT )); then
  echo "ERROR: GPU_COUNT=$GPU_COUNT but GPU_LIST=$GPU_LIST contains ${#REQUESTED_GPUS[@]} entries." >&2
  exit 2
fi
for gpu in "${REQUESTED_GPUS[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]] || (( gpu >= VISIBLE_GPU_COUNT )); then
    echo "ERROR: GPU $gpu is not visible; available GPU indices are 0-$((VISIBLE_GPU_COUNT - 1))." >&2
    exit 2
  fi
done
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export NPROC_PER_NODE="$GPU_COUNT"
export PIPELINE_ROOT="$RUN_ROOT"
export LOG_ROOT="$RUN_ROOT/logs"
export MANIFEST_PATH="$RUN_ROOT/run_manifest.txt"
export TSW_OUTPUT_DIR="$RUN_ROOT/tsw_output"
export DATASET_NAME="huaXiaKyrie/delta-mem-qasper-data"
export DATASET_SPLIT="train"
export TRAIN_FILE=""
export ATTN_IMPLEMENTATION="flash_attention_2"
export TRAIN_VARIANTS_STRING="TSW_rank8_qasper_write8192"
export BENCHMARK_VARIANTS_STRING="TSW_rank8_qasper_write8192"

echo "Launching TSW with GPUs=$CUDA_VISIBLE_DEVICES (NPROC_PER_NODE=$NPROC_PER_NODE)"
echo "Dataset=$DATASET_NAME (local TRAIN_FILE disabled)"
echo "Log=$LOG_FILE"
nohup bash scripts/run_qasper_multimodel_write8192_train_and_benchmark_suite.sh >"$LOG_FILE" 2>&1 &
echo "PID=$!"
