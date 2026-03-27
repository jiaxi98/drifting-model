#!/usr/bin/env bash
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source ../../venv/aigc/bin/activate

WANDB_NETRC="/home/aiops/zhaojx/.netrc_wandb"
export WANDB_API_KEY="$(awk '/password/ {print $2}' "$WANDB_NETRC" | head -n1 | tr -d '[:space:]')"

export WANDB_DIR="$REPO_ROOT/wandb"
export WANDB_CACHE_DIR="$REPO_ROOT/wandb/.cache"
export WANDB_CONFIG_DIR="$REPO_ROOT/wandb/.config"
mkdir -p logs "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

RUN_TAG="$(date +%F_%H%M%S)"
LOG_FILE="logs/imagenet_train_${RUN_TAG}.log"
SYS_LOG_FILE="logs/imagenet_sys_${RUN_TAG}.log"

echo "[$(date '+%F %T')] starting system monitor -> $SYS_LOG_FILE"
(
  while true; do
    echo "[$(date '+%F %T')] === system snapshot ==="
    free -h
    echo "--- top rss processes ---"
    ps -eo pid,ppid,comm,%mem,rss,args --sort=-rss | head -n 12
    echo "--- gpu snapshot ---"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
    echo
    sleep 15
  done
) > "$SYS_LOG_FILE" 2>&1 &
MON_PID=$!

cleanup() {
  kill "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

export OMP_NUM_THREADS=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

echo "[$(date '+%F %T')] launching torchrun -> $LOG_FILE"
set +e
stdbuf -oL -eL torchrun --nproc_per_node=8 train.py \
  --dataset imagenet \
  --data_root /home/aiops/zhaojx/datasets/imagenet \
  --output_dir outputs \
  --model DriftDiT-B16 \
  --batch_nc 16 \
  --batch_n_neg 4 \
  --batch_n_pos 8 \
  --batch_n_uncond 16 \
  --loader_batch_size 64 \
  --num_workers 8 \
  --wandb \
  --wandb_project drifting-model \
  --amp \
  --log_interval 1 \
  2>&1 | awk '{print strftime("[%F %T]"), $0; fflush();}' | tee -a "$LOG_FILE"
TORCH_EXIT=${PIPESTATUS[0]}
set -e

echo "[$(date '+%F %T')] torchrun exit code: $TORCH_EXIT" | tee -a "$LOG_FILE"
exit "$TORCH_EXIT"
