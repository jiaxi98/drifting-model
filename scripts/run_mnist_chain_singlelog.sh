#!/usr/bin/env bash
set -euo pipefail

# Sequential dataset batch-size runs with exactly one kept log per batch.
# For each batch size:
# 1) run the sweep helper for that single batch
# 2) keep only logs/{dataset}_bs{batch}.log

if [[ -z "${NPROC:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC=1
  fi
fi
if [[ "${NPROC}" -lt 1 ]]; then
  NPROC=1
fi
MAX_STEPS="${MAX_STEPS:-10000}"
EPOCHS="${EPOCHS:-500}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATASET="${DATASET:-mnist}"
DATASET_TAG="${DATASET,,}"

if [[ "$#" -gt 0 ]]; then
  BATCHES=("$@")
else
  if [[ "${DATASET_TAG}" == "mnist" ]]; then
    BATCHES=(64 128 256 512 1024 2048)
  elif [[ "${DATASET_TAG}" == "cifar" ]]; then
    BATCHES=(32 64 128 256 512 1024)
  else
    echo "[$(date '+%F %T')] ERROR: unsupported DATASET=${DATASET}"
    exit 1
  fi
fi

mkdir -p logs

wait_for_idle() {
  while pgrep -f "torchrun --nproc_per_node=${NPROC} train.py --dataset ${DATASET}" >/dev/null; do
    echo "[$(date '+%F %T')] Waiting for existing ${DATASET} torchrun to finish..."
    sleep 15
  done
}

for bs in "${BATCHES[@]}"; do
  echo "[$(date '+%F %T')] ===== Start batch size ${bs} ====="

  wait_for_idle

  # Keep exactly one output file per batch size.
  rm -f "logs/${DATASET_TAG}_bs${bs}.log" "logs/${DATASET_TAG}_bs${bs}_runner.log"
  find logs -maxdepth 1 -type f -name "${DATASET_TAG}_ddp_bs${bs}_*.log" -delete

  python scripts/run_mnist_batch_sweep.py \
    --dataset "${DATASET}" \
    --nproc "${NPROC}" \
    --max_steps "${MAX_STEPS}" \
    --epochs "${EPOCHS}" \
    --num_workers "${NUM_WORKERS}" \
    --batches "${bs}" | tee "logs/${DATASET_TAG}_bs${bs}_runner.log"

  latest_ddp_log="$(ls -1t logs/${DATASET_TAG}_ddp_bs${bs}_*.log 2>/dev/null | head -n1 || true)"
  if [[ -z "${latest_ddp_log}" ]]; then
    echo "[$(date '+%F %T')] ERROR: no ddp log found for bs=${bs}"
    exit 1
  fi

  mv "${latest_ddp_log}" "logs/${DATASET_TAG}_bs${bs}.log"
  find logs -maxdepth 1 -type f -name "${DATASET_TAG}_ddp_bs${bs}_*.log" -delete
  rm -f "logs/${DATASET_TAG}_bs${bs}_runner.log"
  find logs -maxdepth 1 -type f -name "${DATASET_TAG}_batch_sweep_summary_*.txt" -delete

  echo "[$(date '+%F %T')] ===== Finished batch size ${bs}, kept logs/${DATASET_TAG}_bs${bs}.log ====="
done

echo "[$(date '+%F %T')] All requested batch sizes finished."
