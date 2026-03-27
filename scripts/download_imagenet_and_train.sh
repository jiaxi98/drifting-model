#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

source ../../venv/aigc/bin/activate

WANDB_NETRC="${WANDB_NETRC:-/home/aiops/zhaojx/.netrc_wandb}"
if [[ -f "$WANDB_NETRC" ]]; then
  export WANDB_API_KEY="$(awk '/password/ {print $2}' "$WANDB_NETRC" | head -n1)"
fi

DATA_ROOT="${DATA_ROOT:-/home/aiops/zhaojx/datasets/imagenet}"
TRAIN_DIR="$DATA_ROOT/train"
mkdir -p "$TRAIN_DIR"
ARCHIVE_PATH="$DATA_ROOT/ILSVRC2012_img_train.tar"
TRAIN_TAR_URL="https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar"
ARCHIVE_DONE_MARK="$DATA_ROOT/.train_archive_downloaded"
CLASS_TAR_DONE_MARK="$DATA_ROOT/.train_class_tars_ready"
IMAGES_DONE_MARK="$DATA_ROOT/.train_extracted_done"
KEEP_ARCHIVE="${KEEP_ARCHIVE:-0}"

if [[ ! -f "$ARCHIVE_DONE_MARK" ]]; then
  echo "[INFO] Downloading ImageNet train archive to $ARCHIVE_PATH (resume enabled)"
  curl --fail --retry 20 --retry-delay 10 -L -C - -o "$ARCHIVE_PATH" "$TRAIN_TAR_URL"
  touch "$ARCHIVE_DONE_MARK"
fi

if [[ ! -f "$CLASS_TAR_DONE_MARK" ]]; then
  echo "[INFO] Extracting top-level class tar archives to $TRAIN_DIR"
  tar -xf "$ARCHIVE_PATH" -C "$TRAIN_DIR"
  class_tar_count="$(find "$TRAIN_DIR" -maxdepth 1 -type f -name '*.tar' | wc -l)"
  if [[ "$class_tar_count" -lt 900 ]]; then
    echo "[ERROR] Expected around 1000 class tar files, found $class_tar_count"
    echo "[ERROR] Archive may be incomplete. Remove $ARCHIVE_DONE_MARK and rerun."
    exit 1
  fi
  touch "$CLASS_TAR_DONE_MARK"
fi

if [[ ! -f "$IMAGES_DONE_MARK" ]]; then
  echo "[INFO] Expanding per-class train tar files"
  shopt -s nullglob
  idx=0
  for class_tar in "$TRAIN_DIR"/*.tar; do
    idx=$((idx + 1))
    class_name="$(basename "$class_tar" .tar)"
    class_dir="$TRAIN_DIR/$class_name"
    mkdir -p "$class_dir"
    tar -xf "$class_tar" -C "$class_dir"
    rm -f "$class_tar"
    if (( idx % 50 == 0 )); then
      echo "[INFO] Expanded $idx class archives..."
    fi
  done
  touch "$IMAGES_DONE_MARK"

  if [[ "$KEEP_ARCHIVE" != "1" ]]; then
    echo "[INFO] Removing downloaded archive $ARCHIVE_PATH"
    rm -f "$ARCHIVE_PATH"
  fi
fi

mkdir -p "$DATA_ROOT/val"

echo "[INFO] Starting 8-GPU ImageNet training"
torchrun --nproc_per_node=8 train.py \
  --dataset imagenet \
  --data_root "$DATA_ROOT" \
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
  --amp
