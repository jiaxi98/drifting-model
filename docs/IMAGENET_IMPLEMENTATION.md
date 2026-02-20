# ImageNet Minimal Implementation Notes

This document summarizes the minimal implementation work added to support ImageNet-scale drifting-model training with CFG-aware loss.

## What Was Added

## 1) ImageNet-capable model presets
- `model.py`
  - Added `DriftDiT-B16` and `DriftDiT-L16` (pixel-space style).
  - Added `DriftDiT-B2` and `DriftDiT-L2` (latent-style patch-2 variants).
  - Set register-token default to `16` to align with paper settings.
  - Fixed CFG pairing so conditional/unconditional branches share identical sampled style-token indices.
- `__init__.py`
  - Exported new model constructors.

## 2) CFG-aware drifting field support
- `drifting.py`
  - `compute_V(...)` now supports optional `neg_weights` for weighted negatives.
  - Self-mask logic supports cases where negatives are `[generated, unconditional]`.
  - `compute_V_multi_temperature(...)` passes optional negative weights.
  - Fixed stale `normalize_features(...)` call path inside `DriftingLoss`.

## 3) Queue infrastructure for unconditional negatives
- `utils.py`
  - Queue sampling is now strictly without replacement.
  - Added `GlobalSampleQueue` for unlabeled/unconditional real samples.

## 4) Feature encoder path for ImageNet
- `feature_encoder.py`
  - Pretrained encoder now supports selectable backbones (`resnet18`, `resnet50`).
  - Added ImageNet branch in `create_feature_encoder(...)`.

## 5) Training pipeline rewrite for ImageNet + DDP
- `train.py`
  - Added dataset support for `imagenet` (ImageFolder `train/` and `val/`).
  - Added DDP (`torchrun`) support with NCCL setup/cleanup.
  - Added CFG-aware loss path using unconditional queue and weighted negatives.
  - Added alpha sampling modes (`uniform`, `power`, `mixed`) and `alpha -> w` mapping.
  - Added optional AMP (`--amp`).
  - Added optional W&B integration (`--wandb`, `--wandb_key_file`).
  - Added ImageNet default config (`DriftDiT-B16`, `img_size=256`, `num_classes=1000`).

## 6) Sampling updates
- `eval.py`
  - Added `imagenet` option.
  - Added `--max_grid_classes` to avoid huge 1000-class grids.
  - Reads model/image/class settings from checkpoint config when available.

## How To Launch Training (8 GPUs)

```bash
source ../../venv/aigc/bin/activate
export WANDB_API_KEY="$(awk '/password/ {print $2}' /home/aiops/zhaojx/.netrc_wandb)"
torchrun --nproc_per_node=8 train.py \
  --dataset imagenet \
  --data_root /path/to/imagenet \
  --output_dir outputs \
  --wandb --wandb_project drifting-model \
  --amp
```

or use the bundled launcher:

```bash
bash scripts/download_imagenet_and_train.sh
```

## Dataset Layout Expected

```text
/path/to/imagenet/
  train/<class_name>/*.JPEG
  val/<class_name>/*.JPEG
```

## Known Simplifications

- This is a minimal implementation, not a full exact reproduction of all paper-scale engineering.
- No latent VAE encode/decode pipeline is wired yet for the `/2` latent variant.
- Feature-loss extraction is simplified relative to full A.5 multi-location recipe.

## TODO (Deferred)

- [ ] Replace current ImageNet feature encoder path with paper-style MAE-pretrained ResNet encoder setup (Table 8, A.3).
- [ ] Implement full A.5 multi-scale feature extraction (per-location vectors, global mean/std, 2x2 and 4x4 pooled stats, plus vanilla term).
- [ ] Add paper-consistent pixel-space feature fusion path (`ResNet + ConvNeXt-V2`) and checkpoint loading.
- [ ] Validate ImageNet runs with paper-aligned encoder settings before reporting reproduction-quality metrics.
