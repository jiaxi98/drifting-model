# Repository Guidelines

## Project Structure & Module Organization
This repository is a compact PyTorch project with source files at the repo root:
- `train.py`: end-to-end training entrypoint
- `eval.py`: one-step sampling, grids, and optional FID/IS
- `model.py`, `drifting.py`, `feature_encoder.py`, `utils.py`: core model/loss/utility modules
- `__init__.py`: package exports
- `assets/`: reference sample images

Generated artifacts belong in `outputs/` or `samples/` and should stay untracked (`.gitignore` already excludes checkpoints and output folders).

## Build, Test, and Development Commands
Use Python directly from the repository root (imports assume this layout).

```bash
pip install torch torchvision einops
# optional for FID/IS in eval.py
pip install torchmetrics
```

```bash
python train.py --dataset mnist --output_dir outputs/mnist
python train.py --dataset cifar --output_dir outputs/cifar
python eval.py --checkpoint outputs/mnist/checkpoint_final.pt --dataset mnist --output_dir samples/mnist
python eval.py --checkpoint outputs/cifar/checkpoint_final.pt --dataset cifar --compute_fid
```

Quick sanity check before opening a PR:

```bash
python -m py_compile *.py
```

## Coding Style & Naming Conventions
Follow existing Python style:
- 4-space indentation, PEP 8 spacing, readable line lengths
- `snake_case` for functions/variables, `PascalCase` (and existing `DriftDiT_*` pattern) for classes
- Type hints and concise docstrings for public functions
- Keep training/sampling scripts thin; move reusable logic into modules

## Testing Guidelines
There is no dedicated `tests/` suite yet. Treat smoke validation as required:
- Syntax check with `python -m py_compile *.py`
- Run at least one training command and one sampling command relevant to your change
- For model behavior changes, attach updated sample grids and key metrics (for example, FID when used)

If you add automated tests, use `pytest` with files named `tests/test_*.py`.

## Configuration Tips
`train.py` currently defaults dataset loading to `/home/qingtianzhu.ty/drifting/data` inside `get_dataset(...)`. Update that path (or adapt the function) for your environment before running long jobs.

## Commit & Pull Request Guidelines
Recent history uses short, direct subjects (for example, `cifar bug fix`, `Update README...`). Keep commits focused and message lines concise.

PRs should include:
- What changed and why
- Exact commands used to validate
- Dataset/checkpoint context and environment notes (GPU/CPU)
- Before/after outputs for model-quality changes (image paths or metrics)
- Linked issue (if applicable)
