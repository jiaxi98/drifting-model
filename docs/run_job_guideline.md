# Job Running Guideline

This file is a practical checklist for running training jobs in this repo.

## 1) Preflight (must do)
- Activate env:
  - `source ../../venv/aigc/bin/activate`
- Verify key dependencies:
  - `python -V`
  - `pip show torch torchvision torchmetrics torch-fidelity`
- Check accidental debugger stops:
  - `rg -n "breakpoint\\(" train.py drifting.py eval.py feature_encoder.py`
- Check GPU status first:
  - `nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`

## 2) W&B rules
- Always enable wandb log in, current venv should already log into the wandb, otherwise please contact human supervisor.
- W&B key is in `~/.netrc_wandb` (netrc format, not raw key file).
- Recommended login command:
  - `WANDB_KEY=$(awk '($1=="password"){print $2; exit}' ~/.netrc_wandb)`
  - `wandb login --relogin "$WANDB_KEY"`
- In launch scripts, export key explicitly to avoid login drift:
  - `export WANDB_API_KEY="$WANDB_KEY"`
- Warning like `permission denied ~/.cache/wandb/logs/...` may appear; run can still sync.

## 3) Output/checkpoint behavior
- Training writes to: `output_dir/<dataset>/...`
  - Example: `--output_dir outputs/cifar_exp` -> `outputs/cifar_exp/cifar/checkpoint.pt`
- Checkpoint name is `checkpoint.pt` (overwritten each save interval).

## 4) Recommended launch commands
It is highly recommended to init the job in a tmux session to avoid killing it by accident.
### 4.1 MNIST on 8x A100 + W&B (DDP)
```bash
cd /home/aiops/zhaojx/projects/drifting-model
source ../../venv/aigc/bin/activate
WANDB_KEY=$(awk '($1=="password"){print $2; exit}' ~/.netrc_wandb)
export WANDB_API_KEY="$WANDB_KEY"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

RUN_NAME="mnist_8gpu_$(date +%m%d_%H%M%S)"
LOG_FILE="logs/${RUN_NAME}.log"
OUT_DIR="outputs/mnist_8gpu"
mkdir -p logs "$OUT_DIR"

setsid ../../venv/aigc/bin/torchrun --standalone --nproc_per_node=8 train.py \
  --dataset mnist \
  --output_dir "$OUT_DIR" \
  --wandb \
  --wandb_project drifting-model \
  --wandb_run_name "$RUN_NAME" \
  --wandb_key_file /home/aiops/zhaojx/.netrc_wandb \
  > "$LOG_FILE" 2>&1 < /dev/null &
echo "RUN_NAME=$RUN_NAME"
echo "LOG_FILE=$LOG_FILE"
```
If running with less than 8 GPU, please avoid using GPU0.

### 4.2 Single-GPU quick smoke
```bash
CUDA_VISIBLE_DEVICES=1 python train.py --dataset cifar --output_dir outputs/cifar_smoke --log_interval 1
```

## 5) Monitoring checklist (first 10-20 steps)
- Process exists:
  - `pgrep -f "<run_name>"`
- GPU actually used:
  - `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits`
- Log tail:
  - `tail -n 120 logs/<run_name>.log`
- For real-time metric lines (sometimes less buffered), also check:
  - `tail -n 120 wandb/run-*/files/output.log`

## 6) Stop job safely
- Find PIDs:
  - `pgrep -f "<run_name>"`
- Stop:
  - `kill <pid1> <pid2> ...`

## 7) Common failure signatures
- Symptom: ImageNet training breaks without even entering model inference
  Cause: Very big image queue for all ranks and all classes stored on CPU, can cost around 700G
  Action: Reduce the queue size (both the unconditional and positive samples)
- Symptom: Run name says `bs256`, but training behavior does not match expected `n_pos/n_neg=256`
  Cause: Only `loader_batch_size` was changed to 256; `batch_n_pos` and `batch_n_neg` stayed at old values.
  Action: Always verify these three fields separately before and after launch:
  - `loader_batch_size`: dataloader mini-batch only
  - `batch_n_pos`: positive samples per selected class in drifting loss
  - `batch_n_neg`: generated/negative samples per selected class in drifting loss
  - Quick check command: `rg -n "loader_batch_size|batch_n_pos|batch_n_neg" config/datasets/*.yaml`
- Symptom: Long run disappears without Python traceback, and logs/checkpoints are hard to map back to the intended job
  Cause: Job launched from interactive shell (not tmux), or run/session/log names are ad-hoc and inconsistent.
  Action: Always launch in tmux with deterministic naming:
  - `RUN_NAME="<dataset>_<key_settings>_$(date +%Y%m%d_%H%M%S)"`
  - `SESSION="${RUN_NAME}_tmux"`
  - `LOG_FILE="logs/${RUN_NAME}.log"`
  - `OUT_DIR="outputs/<exp_dir>"`
- Use the following arguments:
  ```
  --wandb_fid_interval 10 \
  --log_interval 1
  ```
- If using gradient checkpointing, tune recompute cost with:
  ```
  --grad_checkpointing \
  --grad_ckpt_every_n_blocks 2
  ```
  `1` means checkpoint every transformer block; larger values checkpoint fewer blocks (faster, more memory).
