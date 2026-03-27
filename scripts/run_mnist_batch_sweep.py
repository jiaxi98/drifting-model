#!/usr/bin/env python3
"""Run DDP batch-size sweep and monitor loss evolution."""

from __future__ import annotations

import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


STEP_RE = re.compile(r"Step\s+(\d+)\s+\|\s+Loss:\s+([0-9]*\.?[0-9]+)")


def should_stop_for_plateau(losses: list[float], min_step: int, step: int) -> tuple[bool, str]:
    """
    Detect oscillation with little improvement.

    Uses two equal windows:
    - previous window vs recent window mean improvement
    - recent window std as oscillation indicator
    """
    if step < min_step:
        return False, ""
    # NOTE: losses are logged every `log_interval` steps (default 20), not every step.
    # Use window in "logged points" instead of raw steps.
    window = 50  # ~= 1000 steps when log_interval=20
    if len(losses) < window * 2:
        return False, ""

    prev = losses[-2 * window : -window]
    recent = losses[-window:]
    prev_mean = sum(prev) / len(prev)
    recent_mean = sum(recent) / len(recent)
    improvement = prev_mean - recent_mean
    recent_std = statistics.pstdev(recent)

    # Heuristic thresholds tuned for this project's scale (~7-9 loss on MNIST).
    if improvement < 0.01 and recent_std > 0.01:
        return True, (
            f"plateau detected (step={step}, prev_mean={prev_mean:.4f}, "
            f"recent_mean={recent_mean:.4f}, improvement={improvement:.4f}, std={recent_std:.4f})"
        )
    return False, ""


def run_one(
    dataset: str,
    batch_size: int,
    nproc: int,
    max_steps: int,
    epochs: int,
    num_workers: int,
    log_dir: Path,
    output_dir: Path,
    queue_size: int | None,
) -> dict:
    dataset_tag = dataset.lower()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{dataset_tag}_ddp_bs{batch_size}_{ts}"
    log_path = log_dir / f"{run_name}.log"

    effective_queue_size = queue_size if queue_size is not None else max(256, batch_size)

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "train.py",
        "--dataset",
        dataset,
        "--output_dir",
        str(output_dir),
        "--batch_n_pos",
        str(batch_size),
        "--batch_n_neg",
        str(batch_size),
        "--queue_size",
        str(effective_queue_size),
        "--epochs",
        str(epochs),
        "--max_steps",
        str(max_steps),
        "--num_workers",
        str(num_workers),
        "--log_interval",
        "20",
        "--save_interval",
        "100000",
        "--sample_interval",
        "100000",
    ]

    print(f"\n[{datetime.now()}] START {run_name}")
    print("CMD:", " ".join(cmd))
    print("LOG:", log_path)

    losses: list[float] = []
    last_step = 0
    stop_reason = f"reached max_steps={max_steps}"
    terminated_by_monitor = False

    with log_path.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert proc.stdout is not None

        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

            m = STEP_RE.search(line)
            if m:
                last_step = int(m.group(1))
                loss = float(m.group(2))
                losses.append(loss)

                stop, reason = should_stop_for_plateau(losses, min_step=2000, step=last_step)
                if stop:
                    terminated_by_monitor = True
                    stop_reason = reason
                    print(f"\n[{datetime.now()}] STOP EARLY {run_name}: {reason}")
                    proc.terminate()
                    break

        try:
            ret = proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            ret = proc.wait()

    print(f"[{datetime.now()}] END {run_name} (return={ret}, last_step={last_step})")

    tail_k = min(30, len(losses))
    tail_losses = losses[-tail_k:] if tail_k > 0 else []
    converged_mean = statistics.mean(tail_losses) if tail_losses else None
    converged_median = statistics.median(tail_losses) if tail_losses else None

    if losses:
        print(
            f"[{datetime.now()}] METRICS {run_name}: "
            f"best_loss={min(losses):.4f}, final_loss={losses[-1]:.4f}, "
            f"converged_mean(last{tail_k})={converged_mean:.4f}, "
            f"converged_median(last{tail_k})={converged_median:.4f}"
        )

    return {
        "run_name": run_name,
        "batch_size": batch_size,
        "log_path": str(log_path),
        "last_step": last_step,
        "return_code": ret,
        "terminated_by_monitor": terminated_by_monitor,
        "stop_reason": stop_reason,
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "converged_tail_k": tail_k if losses else None,
        "converged_mean": converged_mean,
        "converged_median": converged_median,
    }


def main():
    parser = argparse.ArgumentParser(description="DDP batch-size sweep with monitoring")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--queue_size",
        type=int,
        default=None,
        help="Per-class queue size override; default=max(256, batch_size)",
    )
    args = parser.parse_args()

    dataset_tag = args.dataset.lower()
    if args.batches is None:
        if dataset_tag == "mnist":
            args.batches = [64, 128, 256, 512, 1024, 2048]
        elif dataset_tag == "cifar":
            args.batches = [32, 64, 128, 256, 512, 1024]
        else:
            raise ValueError(f"Unsupported dataset for defaults: {args.dataset}")

    if args.output_dir is None:
        args.output_dir = f"outputs/{dataset_tag}_batch_sweep"

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for bs in args.batches:
        result = run_one(
            dataset=args.dataset,
            batch_size=bs,
            nproc=args.nproc,
            max_steps=args.max_steps,
            epochs=args.epochs,
            num_workers=args.num_workers,
            log_dir=log_dir,
            output_dir=output_dir,
            queue_size=args.queue_size,
        )
        summary.append(result)

    summary_path = log_dir / f"{dataset_tag}_batch_sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        for s in summary:
            f.write(
                f"bs={s['batch_size']} run={s['run_name']} step={s['last_step']} "
                f"ret={s['return_code']} stop='{s['stop_reason']}' "
                f"best_loss={s['best_loss']} final_loss={s['final_loss']} "
                f"converged_mean(last{s['converged_tail_k']})={s['converged_mean']} "
                f"converged_median(last{s['converged_tail_k']})={s['converged_median']} "
                f"log={s['log_path']}\n"
            )
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
