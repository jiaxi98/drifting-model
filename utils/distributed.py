"""Distributed training helpers."""

import os
from typing import Dict, Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[bool, int, int, int]:
    """Initialize DDP if launched with torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    return True, rank, world_size, local_rank


def cleanup_distributed(distributed: bool) -> None:
    """Destroy DDP process group after training."""
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_info(
    info: Dict[str, float],
    device: torch.device,
    distributed: bool,
    world_size: int,
) -> Dict[str, float]:
    """Average scalar metrics across DDP ranks."""
    if not distributed:
        return info
    keys = sorted(info.keys())
    vec = torch.tensor([float(info[k]) for k in keys], dtype=torch.float32, device=device)
    dist.all_reduce(vec, op=dist.ReduceOp.SUM)
    vec = vec / world_size
    return {k: vec[i].item() for i, k in enumerate(keys)}
