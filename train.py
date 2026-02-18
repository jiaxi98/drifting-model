import argparse
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.factory import get_dataset
from drifting import compute_V_from_dists
from utils.config import DATASET_CONFIG_FILES, get_default_config
from utils.distributed import cleanup_distributed, reduce_info, setup_distributed
from feature_encoder import create_feature_encoder, extract_feature_fullsets, extract_feature_sets
from model import DriftDiT_models
from sample import compute_fid_score, compute_is_score, generate_class_grid
from utils import (
    EMA,
    GlobalSampleQueue,
    SampleQueue,
    WarmupLRScheduler,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    save_image_grid,
    set_seed,
)


def sample_alpha(
    n: int,
    alpha_min: float,
    alpha_max: float,
    mode: str,
    power: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample alpha values from configured distribution."""
    if mode == "uniform":
        return torch.empty(n, device=device).uniform_(alpha_min, alpha_max)

    if mode == "power":
        u = torch.rand(n, device=device)
        if abs(power - 1.0) < 1e-6:
            amin = math.log(alpha_min)
            amax = math.log(alpha_max)
            return torch.exp(amin + u * (amax - amin))
        expo = 1.0 - power
        amin = alpha_min ** expo
        amax = alpha_max ** expo
        return (amin + u * (amax - amin)).pow(1.0 / expo)

    if mode == "mixed":
        alpha = sample_alpha(n, alpha_min, alpha_max, "power", power, device)
        choose_one = torch.rand(n, device=device) < 0.5
        alpha[choose_one] = 1.0
        return alpha

    raise ValueError(f"Unsupported alpha sampling mode: {mode}")


def alpha_to_uncond_weight(alpha: torch.Tensor, n_neg: int, n_uncond: int) -> torch.Tensor:
    """Compute uncond negative weight w from CFG alpha."""
    if n_uncond <= 0:
        return torch.zeros_like(alpha)
    denom = max(n_neg - 1, 1)
    return ((alpha - 1.0) * denom / n_uncond).clamp(min=0.0)


def compute_drifting_loss(
    x_gen: torch.Tensor,
    labels_gen: torch.Tensor,
    class_labels: torch.Tensor,
    class_alphas: torch.Tensor,
    x_pos: torch.Tensor,
    labels_pos: torch.Tensor,
    x_uncond: Optional[torch.Tensor],
    feature_encoder: Optional[nn.Module],
    vae_decoder: Optional[nn.Module],
    temperatures: List[float],
    n_neg: int,
    n_uncond: int,
    use_pixel_space: bool,
    use_full_feature_set: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """CFG-aware drifting loss (minimal implementation)."""
    device = x_gen.device

    feature_fn = extract_feature_fullsets if use_full_feature_set else extract_feature_sets
    feat_gen_list = feature_fn(x_gen, feature_encoder, vae_decoder, use_pixel_space)
    feat_pos_list = feature_fn(x_pos, feature_encoder, vae_decoder, use_pixel_space)
    feat_uncond_list = (
        feature_fn(x_uncond, feature_encoder, vae_decoder, use_pixel_space)
        if x_uncond is not None
        else []
    )

    total_loss = torch.tensor(0.0, device=device)
    Z_pos = 0.0
    Z_neg = 0.0
    n_terms = 0

    for class_idx, class_id in enumerate(class_labels.tolist()):
        alpha_c = class_alphas[class_idx]
        uncond_w = alpha_to_uncond_weight(alpha_c.unsqueeze(0), n_neg=n_neg, n_uncond=n_uncond)[0]
        mask_gen = labels_gen == class_id
        mask_pos = labels_pos == class_id
        if not mask_gen.any() or not mask_pos.any():
            continue

        for feature_idx, (feat_gen, feat_pos) in enumerate(zip(feat_gen_list, feat_pos_list)):
            feat_gen_c = feat_gen[mask_gen]
            feat_pos_c = feat_pos[mask_pos]
            if feat_gen_c.shape[0] == 0 or feat_pos_c.shape[0] == 0:
                continue

            if len(feat_uncond_list) > 0:
                feat_uncond = feat_uncond_list[feature_idx]
                feat_neg = torch.cat([feat_gen_c, feat_uncond], dim=0)
                neg_weights = torch.cat([
                    torch.ones(feat_gen_c.shape[0], device=device),
                    torch.full((feat_uncond.shape[0],), uncond_w.item(), device=device),
                ], dim=0)
            else:
                feat_neg = feat_gen_c
                neg_weights = None

            # Reuse the same pairwise distances for A.6 feature scale and all temperatures.
            dist_pos_raw = torch.cdist(feat_gen_c, feat_pos_c, p=2)
            dist_neg_raw = torch.cdist(feat_gen_c, feat_neg, p=2)
            c_j = float(feat_gen_c.shape[-1])

            with torch.no_grad():
                # S_j = (1/sqrt(C_j)) * E[||phi_j(x)-phi_j(y)||], with stop-grad.
                pos_sum = dist_pos_raw.sum()
                pos_count = dist_pos_raw.new_tensor(float(dist_pos_raw.numel()))
                if neg_weights is None:
                    neg_sum = dist_neg_raw.sum()
                    neg_count = dist_neg_raw.new_tensor(float(dist_neg_raw.numel()))
                else:
                    safe_weights = neg_weights.clamp_min(0.0)
                    neg_sum = (dist_neg_raw * safe_weights.unsqueeze(0)).sum()
                    neg_count = safe_weights.sum() * float(dist_neg_raw.shape[0])
                avg_dist = (pos_sum + neg_sum) / (pos_count + neg_count + 1e-12)
                s_j = avg_dist / math.sqrt(c_j)
                s_j = s_j.clamp_min(1e-6)

            feat_gen_n = feat_gen_c / s_j
            feat_pos_n = feat_pos_c / s_j
            feat_neg_n = feat_neg / s_j
            dist_pos = dist_pos_raw / s_j
            dist_neg = dist_neg_raw / s_j

            v_total = torch.zeros_like(feat_gen_n)
            for tau in temperatures:
                v_tau, a_pos, a_neg = compute_V_from_dists(
                    dist_pos=dist_pos,
                    dist_neg=dist_neg,
                    y_pos=feat_pos_n,
                    y_neg=feat_neg_n,
                    temperature=tau * math.sqrt(c_j),
                    mask_self=True,
                    neg_weights=neg_weights,
                )
                Z_pos += float(torch.mean(torch.sum(a_pos, dim=-1)).item())
                Z_neg += float(torch.mean(torch.sum(a_neg, dim=-1)).item())
                lambda_j = torch.sqrt(torch.mean(torch.sum(v_tau**2, dim=-1) / c_j) + 1e-8)
                v_tau = v_tau / lambda_j
                v_total = v_total + v_tau

            target = (feat_gen_n + v_total).detach()
            # NOTE: loss term is consistent with normalization of v_tau
            loss_term = F.mse_loss(feat_gen_n, target)
            total_loss = total_loss + loss_term
            n_terms += 1

    if n_terms == 0:
        # Keep graph valid for backward while avoiding divide-by-zero.
        zero_loss = x_gen.sum() * 0.0
        return zero_loss, {"loss": 0.0, "Z_pos": 0.0, "Z_neg": 0.0}

    loss = total_loss / n_terms
    info = {
        "loss": float(loss.item()),
        "Z_pos": Z_pos / n_terms,
        "Z_neg": Z_neg / n_terms,
    }
    return loss, info


def train_step(
    model_for_train: nn.Module,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupLRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    ema: EMA,
    class_queue: SampleQueue,
    uncond_queue: Optional[GlobalSampleQueue],
    config: Dict[str, Any],
    device: torch.device,
    feature_encoder: Optional[nn.Module],
    vae_decoder: Optional[nn.Module] = None,
) -> Optional[Dict[str, float]]:
    """Run one optimization step. Returns None if queues are not ready."""
    if not class_queue.is_ready(config["batch_n_pos"]):
        return None
    if uncond_queue is not None and not uncond_queue.is_ready(config["batch_n_uncond"]):
        return None

    model_for_train.train()
    batch_nc = config["batch_nc"]
    n_pos = config["batch_n_pos"]
    n_neg = config["batch_n_neg"]
    n_uncond = config["batch_n_uncond"]

    if batch_nc <= config["num_classes"]:
        class_labels = torch.randperm(config["num_classes"], device=device)[:batch_nc]
    else:
        class_labels = torch.randint(0, config["num_classes"], (batch_nc,), device=device)
    class_alphas = sample_alpha(
        batch_nc,
        alpha_min=config["alpha_min"],
        alpha_max=config["alpha_max"],
        mode=config["alpha_sampling"],
        power=config["alpha_power"],
        device=device,
    )
    labels_gen = class_labels.repeat_interleave(n_neg)
    alpha_gen = class_alphas.repeat_interleave(n_neg)
    x_pos_list = []
    y_pos_list = []
    for c in class_labels.tolist():
        x_c = class_queue.sample(c, n_pos, device=device)
        y_c = torch.full((n_pos,), c, device=device, dtype=torch.long)
        x_pos_list.append(x_c)
        y_pos_list.append(y_c)
    x_pos = torch.cat(x_pos_list, dim=0)
    labels_pos = torch.cat(y_pos_list, dim=0)
    x_uncond = (
        uncond_queue.sample(n_uncond, device=device)
        if n_uncond > 0 and uncond_queue is not None
        else None
    )

    noise = torch.randn(
        labels_gen.shape[0],
        config["in_channels"],
        config["img_size"],
        config["img_size"],
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
        x_gen = model_for_train(noise, labels_gen, alpha_gen)
        loss, info = compute_drifting_loss(
            x_gen=x_gen,
            labels_gen=labels_gen,
            class_labels=class_labels,
            class_alphas=class_alphas,
            x_pos=x_pos,
            labels_pos=labels_pos,
            x_uncond=x_uncond,
            feature_encoder=feature_encoder,
            vae_decoder=vae_decoder,
            temperatures=config["temperatures"],
            n_neg=n_neg,
            n_uncond=n_uncond,
            use_pixel_space=not config["use_feature_encoder"],
            use_full_feature_set=bool(config.get("use_full_feature_set", False)),
        )

    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()

    info["grad_norm"] = float(grad_norm.item())
    # Alpha diagnostics for sampled class-level CFG strengths.
    info["alpha_mean"] = float(class_alphas.mean().item())
    info["alpha_std"] = float(class_alphas.std(unbiased=False).item())
    info["alpha_min"] = float(class_alphas.min().item())
    info["alpha_max"] = float(class_alphas.max().item())

    ema.update(model)
    scheduler.step()
    return info


def fill_queues(
    class_queue: SampleQueue,
    uncond_queue: Optional[GlobalSampleQueue],
    dataloader: DataLoader,
    min_class_samples: int,
    min_uncond_samples: int = 0,
):
    """Warm up sample queues with real data."""
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            x, labels = batch[0], batch[1]
        else:
            x = batch
            labels = torch.zeros(x.shape[0], dtype=torch.long)
        class_queue.add(x, labels)
        if uncond_queue is not None:
            uncond_queue.add(x)

        class_ready = class_queue.is_ready(min_class_samples)
        uncond_ready = uncond_queue is None or uncond_queue.is_ready(min_uncond_samples)
        if class_ready and uncond_ready:
            break


def train(args):
    distributed, rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    device = (
        torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    seed = args.seed + rank
    set_seed(seed)

    config = get_default_config(args.dataset)
    config["dataset"] = args.dataset.lower()
    if args.feature_encoder_arch is not None:
        config["feature_encoder_arch"] = args.feature_encoder_arch
    if args.feature_encoder_path is not None:
        config["feature_encoder_path"] = args.feature_encoder_path
    if args.vae_path is not None:
        config["vae_path"] = args.vae_path

    # Ensure queues can satisfy per-step sampling requirements.
    if config["queue_size"] < config["batch_n_pos"]:
        config["queue_size"] = config["batch_n_pos"]
    if config["batch_n_uncond"] > 0 and config["uncond_queue_size"] < config["batch_n_uncond"]:
        config["uncond_queue_size"] = config["batch_n_uncond"]
    if is_main:
        print(f"Using device: {device}")
        print(f"Dataset: {config['dataset']}")
        print(f"Distributed: {distributed} (world_size={world_size})")

    output_dir = Path(args.output_dir) / config["dataset"]
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    need_eval_dataset = is_main and args.wandb and args.wandb_fid_interval > 0
    train_dataset, test_dataset = get_dataset(
        config["dataset"],
        root=args.data_root,
        img_size=config["img_size"],
        include_eval=need_eval_dataset,
    )

    dataset_num_classes = len(getattr(train_dataset, "classes", []))
    if dataset_num_classes > 0 and dataset_num_classes != config["num_classes"]:
        if is_main:
            print(
                f"Detected {dataset_num_classes} classes from dataset, overriding "
                f"config num_classes={config['num_classes']} for this run."
            )
        config["num_classes"] = dataset_num_classes

    sampler = None
    if distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["loader_batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    model_fn = DriftDiT_models[config["model"]]
    model = model_fn(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        label_dropout=config["label_dropout"],
    ).to(device)
    model_for_train: nn.Module = model
    if distributed:
        model_for_train = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main:
        print(f"Model: {config['model']}, Parameters: {count_parameters(model):,}")

    feature_encoder = None
    if config["use_feature_encoder"]:
        if is_main:
            print("Creating feature encoder...")
        feature_encoder = create_feature_encoder(
            dataset=config["dataset"],
            feature_dim=512,
            multi_scale=True,
            use_pretrained=True,
            pretrained_arch=config["feature_encoder_arch"],
            pretrained_path=config.get("feature_encoder_path"),
        ).to(device)
        feature_encoder.eval()
        for p in feature_encoder.parameters():
            p.requires_grad = False

    vae_decoder: Optional[nn.Module] = None
    if config["use_feature_encoder"] and feature_encoder is not None:
        encoder_channels = int(getattr(feature_encoder, "expected_in_channels", config["in_channels"]))
        if encoder_channels != config["in_channels"]:
            if config["in_channels"] != 4 or encoder_channels != 3:
                raise ValueError(
                    "Unsupported channel setup for feature extraction: "
                    f"model in_channels={config['in_channels']}, "
                    f"feature encoder in_channels={encoder_channels}."
                )
            vae_path = config.get("vae_path")
            if not vae_path:
                raise ValueError(
                    "Model outputs 4-channel latents but feature encoder expects RGB. "
                    "Please set `vae_path` in config or pass `--vae_path`."
                )
            if is_main:
                print(f"Loading VAE decoder from: {vae_path}")
            from diffusers import AutoencoderKL
            vae_decoder = AutoencoderKL.from_pretrained(
                vae_path,
                local_files_only=True,
                use_safetensors=True,
            ).to(device)
            vae_decoder.eval()
            for p in vae_decoder.parameters():
                p.requires_grad = False

    ema = EMA(model, decay=config["ema_decay"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.95),
        weight_decay=config["weight_decay"],
    )
    scheduler = WarmupLRScheduler(
        optimizer,
        warmup_steps=config["warmup_steps"],
        base_lr=config["lr"],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and torch.cuda.is_available())

    class_queue = SampleQueue(
        num_classes=config["num_classes"],
        queue_size=config["queue_size"],
        sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
    )
    uncond_queue: Optional[GlobalSampleQueue] = None
    if config["batch_n_uncond"] > 0:
        uncond_queue = GlobalSampleQueue(
            queue_size=config["uncond_queue_size"],
            sample_shape=(config["in_channels"], config["img_size"], config["img_size"]),
        )

    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, ema, optimizer, scheduler)
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["step"])
        if is_main:
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    wandb_run = None
    if args.wandb and is_main:
        key_file = Path(args.wandb_key_file) if args.wandb_key_file else None
        if key_file and key_file.exists() and "WANDB_API_KEY" not in os.environ:
            os.environ["WANDB_API_KEY"] = key_file.read_text().strip()

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config,
        )
    fid_real_images: Optional[torch.Tensor] = None
    if is_main and wandb_run is not None and args.wandb_fid_interval > 0:
        fid_workers = max(0, args.fid_num_workers)
        fid_loader = DataLoader(
            test_dataset,
            batch_size=args.fid_batch_size,
            shuffle=True,
            num_workers=fid_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=fid_workers > 0,
        )
        real_batches = []
        real_count = 0
        for batch in fid_loader:
            if isinstance(batch, (list, tuple)):
                x_real = batch[0]
            else:
                x_real = batch
            remaining = args.fid_num_samples - real_count
            if remaining <= 0:
                break
            if x_real.shape[0] > remaining:
                x_real = x_real[:remaining]
            real_batches.append(x_real.cpu())
            real_count += int(x_real.shape[0])
            if real_count >= args.fid_num_samples:
                break
        if real_batches:
            fid_real_images = torch.cat(real_batches, dim=0)
        else:
            print("No real images collected for FID; disabling FID logging.")

    if is_main:
        print(f"Starting training for {config['epochs']} epochs...")

    for epoch in range(start_epoch, config["epochs"]):
        if sampler is not None:
            sampler.set_epoch(epoch)

        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_grad = 0.0
        n_batches = 0

        fill_queues(
            class_queue,
            uncond_queue,
            train_loader,
            min_class_samples=min(config["batch_n_pos"], config["queue_size"]),
            min_uncond_samples=(
                max(1, min(config["batch_n_uncond"], config["uncond_queue_size"]))
                if uncond_queue is not None
                else 0
            ),
        )

        for batch in train_loader:
            if isinstance(batch, (list, tuple)):
                x_real, labels_real = batch[0], batch[1]
            else:
                x_real = batch
                labels_real = torch.zeros(x_real.shape[0], dtype=torch.long)
            x_real = x_real.to(device, non_blocking=True)
            labels_real = labels_real.to(device, non_blocking=True)

            class_queue.add(x_real, labels_real)
            if uncond_queue is not None:
                uncond_queue.add(x_real)

            info = train_step(
                model_for_train=model_for_train,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                class_queue=class_queue,
                uncond_queue=uncond_queue,
                config=config,
                device=device,
                feature_encoder=feature_encoder,
                vae_decoder=vae_decoder,
            )
            if info is None:
                continue
            global_step += 1

            info = reduce_info(info, device=device, distributed=distributed, world_size=world_size)
            epoch_loss += info["loss"]
            epoch_grad += info["grad_norm"]
            n_batches += 1

            if is_main and global_step % args.log_interval == 0:
                lr = scheduler.get_lr()
                print(
                    f"Epoch {epoch + 1}/{config['epochs']} | Step {global_step} | "
                    f"Loss: {info['loss']:.4f} | "
                    f"Grad: {info['grad_norm']:.4f} | LR: {lr:.6f}"
                )
                if wandb_run is not None:
                    train_payload = {
                        "train/loss": info["loss"],
                        "train/grad_norm": info["grad_norm"],
                        "train/lr": lr,
                    }
                    if "Z_pos" in info:
                        train_payload["train/Z_pos"] = info["Z_pos"]
                    if "Z_neg" in info:
                        train_payload["train/Z_neg"] = info["Z_neg"]

                    for key in ("alpha_mean", "alpha_std", "alpha_min", "alpha_max"):
                        if key in info:
                            train_payload[f"train/{key}"] = info[key]

                    wandb_run.log(train_payload, step=global_step)
            if (
                is_main
                and wandb_run is not None
                and fid_real_images is not None
                and args.wandb_fid_interval > 0
                and global_step % args.wandb_fid_interval == 0
            ):
                fid_score = compute_fid_score(
                    model=ema.shadow,
                    real_images=fid_real_images,
                    in_channels=config["in_channels"],
                    img_size=config["img_size"],
                    num_classes=max(1, config["num_classes"]),
                    device=device,
                    num_samples=args.fid_num_samples,
                    batch_size=args.fid_batch_size,
                    alpha=args.preview_alpha,
                )
                if math.isfinite(fid_score):
                    print(f"Step {global_step} | FID: {fid_score:.4f}")
                    wandb_run.log({"eval/fid": fid_score}, step=global_step)

                is_mean, is_std = compute_is_score(
                    model=ema.shadow,
                    in_channels=config["in_channels"],
                    img_size=config["img_size"],
                    num_classes=max(1, config["num_classes"]),
                    device=device,
                    num_samples=args.fid_num_samples,
                    batch_size=args.fid_batch_size,
                    alpha=args.preview_alpha,
                )
                if math.isfinite(is_mean) and math.isfinite(is_std):
                    print(f"Step {global_step} | IS: {is_mean:.4f} ± {is_std:.4f}")
                    wandb_run.log({"eval/is": is_mean, "eval/is_std": is_std}, step=global_step)

            if is_main and args.sample_every_steps > 0 and global_step % args.sample_every_steps == 0:
                sample_path = output_dir / f"samples_step{global_step}.png"
                preview_images = generate_class_grid(
                    model=ema.shadow,
                    in_channels=config["in_channels"],
                    img_size=config["img_size"],
                    num_classes=min(config["num_classes"], config["preview_classes"]),
                    device=device,
                    samples_per_class=args.preview_samples_per_class,
                    alpha=args.preview_alpha,
                )
                save_image_grid(preview_images, str(sample_path), nrow=args.preview_samples_per_class)
                if wandb_run is not None:
                    wandb_run.log({"samples/step_grid": wandb.Image(str(sample_path))}, step=global_step)
                print(f"Saved samples to {sample_path}")

        if n_batches == 0:
            avg_loss = 0.0
            avg_grad = 0.0
        else:
            avg_loss = epoch_loss / n_batches
            avg_grad = epoch_grad / n_batches

        if is_main:
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch + 1} done in {epoch_time:.1f}s | "
                f"Avg Loss: {avg_loss:.4f} | Avg Grad: {avg_grad:.4f}"
            )

        if is_main and (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / "checkpoint.pt"
            save_checkpoint(
                str(ckpt_path),
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()
    cleanup_distributed(distributed)


def main():
    parser = argparse.ArgumentParser(description="Train Drifting Models")

    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(DATASET_CONFIG_FILES.keys()),
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--sample_every_steps", type=int, default=1000)
    parser.add_argument("--preview_samples_per_class", type=int, default=4)
    parser.add_argument("--preview_alpha", type=float, default=1.5)

    parser.add_argument("--amp", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="drifting-model",
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_key_file",
        type=str,
        default="../../.netcr_wandb",
    )
    parser.add_argument(
        "--wandb_fid_interval",
        type=int,
        default=0,
        help="Log FID to W&B every N steps (0 disables FID logging).",
    )
    parser.add_argument(
        "--fid_num_samples",
        type=int,
        default=1000,
        help="Number of real/fake samples used per FID evaluation.",
    )
    parser.add_argument(
        "--fid_batch_size",
        type=int,
        default=128,
        help="Batch size used for generated and real samples in FID evaluation.",
    )
    parser.add_argument(
        "--fid_num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers used for FID real-image loader.",
    )
    parser.add_argument(
        "--feature_encoder_arch",
        type=str,
        default=None,
        help="Optional override for feature encoder architecture.",
    )
    parser.add_argument(
        "--feature_encoder_path",
        type=str,
        default=None,
        help="Optional local path/name for feature encoder weights.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Optional local path for SD-VAE decoder when model outputs latents.",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
