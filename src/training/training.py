"""
File: training.py

Description:
    Training loop, stress test, and checkpoint save/load.

Purpose:
    Encapsulate the training machinery so the entry point only orchestrates
    data loading, model creation, and calls to these functions.

Usage:
    from src.training.training import stress_test, train_epoch, save_checkpoint, load_checkpoint
"""

import math
from collections import deque

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    EMA_DECAY,
    GRAD_CLIP_MAX_NORM,
)
from src.training.dataset import move_to_device
from src.training.evaluation import compute_metrics, evaluate, save_validation_report
from src.training.losses import LossCombiner

# ──────────────────────────────────────────────
# Checkpoint
# ──────────────────────────────────────────────


def save_checkpoint(
    path: str,
    epoch: int,
    training_step: int,
    model: torch.nn.Module,
    total_loss: LossCombiner,
    model_optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
) -> None:
    """Save full training state to disk."""
    checkpoint = {
        "epoch": epoch,
        "training_step": training_step,
        "model_state_dict": model.state_dict(),
        "model_optimizer_state_dict": model_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss_combiner_state_dict": total_loss.state_dict(),
        "loss_phase": total_loss.phase,
        "loss_means": total_loss.loss_means,
        "loss_sum": total_loss.loss_sum,
        "loss_counts": total_loss.loss_counts,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    total_loss: LossCombiner,
    model_optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int]:
    """Restore training state from checkpoint.

    Returns:
        (start_epoch, training_step)
    """
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model_optimizer.load_state_dict(checkpoint["model_optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    total_loss.load_state_dict(checkpoint["loss_combiner_state_dict"])

    total_loss.phase = checkpoint["loss_phase"]
    total_loss.loss_means = checkpoint["loss_means"]
    total_loss.loss_sum = checkpoint["loss_sum"]
    total_loss.loss_counts = checkpoint["loss_counts"]

    start_epoch = checkpoint["epoch"] + 1
    training_step = checkpoint["training_step"]

    print(f"Resuming from epoch={start_epoch}, training_step={training_step}")
    return start_epoch, training_step


# ──────────────────────────────────────────────
# Stress test
# ──────────────────────────────────────────────


def stress_test(
    model: torch.nn.Module,
    total_loss: LossCombiner,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 1,
    height: int = 448,
    width: int = 1344,
    use_checkpoint: bool = False,
    active_heads: set[str] | None = None,
) -> None:
    """Run a single forward/backward/optimizer pass to verify GPU memory fits."""
    model.train()

    heads_str = "all" if active_heads is None else ",".join(sorted(active_heads))
    print(f"Testing batch={batch_size}, resolution={height}x{width}, checkpoint={use_checkpoint}, heads={heads_str}")

    total_gpu = torch.cuda.get_device_properties(device).total_memory / 1024**3
    print(f"Total GPU memory: {total_gpu:.2f} GB")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    baseline_mem = torch.cuda.memory_allocated(device) / 1024**3

    dummy_image = torch.randn(batch_size, 3, height, width, device=device)
    outputs = model(dummy_image, use_checkpoint=use_checkpoint, active_heads=active_heads)
    forward_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    # build only the targets needed for active heads
    dummy_targets: dict[str, torch.Tensor] = {}
    if active_heads is None or "food_type" in active_heads:
        dummy_targets["food_type"] = torch.randint(0, 5, (batch_size,), device=device)
    if active_heads is None or "dish_name" in active_heads:
        dummy_targets["dish_name"] = torch.randint(0, 602, (batch_size,), device=device)
    if active_heads is None or "ingredients" in active_heads:
        dummy_targets["ingredients"] = torch.randint(0, 2, (batch_size, 589), device=device).float()
    if active_heads is None or "portions" in active_heads:
        dummy_targets["portion_presence"] = torch.randint(0, 2, (batch_size, 437), device=device).float()
        dummy_targets["portion_weight"] = torch.randn(batch_size, 437, device=device)
    if active_heads is None or "cooking_method" in active_heads:
        dummy_targets["cooking_method"] = torch.randint(0, 2, (batch_size, 15), device=device).float()
    if active_heads is None or "calories" in active_heads:
        dummy_targets["calories_kcal"] = torch.randn(batch_size, device=device)
    if active_heads is None or "fats" in active_heads:
        dummy_targets["fat_g"] = torch.randn(batch_size, device=device)
    if active_heads is None or "carbohydrates" in active_heads:
        dummy_targets["carbohydrate_g"] = torch.randn(batch_size, device=device)
    if active_heads is None or "proteins" in active_heads:
        dummy_targets["protein_g"] = torch.randn(batch_size, device=device)
    if active_heads is None or "binary" in active_heads:
        dummy_targets["food_prob"] = torch.rand(batch_size, device=device)
        dummy_targets["camera_or_phone_prob"] = torch.rand(batch_size, device=device)

    optimizer.zero_grad(set_to_none=True)

    loss, _, _, _ = total_loss(outputs, dummy_targets)
    loss.backward()
    backward_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    optimizer.step()
    step_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print("Loss:", loss.item())
    print(f"Total GPU:       {total_gpu:.2f} GB")
    print(f"Baseline VRAM:   {baseline_mem:.2f} GB")
    print(f"After forward:   {forward_mem:.2f} GB ({forward_mem / total_gpu * 100:.0f}%)")
    print(f"After backward:  {backward_mem:.2f} GB ({backward_mem / total_gpu * 100:.0f}%)")
    print(f"After optimizer: {step_mem:.2f} GB ({step_mem / total_gpu * 100:.0f}%)")
    print(f"Peak VRAM:       {step_mem:.2f} GB ({step_mem / total_gpu * 100:.0f}%)")

    optimizer.zero_grad(set_to_none=True)
    del dummy_image, outputs, dummy_targets
    torch.cuda.empty_cache()

    print("Stress test passed.")


# ──────────────────────────────────────────────
# Single training epoch
# ──────────────────────────────────────────────


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_dataset: torch.utils.data.Dataset,
    loss_dictionary: dict[str, torch.nn.Module],
    total_loss: LossCombiner,
    model_optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    effective_batch_size: int,
    accumulation_steps: int,
    model_name: str,
    val_loader: torch.utils.data.DataLoader,
    checkpoint_path: str,
    log_path: str,
    val_log_path: str,
    writer,  # SummaryWriter
    HEADERS: list[str],
    build_log_row_fn,  # function reference
    gradient_checkpointing: bool = False,
    val_save_dir: str | None = None,
    active_heads: set[str] | None = None,
) -> int:
    """Run one full training epoch with AMP, gradient accumulation, and validation.

    Args:
        gradient_checkpointing: If True, use torch checkpointing to reduce activation memory.

    Returns:
        Updated training_step count.
    """
    model.train()
    model_optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(
        total=math.ceil(len(train_dataset) / effective_batch_size),
        desc=f"Epoch {epoch + 1}/{epochs}",
    )
    accum_counter = 0
    running_loss = 0.0
    ema_score = 0.0
    window_losses: deque[float] = deque(maxlen=100)
    training_step = 0

    for batch in train_loader:
        batch = move_to_device(batch, device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(batch["image"], use_checkpoint=gradient_checkpointing, active_heads={"food_type"})

            loss = loss_dictionary["food_type"](outputs["food_type_logits"], batch["food_type"])

        running_loss += loss.detach()

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        accum_counter += 1

        if accum_counter == accumulation_steps:
            scaler.unscale_(model_optimizer)

            if not torch.isfinite(running_loss):
                if training_step % 50 == 0:
                    print(f"Invalid loss at step {training_step}, skipping")
                accum_counter = 0
                running_loss = 0.0
                model_optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()),
                max_norm=GRAD_CLIP_MAX_NORM,
            )

            scaler.step(model_optimizer)
            scaler.update()
            scheduler.step()
            model_optimizer.zero_grad(set_to_none=True)

            effective_loss = running_loss.item() / accumulation_steps
            ema_score = EMA_DECAY * ema_score + (1 - EMA_DECAY) * effective_loss
            window_losses.append(effective_loss)
            window_avg = sum(window_losses) / len(window_losses)

            pbar.update(1)
            pbar.set_postfix({
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            })

            with open(log_path, "a") as f:
                row = [epoch, pbar.n, f"{effective_loss:.4f}", f"{ema_score:.4f}", f"{window_avg:.4f}"]
                f.write(",".join(map(str, row)) + "\n")

            accum_counter = 0
            running_loss = 0.0
            training_step += 1

    # handle remainder
    if accum_counter > 0:
        scaler.unscale_(model_optimizer)
        model_optimizer.step()
        scaler.update()
        model_optimizer.zero_grad(set_to_none=True)

        effective_loss = running_loss.item() / accum_counter
        pbar.update(1)
        pbar.set_postfix({
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    pbar.close()

    # ── Validation ──
    val_state = evaluate(model, val_loader, device, total_loss, desc=f"Validating epoch {epoch + 1}", active_heads=active_heads)
    metrics = compute_metrics(val_state, model_name=model_name, epoch=epoch)
    print(metrics, HEADERS)

    save_checkpoint(
        checkpoint_path, epoch, training_step, model, total_loss,
        model_optimizer, scheduler, scaler,
    )

    row = build_log_row_fn(metrics, HEADERS)
    with open(val_log_path, "a") as f:
        f.write(",".join(map(str, row)) + "\n")

    if val_save_dir is not None:
        save_validation_report(val_state, val_save_dir, epoch)

    for key, val in metrics.items():
        if isinstance(val, (int, float)):
            writer.add_scalar(key, val, epoch)

    return training_step
