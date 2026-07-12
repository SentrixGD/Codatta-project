"""
File: evaluation.py

Description:
    Evaluation functions: validation loops, metric computation, confusion matrix,
    and per-task metric helpers.

Purpose:
    Encapsulate all evaluation logic so the training loop calls evaluate()
    and compute_metrics() without knowing the details.

Usage:
    from src.training.evaluation import evaluate, compute_metrics, make_confusion_matrix
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from src.config import CONFUSION_CLASS_NAMES, CONFUSION_LABELS, RESULTS_DIR
from src.training.dataset import move_to_device
from src.training.losses import LossCombiner

# ──────────────────────────────────────────────
# Per-task metric helpers
# ──────────────────────────────────────────────


def val_single_class(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    name: str,
    topk: int,
) -> tuple[int, int, int]:
    """Compute top-1 and top-k accuracy for a single-label classification task.

    Returns:
        (correct, total, correctk)
    """
    logits = outputs[f"{name}_logits"]
    target = batch[name]

    pred = logits.argmax(dim=1)
    correct = (pred == target).sum().item()
    total = target.numel()

    if topk > 1:
        topk_pred = logits.topk(topk, dim=1).indices
        correctk = topk_pred.eq(target.unsqueeze(1)).any(dim=1).sum().item()
    else:
        correctk = 0

    return correct, total, correctk


def val_regression(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    task_name: str,
    target_name: str,
) -> tuple[float, int]:
    """Compute MAE sum and count for a regression task.

    Returns:
        (mae_sum, count)
    """
    prediction = outputs[f"{task_name}_logits"].squeeze(1)
    target = batch[target_name]
    mae_sum = torch.abs(prediction - target).sum().item()
    count = target.numel()
    return mae_sum, count


def val_binary_prob(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    logit_idx: int,
    target_name: str,
) -> tuple[float, int]:
    """Compute MAE sum and count for a binary probability task.

    Returns:
        (mae_sum, count)
    """
    prediction = torch.sigmoid(outputs["binary_logits"][:, logit_idx])
    target = batch[target_name]
    mae_sum = torch.abs(prediction - target).sum().item()
    count = target.numel()
    return mae_sum, count


# ──────────────────────────────────────────────
# Log row builder
# ──────────────────────────────────────────────


def build_log_row(metrics: dict, headers: list[str]) -> list:
    """Build an ordered list of metric values matching the header order."""
    row = []
    for h in headers:
        if h not in metrics:
            raise KeyError(f"Missing metric: {h}")
        row.append(metrics[h])
    return row


# ──────────────────────────────────────────────
# Main evaluate function
# ──────────────────────────────────────────────


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    total_loss: LossCombiner,
    desc: str = "Validating",
    active_heads: set[str] | None = None,
) -> dict:
    """Run a full evaluation pass and return raw state for compute_metrics."""
    model.eval()

    val_total_loss = 0.0
    val_batches = 0

    cls_correct: dict[str, int] = {}
    cls_total: dict[str, int] = {}
    cls_topk_correct: dict[str, int] = {}

    all_preds: list[int] = []
    all_targets: list[int] = []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = move_to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(batch["image"], active_heads=active_heads)

                loss, _, _, _ = total_loss(outputs, batch)
            val_total_loss += loss.item()
            val_batches += 1

            correct, total, correctk = val_single_class(outputs, batch, "food_type", 5)
            cls_correct["food_type"] = cls_correct.get("food_type", 0) + correct
            cls_total["food_type"] = cls_total.get("food_type", 0) + total
            cls_topk_correct["food_type"] = cls_topk_correct.get("food_type", 0) + correctk

            preds = outputs["food_type_logits"].argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(batch["food_type"].cpu().tolist())

    return {
        "val_total_loss": val_total_loss,
        "val_batches": val_batches,
        "cls_correct": cls_correct,
        "cls_total": cls_total,
        "cls_topk_correct": cls_topk_correct,
        "all_preds": all_preds,
        "all_targets": all_targets,
    }


# ──────────────────────────────────────────────
# Metric computation
# ──────────────────────────────────────────────


def compute_metrics(state: dict, model_name: str, epoch: int | None = None) -> dict:
    """Compute scalar metrics from raw evaluation state."""
    val_total_loss = state["val_total_loss"] / max(state["val_batches"], 1)
    food_type_acc = state["cls_correct"].get("food_type", 0) / max(state["cls_total"].get("food_type", 1), 1)
    food_type_top5_acc = state["cls_topk_correct"].get("food_type", 0) / max(
        state["cls_total"].get("food_type", 1), 1
    )

    all_preds = state.get("all_preds", [])
    all_targets = state.get("all_targets", [])

    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    if all_preds and all_targets:
        p, r, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average="macro", zero_division=0
        )
        macro_p, macro_r, macro_f1 = float(p), float(r), float(f1)

    return {
        "epoch": epoch,
        "model_name": model_name,
        "total_loss": val_total_loss,
        "food_type_acc": food_type_acc,
        "food_type_top5_acc": food_type_top5_acc,
        "precision": macro_p,
        "recall": macro_r,
        "f1": macro_f1,
    }


# ──────────────────────────────────────────────
# Validation report (confusion matrix + per-class CSV)
# ──────────────────────────────────────────────


def save_validation_report(
    state: dict,
    save_dir: str,
    epoch: int,
    class_names: list[str] | None = None,
    labels: list[int] | None = None,
) -> None:
    """Save confusion matrix CSV and per-class precision/recall/F1 CSV."""
    all_preds = state.get("all_preds", [])
    all_targets = state.get("all_targets", [])
    if not all_preds:
        return

    if class_names is None:
        class_names = CONFUSION_CLASS_NAMES
    if labels is None:
        labels = CONFUSION_LABELS

    os.makedirs(save_dir, exist_ok=True)

    cm = confusion_matrix(all_targets, all_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = os.path.join(save_dir, f"confusion_matrix_epoch{epoch}.csv")
    cm_df.to_csv(cm_path)

    p, r, f1, support = precision_recall_fscore_support(
        all_targets, all_preds, labels=labels, zero_division=0
    )
    report_df = pd.DataFrame({
        "class": class_names,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    })
    report_path = os.path.join(save_dir, f"class_metrics_epoch{epoch}.csv")
    report_df.to_csv(report_path, index=False)

    print(f"Confusion matrix saved: {cm_path}")
    print(f"Per-class metrics saved: {report_path}")


# ──────────────────────────────────────────────
# Confusion matrix (test time)
# ──────────────────────────────────────────────


def make_confusion_matrix(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Generate, save, and return the confusion matrix for food_type."""
    model.eval()

    y_true: list = []
    y_pred: list = []

    pbar = tqdm(dataloader, desc="Testing", leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = move_to_device(batch, device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(batch["image"])["food_type_logits"]
            preds = outputs.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch["food_type"].cpu().numpy())

    cm = confusion_matrix(y_true, y_pred, labels=CONFUSION_LABELS)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # CSV
    cm_df = pd.DataFrame(cm, index=CONFUSION_CLASS_NAMES, columns=CONFUSION_CLASS_NAMES)
    cm_df.to_csv(os.path.join(RESULTS_DIR, "food_type_confusion_matrix.csv"))

    # Per-class metrics CSV
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=CONFUSION_LABELS, zero_division=0
    )
    report_df = pd.DataFrame({
        "class": CONFUSION_CLASS_NAMES,
        "precision": p,
        "recall": r,
        "f1": f1,
        "support": support,
    })
    report_df.to_csv(os.path.join(RESULTS_DIR, "food_type_class_metrics.csv"), index=False)

    # Classification report
    print(classification_report(
        y_true, y_pred, labels=CONFUSION_LABELS,
        target_names=CONFUSION_CLASS_NAMES, zero_division=0,
    ))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CONFUSION_CLASS_NAMES)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30)
    ax.set_title("Food Type Test Confusion Matrix", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "food_type_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cm
