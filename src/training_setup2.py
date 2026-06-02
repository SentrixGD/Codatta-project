"""
File: training_setup2.py

Description:
    Main file for training the model.

Purpose:
    To train the model, this file contains all the necessary components and the model class itself.

Inputs:
    - Resized images are stored in the /data/resized_images directory.
    - Labels are stored in the /data/{train, val, test}_sorted_labels.parquet files.

Outputs:
    - Trained model checkpoints.
    - Training logs.
    - Confusion matrices.

Dependencies:
    - pandas
    - PIL
    - torch
    - torchinfo
    - torchviz
    - torchvision
    - tqdm
    - wandb
    - matplotlib
    - sklearn
    - numpy


Usage:
    python -m src.data_normalization
"""  # noqa: E501

import ast
import json
import math
import os
from collections import defaultdict
from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import transforms
from torchviz import make_dot
from tqdm import tqdm

import wandb
from src.model import SwinModel, init_weights

LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

wandb.init(project="your_project", mode="online")


def normalize(s):

    return s.replace(" ", "_").lower()


def load_image(image_id: str) -> torch.Tensor:
    path = os.path.join(DATA_DIR, "resized_images", image_id)
    image = Image.open(path).convert("RGB")
    image = train_transform(image)
    return image


def build_vocab(values):
    """
    Build a vocabulary from a list of values.

    Args:
        values (list): List of values to build the vocabulary from.

    Returns:
        tuple: A tuple containing the vocabulary and its inverse.
    """
    unique = sorted(set(values))
    stoi = {v: i for i, v in enumerate(unique)}
    itos = {i: v for v, i in stoi.items()}
    return stoi, itos


def build_multilabel_vocab(column):
    """
    Build a vocabulary from a list of labels.

    Args:
        column (list): List of labels to build the vocabulary from.

    Returns:
        tuple: A tuple containing the vocabulary and its inverse.
    """
    unique = set()
    for labels in column:
        unique.update(labels)  # flatten
    unique = sorted(unique)

    stoi = {v: i for i, v in enumerate(unique)}
    itos = {i: v for v, i in stoi.items()}
    return stoi, itos


class FoodDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        food_type_vocab: Dict[str, int],
        dish_name_vocab: Dict[str, int],
        cooking_method_vocab: Dict[str, int],
        ingredients_vocab: Dict[str, int],
        portion_ingredients_vocab: Dict[str, int],
        regression_normalization: Dict[str, Dict[str, float]],
    ):
        """
        Dataset for food classification.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset.
            food_type_vocab (Dict[str, int]): Vocabulary for food type.
            dish_name_vocab (Dict[str, int]): Vocabulary for dish name.
            cooking_method_vocab (Dict[str, int]): Vocabulary for cooking method.
            ingredients_vocab (Dict[str, int]): Vocabulary for ingredients.
            portion_ingredients_vocab (Dict[str, int]): Vocabulary for portion ingredients.
            regression_normalization (Dict[str, Dict[str, float]]): Normalization parameters for regression.
        """
        self.df = df.reset_index(drop=True)

        self.image_loader = load_image
        self.food_type_vocab = food_type_vocab
        self.dish_name_vocab = dish_name_vocab
        self.cooking_method_vocab = cooking_method_vocab
        self.ingredients_vocab = ingredients_vocab
        self.portion_ingredients_vocab = portion_ingredients_vocab
        self.regression_normalization = regression_normalization

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: A tuple containing the image and the labels.
        """
        row = self.df.iloc[idx]

        # --- image ---
        image = self.image_loader(row["image_path"])

        # --- labels (placeholders for your encoding logic) ---
        row = self.df.iloc[idx]

        targets = {}

        targets["food_type"] = encode_single(row["food_type"], self.food_type_vocab)
        targets["dish_name"] = encode_single(row["dish_name"], self.dish_name_vocab)

        targets["cooking_method"] = encode_multilabel(row["cooking_method"], self.cooking_method_vocab)

        targets["ingredients"] = encode_multilabel(row["ingredients"], self.ingredients_vocab)

        presence, weight = encode_portion(row["portion_size"], self.portion_ingredients_vocab)

        targets["portion_presence"] = presence
        weight = weight.to(torch.float32)

        mask = weight > 0

        targets["portion_weight"] = torch.zeros_like(weight)

        targets["portion_weight"][mask] = torch.tensor(
            (np.log1p(weight[mask]) - self.regression_normalization["portion_size"]["log_mean"])
            / (self.regression_normalization["portion_size"]["log_std"]),
            dtype=torch.float32,
        )

        targets["fat_g"] = torch.tensor(
            (np.log1p(row["fat_g"]) - self.regression_normalization["fat_g"]["log_mean"])
            / (self.regression_normalization["fat_g"]["log_std"]),
            dtype=torch.float32,
        )
        targets["protein_g"] = torch.tensor(
            (np.log1p(row["protein_g"]) - self.regression_normalization["protein_g"]["log_mean"])
            / (self.regression_normalization["protein_g"]["log_std"]),
            dtype=torch.float32,
        )
        targets["carbohydrate_g"] = torch.tensor(
            (np.log1p(row["carbohydrate_g"]) - self.regression_normalization["carbohydrate_g"]["log_mean"])
            / (self.regression_normalization["carbohydrate_g"]["log_std"]),
            dtype=torch.float32,
        )
        targets["calories_kcal"] = torch.tensor(
            (np.log1p(row["calories_kcal"]) - self.regression_normalization["calories_kcal"]["log_mean"])
            / (self.regression_normalization["calories_kcal"]["log_std"]),
            dtype=torch.float32,
        )

        targets["camera_or_phone_prob"] = torch.tensor(row["camera_or_phone_prob"], dtype=torch.float32)
        targets["food_prob"] = torch.tensor(row["food_prob"], dtype=torch.float32)

        return {
            "image": image,
            "food_type": targets["food_type"],
            "dish_name": targets["dish_name"],
            "cooking_method": targets["cooking_method"],
            "ingredients": targets["ingredients"],
            "portion_presence": targets["portion_presence"],
            "portion_weight": targets["portion_weight"],
            "fat_g": targets["fat_g"],
            "protein_g": targets["protein_g"],
            "carbohydrate_g": targets["carbohydrate_g"],
            "calories_kcal": targets["calories_kcal"],
            "camera_or_phone_prob": targets["camera_or_phone_prob"],
            "food_prob": targets["food_prob"],
        }


class LossCombiner(nn.Module):
    def __init__(self, losses: Dict[str, nn.Module]):
        """
        Loss combiner module.

        Args:
            losses (Dict[str, nn.Module]): Dictionary of loss modules.
        """
        super().__init__()

        self.losses = losses
        self.loss_names = list(losses.keys())
        self.loss_means = {name: 1.0 for name in self.loss_names}

        # Kendall parameters
        self.raw_weights = nn.Parameter(torch.ones(len(losses)))

        # phase control
        self.phase = "calibration"

        # running stats (ONLY for init)
        self.loss_sum = {name: 0.0 for name in self.loss_names}
        self.loss_counts = {name: 0 for name in self.loss_names}

        self.tasks = {"food_type": {"target": "food_type", "type": "cls"}}

    def reset_means(self):
        self.loss_sum = {name: 0.0 for name in self.loss_names}
        self.loss_counts = {name: 0 for name in self.loss_names}

    def set_mean_phase(self):
        """Call after calibration phase."""
        with torch.no_grad():
            for name in self.loss_names:
                self.loss_means[name] = self.loss_sum[name] / max(self.loss_counts[name], 1)

        self.phase = "mean"

    def set_kendall_phase(self):
        """Call after mean phase."""
        self.raw_weights.data.fill_(0.5413248546)
        self.phase = "kendall"

    def forward(self, outputs, targets):
        """
        Forward pass.

        Args:
            outputs (Dict[str, torch.Tensor]): Dictionary of output tensors.
            targets (Dict[str, torch.Tensor]): Dictionary of target tensors.

        Returns:
            torch.Tensor: Total loss.
        """
        total_loss = 0.0
        loss_dict = {}
        raw_loss_dict = {}

        for i, name in enumerate(self.loss_names):
            task = self.tasks[name]

            # targets
            if isinstance(task["target"], list):
                target = torch.stack([targets[t] for t in task["target"]], dim=1)
            else:
                target = targets[task["target"]]

            output = outputs[f"{name}_logits"]

            if task["type"] == "reg":
                target = target.float()
                if target.ndim == 1:
                    target = target.unsqueeze(1)

            if task["target"] == "portion_weight":
                output = output * targets["portion_presence"]

            base_loss = self.losses[name](output, target)

            if self.phase == "calibration":
                self.loss_sum[name] += base_loss.detach().item()
                self.loss_counts[name] += 1

                weighted_loss = base_loss  # no Kendall yet
            elif self.phase == "mean":
                weighted_loss = base_loss / self.loss_means[name]
            else:
                weight = F.softplus(self.raw_weights[i]) + 1e-8
                normalized_loss = base_loss / self.loss_means[name]
                if task["type"] == "reg":
                    weighted_loss = normalized_loss / weight.pow(2) + 0.5 * torch.log(weight)
                else:
                    weighted_loss = normalized_loss / weight.pow(2) + torch.log(weight)

            loss_dict[name] = weighted_loss.detach()
            raw_loss_dict[name] = base_loss.detach()
            total_loss += weighted_loss

        kendall_weights = {name: F.softplus(self.raw_weights[i]).item() for i, name in enumerate(self.loss_names)}

        return total_loss, loss_dict, raw_loss_dict, kendall_weights


class AsymmetricLoss(nn.Module):
    def __init__(self, class_weights=None, gamma_neg=4, gamma_pos=1, clip=0.05):
        """
        Asymmetric loss.

        Args:
            class_weights (torch.Tensor, optional): Class weights. Defaults to None.
            gamma_neg (float, optional): Gamma value for negative examples. Defaults to 4.
            gamma_pos (float, optional): Gamma value for positive examples. Defaults to 1.
            clip (float, optional): Clip value. Defaults to 0.05.
        """
        super().__init__()
        self.class_weights = class_weights  # shape [C] or None
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits, targets):
        """
        Forward pass.

        Args:
            logits (torch.Tensor): Logits tensor.
            targets (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Loss tensor.
        """
        probs = torch.sigmoid(logits)

        pos = targets
        neg = 1 - targets

        xs_pos = probs.clamp(min=1e-8, max=1 - 1e-8)
        xs_neg = (1 - probs).clamp(min=1e-8, max=1 - 1e-8)

        loss_pos = pos * torch.log(xs_pos)
        loss_neg = neg * torch.log(xs_neg)

        loss_pos *= (1 - xs_pos).pow(self.gamma_pos)
        loss_neg *= xs_pos.pow(self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.class_weights is not None:
            loss = loss * self.class_weights.view(1, -1)

        return loss.mean()


def encode_single(label, vocab):
    return torch.tensor(vocab[label], dtype=torch.long)


def encode_multilabel(labels, vocab):
    """
    Encode a list of labels into a tensor of 0s and 1s.

    Args:
        labels (list): List of labels.
        vocab (dict): Vocabulary dictionary.

    Returns:
        torch.Tensor: Tensor of 0s and 1s.
    """
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for label in labels:
        if label in vocab:
            vec[vocab[label]] = 1.0
    return vec


def encode_portion(portion_list, vocab):
    """
    Encode a portion list into a tensor of 0s and 1s.

    Args:
        portion_list (list): List of (name, weight) tuples.
        vocab (dict): Vocabulary dictionary.

    Returns:
        torch.Tensor: Tensor of 0s and 1s.
    """
    presence = torch.zeros(len(vocab), dtype=torch.float32)
    weight = torch.zeros(len(vocab), dtype=torch.float32)
    for name, w in portion_list:
        if name in vocab:
            idx = vocab[name]
            presence[idx] = 1.0
            weight[idx] = float(w)

    return presence, weight


def build_class_weights(file_name: str, vocab: Dict[str, int], ignore_class: str = None):
    """
    Build class weights.

    Args:
        file_name (str): File name.
        vocab (dict): Vocabulary dictionary.
        ignore_class (str, optional): Class to ignore. Defaults to None.

    Returns:
        torch.Tensor: Class weights tensor.
    """
    df = pd.read_csv(os.path.join(ROOT_DIR, "stats", "data", file_name))

    key_col = df.columns[0]
    val_col = df.columns[1]

    counts = df[val_col].values.astype(np.float32)

    weights = 1 / np.sqrt(counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.2, 5.0)

    df["weight"] = weights

    if ignore_class is not None:
        df.loc[df[key_col] == ignore_class, "weight"] = 0.0

    weight_tensor = torch.ones(len(vocab))

    for _, row in df.iterrows():
        cls = row[key_col]
        if cls in vocab:
            weight_tensor[vocab[cls]] = row["weight"]
    return weight_tensor


def collate_fn(batch):
    """
    Collate function for the dataset.

    Args:
        batch (list): Batch of data.

    Returns:
        dict: Collated data.
    """
    images = [item["image"] for item in batch]

    # get max size
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape

        pad_h = max_h - h
        pad_w = max_w - w

        # pad: (left, right, top, bottom)
        padded = F.pad(img, (0, pad_w, 0, pad_h))
        padded_images.append(padded)

    images = torch.stack(padded_images)

    # collate the rest normally
    collated = {}
    for key in batch[0].keys():
        if key == "image":
            collated[key] = images
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated


def stress_test(
    model: SwinModel,
    total_loss: LossCombiner,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int = 2,
    height: int = 448,
    width: int = 672,
):
    """
    Stress test the model.

    Args:
        model (SwinModel): Model to test.
        total_loss (LossCombiner): Loss combiner.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Device to run on.
        batch_size (int, optional): Batch size. Defaults to 2.
        height (int, optional): Image height. Defaults to 448.
        width (int, optional): Image width. Defaults to 672.
    """
    model.train()

    print(f"Testing batch={batch_size}, resolution={height}x{width}")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dummy_batch = {
        "image": torch.randn(
            batch_size,
            3,
            height,
            width,
            device=device,
        ),
        # classification targets
        "food_type": torch.randint(
            0,
            model.food_type_classes,
            (batch_size,),
            device=device,
        ),
        "dish_name": torch.randint(
            0,
            model.dish_names_classes,
            (batch_size,),
            device=device,
        ),
        # multilabel targets
        "ingredients": torch.randint(
            0,
            2,
            (batch_size, model.ingredients_classes),
            device=device,
        ).float(),
        "portion_presence": torch.randint(
            0,
            2,
            (batch_size, model.portion_size_classes),
            device=device,
        ).float(),
        "cooking_method": torch.randint(
            0,
            2,
            (batch_size, model.cooking_method_classes),
            device=device,
        ).float(),
        # regressions
        "portion_weight": torch.randn(
            batch_size,
            model.portion_size_classes,
            device=device,
        ),
        "calories_kcal": torch.randn(batch_size, device=device),
        "fat_g": torch.randn(batch_size, device=device),
        "carbohydrate_g": torch.randn(batch_size, device=device),
        "protein_g": torch.randn(batch_size, device=device),
        "food_prob": torch.rand(batch_size, device=device),
        "camera_or_phone_prob": torch.rand(batch_size, device=device),
    }

    optimizer.zero_grad(set_to_none=True)

    outputs = model(dummy_batch["image"])
    loss, _, _, _ = total_loss(outputs, dummy_batch)

    loss.backward()

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3

    print("Loss:", loss.item())
    print(f"Peak VRAM: {peak_mem:.2f} GB")

    optimizer.zero_grad(set_to_none=True)

    print("Stress test passed.")


def val_single_class(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    name: str,
    topk: int,
) -> Tuple[int, int, int]:
    """
    Validate a single class.

    Args:
        outputs (Dict[str, torch.Tensor]): Dictionary of output tensors.
        batch (Dict[str, torch.Tensor]): Dictionary of target tensors.
        name (str): Name of the class.
        topk (int): Top k accuracy.

    Returns:
        Tuple[int, int, int]: Correct, total, correctk.
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
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    name: str,
    data_name: str,
) -> Tuple[float, int]:
    """
    Validate a regression task.

    Args:
        outputs (Dict[str, torch.Tensor]): Dictionary of output tensors.
        batch (Dict[str, torch.Tensor]): Dictionary of target tensors.
        name (str): Name of the task.
        data_name (str): Name of the data.

    Returns:
        Tuple[float, int]: MAE, count.
    """
    prediction = outputs[f"{name}_logits"].squeeze(1)

    target = batch[f"{data_name}"]

    mae_sum = torch.abs(prediction - target).sum().item()

    count = target.numel()

    return mae_sum, count


def val_binary_prob(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    logit_idx: int,
    target_name: str,
) -> Tuple[float, int]:
    """
    Validate a binary probability task.

    Args:
        outputs (Dict[str, torch.Tensor]): Dictionary of output tensors.
        batch (Dict[str, torch.Tensor]): Dictionary of target tensors.
        logit_idx (int): Index of the logit to use.
        target_name (str): Name of the target.

    Returns:
        Tuple[float, int]: MAE, count.
    """

    prediction = torch.sigmoid(outputs["binary_logits"][:, logit_idx])

    target = batch[target_name]

    mae_sum = torch.abs(prediction - target).sum().item()

    count = target.numel()

    return mae_sum, count


def build_log_row(metrics: dict, headers: list[str]):
    """
    Build a log row for a CSV file.

    Args:
        metrics (dict): Dictionary of metrics.
        headers (list[str]): List of headers.

    Returns:
        list: List of values.
    """
    row = []
    for h in headers:
        if h not in metrics:
            raise KeyError(f"Missing metric: {h}")
        row.append(metrics[h])
    return row


def evaluate(model, dataloader, device):
    """
    Evaluate a model.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader to use.
        device (torch.device): Device to use.

    Returns:
        dict: Dictionary of metrics.
    """
    model.eval()

    val_total_loss = 0.0
    val_batches = 0

    cls_correct = defaultdict(int)
    cls_total = defaultdict(int)
    cls_topk_correct = defaultdict(int)

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model(batch["image"])

            loss, loss_dict, _, _ = total_loss(outputs, batch)

            val_total_loss += loss.item()

            val_batches += 1

            correct, total, correctk = val_single_class(outputs, batch, "food_type", 5)
            cls_correct["food_type"] += correct
            cls_total["food_type"] += total
            cls_topk_correct["food_type"] += correctk
    return {
        "val_total_loss": val_total_loss,
        "val_batches": val_batches,
        "cls_correct": cls_correct,
        "cls_total": cls_total,
        "cls_topk_correct": cls_topk_correct,
    }


def make_confusion_matrix(model, dataloader, device):
    """
    Make a confusion matrix.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader to use.
        device (torch.device): Device to use.

    Returns:
        None
    """
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model(batch["image"])["food_type_logits"]
            preds = outputs.argmax(dim=1)

            y_pred.extend(preds.cpu().numpy())
            y_true.extend(batch["food_type"].cpu().numpy())

    # Explicit class ordering
    labels = [0, 4, 3, 2, 1]

    class_names = [
        "Homemade food",
        "Restaurant food",
        "Raw vegetables and fruits",
        "Packaged food",
        "Others",
    ]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    save_dir = os.path.join(ROOT_DIR, "stats", "results")
    os.makedirs(save_dir, exist_ok=True)

    # Save CSV
    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )

    csv_path = os.path.join(
        save_dir,
        "food_type_confusion_matrix.csv",
    )
    cm_df.to_csv(csv_path)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=30,
    )

    ax.set_title(
        "Food Type Test Confusion Matrix",
        fontsize=16,
        pad=20,
    )

    plt.tight_layout()

    image_path = os.path.join(
        save_dir,
        "food_type_confusion_matrix.png",
    )

    plt.savefig(
        image_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    return cm


def compute_metrics(state, model_name, epoch=None):
    """
    Compute metrics.

    Args:
        state (dict): Dictionary of metrics.
        model_name (str): Name of the model.
        epoch (int): Epoch number.

    Returns:
        dict: Dictionary of metrics.
    """

    val_total_loss = state["val_total_loss"] / state["val_batches"]

    food_type_acc = state["cls_correct"]["food_type"] / max(state["cls_total"]["food_type"], 1)

    food_type_top5_acc = state["cls_topk_correct"]["food_type"] / max(state["cls_total"]["food_type"], 1)

    metrics = {
        "epoch": epoch,
        "model_name": model_name,
        "total_loss": val_total_loss,
        "food_type_acc": food_type_acc,
        "food_type_top5_acc": food_type_top5_acc,
    }

    return metrics


def save_checkpoint(
    path,
    epoch,
    training_step,
    model,
    total_loss,
    model_optimizer,
    weight_optimizer,
    scheduler,
    scaler,
):
    """
    Save checkpoint.

    Args:
        path (str): Path to save checkpoint.
        epoch (int): Epoch number.
        training_step (int): Training step number.
        model (nn.Module): Model to save.
        total_loss (LossCombiner): Total loss.
        model_optimizer (Optimizer): Model optimizer.
        weight_optimizer (Optimizer): Weight optimizer.
        scheduler (Scheduler): Scheduler.
        scaler (GradScaler): Grad scaler.

    Returns:
        None
    """
    checkpoint = {
        "epoch": epoch,
        "training_step": training_step,
        "model_state_dict": model.state_dict(),
        "model_optimizer_state_dict": model_optimizer.state_dict(),
        "weight_optimizer_state_dict": weight_optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss_combiner_state_dict": total_loss.state_dict(),
        "loss_phase": total_loss.phase,
        "loss_means": total_loss.loss_means,
        "loss_sum": total_loss.loss_sum,
        "loss_counts": total_loss.loss_counts,
    }

    torch.save(checkpoint, path)


if __name__ == "__main__":
    # ------------------------
    # Loading data
    # ------------------------

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    model_name = "single"
    REGRESSION_METRICS = ["calories"]

    train = pd.read_parquet(os.path.join(DATA_DIR, "train_labels_sorted.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val_labels_sorted.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test_labels_sorted.parquet"))

    image_norms = json.load(open(os.path.join(ROOT_DIR, "stats", "data", "image_normalization.json"), "r"))
    regression_norms = json.load(
        open(
            os.path.join(ROOT_DIR, "stats", "data", "regression_labels_stats.json"),
            "r",
        )
    )
    image_norm_mean = image_norms["mean"]
    image_norm_std = image_norms["std"]
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(image_norm_mean, image_norm_std)])

    food_type_vocab, food_type_inverse_vocab = build_vocab(train["food_type"])
    dish_name_vocab, dish_name_inverse_vocab = build_vocab(train["dish_name"])
    cooking_method_vocab, cooking_method_inverse_vocab = build_multilabel_vocab(train["cooking_method"])

    train["portion_size"] = train["portion_size"].apply(ast.literal_eval)
    val["portion_size"] = val["portion_size"].apply(ast.literal_eval)
    test["portion_size"] = test["portion_size"].apply(ast.literal_eval)

    portion_ingredient_vocab, portion_ingredient_inverse_vocab = build_multilabel_vocab(
        train["portion_size"].apply(lambda x: list(i[0] for i in x))
    )
    ingredients_vocab, ingredients_inverse_vocab = build_multilabel_vocab(train["ingredients"])

    train_dataset = FoodDataset(
        train,
        food_type_vocab,
        dish_name_vocab,
        cooking_method_vocab,
        ingredients_vocab,
        portion_ingredient_vocab,
        regression_norms,
    )
    val_dataset = FoodDataset(
        val,
        food_type_vocab,
        dish_name_vocab,
        cooking_method_vocab,
        ingredients_vocab,
        portion_ingredient_vocab,
        regression_norms,
    )
    test_dataset = FoodDataset(
        test,
        food_type_vocab,
        dish_name_vocab,
        cooking_method_vocab,
        ingredients_vocab,
        portion_ingredient_vocab,
        regression_norms,
    )

    # ------------------------
    # Loading model
    # ------------------------

    device = torch.device("cuda")

    x = torch.randn(2, 3, 224, math.ceil(224 * 4.833753148614609), device=device)

    model = SwinModel(
        heads_ratio=16,
        dim=64,
        dropout_mha=0,
        dropout_swin_mlp=0,
        dropout_outer=0,
        dropout_shared_mlp=0.25,
        dropout_pre_output=0.1,
        droppath=0.1,
        window_size=7,
        input_channels=3,
        depths=[2, 2, 18, 2],
        stage_num=4,
        shared_mlp_size=1024,
        ingredients_mlp_size=768,
        portions_mlp_size=768,
        dish_names_mlp_size=768,
        food_type_classes=5,
        ingredients_classes=589,
        portion_size_classes=437,
        dish_names_classes=602,
        cooking_method_classes=15,
        binary_classes=2,
    ).to(device)
    model.apply(init_weights)
    out = model(x)
    for i in out:
        print(i, out[i].shape)

    summary(model, input_size=x.shape)
    y = model(x)
    out = sum(v.sum() for v in y.values())
    dot = make_dot(out, params=dict(model.named_parameters()))
    dot.render("model_graph", format="pdf")

    # ------------------------
    # Training setup
    # ------------------------

    epochs = 15
    effective_batch_size = 32
    micro_batch_size = 2
    steps_per_epoch = train.shape[0] // effective_batch_size
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    ema_decay = 0.99

    cooking_method_weights = build_class_weights(
        "cooking_method_counter.csv", cooking_method_vocab, ignore_class="unknown"
    )
    dish_name_weights = build_class_weights("dish_name_counter.csv", dish_name_vocab)
    food_type_weights = build_class_weights("food_type_counter.csv", food_type_vocab)
    ingredient_weights = build_class_weights("ingredient_counter.csv", ingredients_vocab)
    portion_ingredient_weights = build_class_weights("weight_ingredient_counter.csv", portion_ingredient_vocab)
    food_type_weights = food_type_weights.to(device)
    dish_name_weights = dish_name_weights.to(device)
    ingredient_weights = ingredient_weights.to(device)
    cooking_method_weights = cooking_method_weights.to(device)
    portion_ingredient_weights = portion_ingredient_weights.to(device)

    loss_dictionary = {"food_type": nn.CrossEntropyLoss(weight=food_type_weights)}
    print(food_type_weights)

    total_loss = LossCombiner(losses=loss_dictionary)

    METRIC_SCHEMA = {
        "meta": ["epoch", "model_name", "total_loss"],
        "classification": ["food_type_acc"],
    }
    HEADERS = sum(METRIC_SCHEMA.values(), [])

    train_set = DataLoader(train_dataset, batch_size=micro_batch_size, collate_fn=collate_fn)
    val_set = DataLoader(val_dataset, batch_size=micro_batch_size, collate_fn=collate_fn)
    test_set = DataLoader(test_dataset, batch_size=micro_batch_size, collate_fn=collate_fn)

    log_path = os.path.join(ROOT_DIR, "stats", "losses2.csv")
    checkpoint_path = os.path.join(os.path.dirname(log_path), "checkpoint2.pt")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            headers = ["epoch", "step", "total_loss", "ema_moving_loss"]
            f.write(",".join(headers) + "\n")

    val_log_path = os.path.join(ROOT_DIR, "stats", "validation_log.csv")

    test_log_path = os.path.join(ROOT_DIR, "stats", "test_log.csv")

    if not os.path.exists(val_log_path):
        with open(val_log_path, "w") as f:
            f.write(",".join(HEADERS) + "\n")

    if not os.path.exists(test_log_path):
        with open(test_log_path, "w") as f:
            f.write(",".join(HEADERS) + "\n")

    model_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    weight_optimizer = torch.optim.Adam(
        total_loss.parameters(),
        lr=1e-2,
    )

    scheduler = CosineAnnealingLR(model_optimizer, T_max=total_steps, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")

    stress_test(
        model,
        total_loss,
        model_optimizer,
        device,
        batch_size=2,
        height=224,
        width=math.ceil(224 * 4.833753148614609),
    )

    assert effective_batch_size % micro_batch_size == 0, (
        "Batch size must be divisible by micro batch size, no trunkation allowed"
    )
    accumulation_steps = effective_batch_size // micro_batch_size
    training_step = 0
    start_epoch = 0

    # ------------------------
    # Resume training
    # ------------------------

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        # model
        model.load_state_dict(checkpoint["model_state_dict"])

        # optimizers
        model_optimizer.load_state_dict(checkpoint["model_optimizer_state_dict"])

        weight_optimizer.load_state_dict(checkpoint["weight_optimizer_state_dict"])

        # scheduler
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # AMP scaler
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Kendall weights
        total_loss.load_state_dict(checkpoint["loss_combiner_state_dict"])

        # custom LossCombiner state
        total_loss.phase = checkpoint["loss_phase"]
        total_loss.loss_means = checkpoint["loss_means"]
        total_loss.loss_sum = checkpoint["loss_sum"]
        total_loss.loss_counts = checkpoint["loss_counts"]

        # resume position
        start_epoch = checkpoint["epoch"] + 1
        training_step = checkpoint["training_step"]

        print(f"Resuming from epoch={start_epoch}, training_step={training_step}")

    # ------------------------
    # Training loop
    # ------------------------

    for epoch in range(start_epoch, epochs):
        model.train()
        model_optimizer.zero_grad(set_to_none=True)
        weight_optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(
            total=math.ceil(len(train_dataset) / effective_batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        accum_counter = 0
        running_loss = 0.0

        ema_score = 0.0

        for batch in train_set:
            # ------------------------
            # Forward pass
            # ------------------------

            found_inf = False
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # with torch.amp.autocast("cuda", dtype=torch.float16):
            outputs = model(batch["image"])
            for i in outputs:
                if not torch.isfinite(outputs[i]).all():
                    print(f"Invalid {i} logits at step", training_step)
                    found_inf = True

            loss = loss_dictionary["food_type"](outputs["food_type_logits"], batch["food_type"])
            if not torch.isfinite(loss):
                print("Invalid global loss at step", training_step)
                found_inf = True

            running_loss += loss.item()

            # ------------------------
            # Backward pass with safeguard
            # ------------------------

            if found_inf:
                accum_counter = 0
                running_loss = 0.0
                running_all_losses = {name: 0.0 for name in total_loss.loss_names}
                model_optimizer.zero_grad(set_to_none=True)
                weight_optimizer.zero_grad(set_to_none=True)
                continue

            loss = loss / accumulation_steps
            loss.backward()
            """
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("Invalid gradients at step", training_step)
                    found_inf = True
            """  # use only if training collapses

            accum_counter += 1

            if accum_counter == accumulation_steps:
                # ------------------------
                # Update weights
                # ------------------------

                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()),
                        max_norm=1.0,
                        error_if_nonfinite=True,  # if training collapses, return to True, currently it does not skip broken batches  # noqa: E501
                    )
                except RuntimeError as e:
                    print("Gradient error:", e)
                    print("Invalid gradients at step", training_step)
                    if training_step < 1000:
                        pass
                    else:
                        accum_counter = 0
                        running_loss = 0.0

                        scaler.update()
                        model_optimizer.zero_grad(set_to_none=True)
                        continue
                model_optimizer.step()
                weight_optimizer.step()
                scheduler.step()
                weight_optimizer.zero_grad(set_to_none=True)
                model_optimizer.zero_grad(set_to_none=True)

                effective_loss = running_loss / accumulation_steps

                ema_score = ema_decay * ema_score + (1 - ema_decay) * effective_loss

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{effective_loss:.4f}",
                        "ema_loss": f"{ema_score:.4f}",
                        "lr": scheduler.get_last_lr()[0],
                    }
                )

                with open(log_path, "a") as f:
                    row = [
                        epoch,
                        pbar.n,
                        f"{effective_loss:.4f}",
                        f"{ema_score:.4f}",
                    ]
                    f.write(",".join(map(str, row)) + "\n")

                accum_counter = 0
                running_loss = 0.0
                running_all_losses = {name: 0.0 for name in total_loss.loss_names}
                training_step += 1

        if accum_counter > 0:
            model_optimizer.step()
            model_optimizer.zero_grad(set_to_none=True)

            effective_loss = running_loss / accum_counter

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{effective_loss:.4f}",
                    "ema_loss": f"{ema_score:.4f}",
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        # ------------------------
        # Validation and checkpoint
        # ------------------------

        val_state = evaluate(model, val_set, device)
        metrics = compute_metrics(val_state, model_name=model_name, epoch=epoch)
        print(metrics, HEADERS)
        save_checkpoint(
            checkpoint_path,
            epoch,
            training_step,
            model,
            total_loss,
            model_optimizer,
            weight_optimizer,
            scheduler,
            scaler,
        )

        row = build_log_row(metrics, HEADERS)

        with open(val_log_path, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

        wandb.log(metrics, step=epoch)
    # test_state = evaluate(model, test_set, device)
    # test_metrics = compute_metrics(test_state, model_name=model_name)
    print(make_confusion_matrix(model, test_set, device))

    # row = build_log_row(test_metrics, HEADERS)

    # with open(test_log_path, "a") as f:
    #    f.write(",".join(map(str, row)) + "\n")
    # wandb.log(test_metrics)
    wandb.finish()
