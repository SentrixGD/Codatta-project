import ast
import json
import math
import os
from collections import defaultdict
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from sklearn import f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchinfo import summary
from torchvision import transforms
from torchviz import make_dot
from tqdm import tqdm

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
    unique = sorted(set(values))
    stoi = {v: i for i, v in enumerate(unique)}
    itos = {i: v for v, i in stoi.items()}
    return stoi, itos


def build_multilabel_vocab(column):
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
    def __init__(self, losses: Dict[str, nn.Module], ema_factor: float):
        super().__init__()
        self.losses = losses

        self.loss_names = list(losses.keys())
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))  # s = log(sigma^2)
        self.ema_factor = ema_factor
        self.ema_scores = {name: 0.0 for name in self.loss_names}

        self.tasks = {
            "food_type": {"target": "food_type", "type": "cls"},
            "ingredients": {"target": "ingredients", "type": "multi"},
            "portion_presence": {"target": "portion_presence", "type": "multi"},
            "cooking_method": {"target": "cooking_method", "type": "multi"},
            "dish_name": {"target": "dish_name", "type": "cls"},
            "portion_weight": {"target": "portion_weight", "type": "reg"},
            "calories": {"target": "calories_kcal", "type": "reg"},
            "fats": {"target": "fat_g", "type": "reg"},
            "carbohydrates": {"target": "carbohydrate_g", "type": "reg"},
            "proteins": {"target": "protein_g", "type": "reg"},
            "binary": {
                "target": ["food_prob", "camera_or_phone_prob"],
                "type": "multi_bin",
            },
        }

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        total_loss = 0.0
        loss_dict = {}

        for i, name in enumerate(self.loss_names):
            s = self.log_vars[i]
            task = self.tasks[name]

            if isinstance(task["target"], list):
                target = torch.stack([targets[t] for t in task["target"]], dim=1)  # shape [B, 2]
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
            self.ema_scores[name] = self.ema_factor * self.ema_scores[name] + (1 - self.ema_factor) * base_loss.detach()

            scaled_loss = base_loss / (self.ema_scores[name] + 1e-8)

            if task["type"] == "reg":
                weighted_loss = torch.exp(-s) * scaled_loss + 0.5 * s
            else:
                weighted_loss = torch.exp(-s) * scaled_loss + s

            total_loss += weighted_loss
            loss_dict[name] = scaled_loss.detach()

        kendall_weights = {name: torch.exp(-self.log_vars[i]).item() for i, name in enumerate(self.loss_names)}

        return total_loss, loss_dict, kendall_weights


def encode_single(label, vocab):
    return torch.tensor(vocab[label], dtype=torch.long)


def encode_multilabel(labels, vocab):
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for label in labels:
        if label in vocab:
            vec[vocab[label]] = 1.0
    return vec


def encode_portion(portion_list, vocab):
    presence = torch.zeros(len(vocab), dtype=torch.float32)
    weight = torch.zeros(len(vocab), dtype=torch.float32)
    for name, w in portion_list:
        if name in vocab:
            idx = vocab[name]
            presence[idx] = 1.0
            weight[idx] = float(w)

    return presence, weight


def build_class_weights(file_name: str, vocab: Dict[str, int], ignore_class: str = None):
    df = pd.read_csv(os.path.join(ROOT_DIR, "stats", "data", file_name))

    key_col = df.columns[0]
    val_col = df.columns[1]

    counts = df[val_col].values.astype(np.float32)

    weights = 1 / np.sqrt(counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.3, 3.0)

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


def stress_test(model, total_loss, optimizer, device, batch_size=2, height=448, width=672):
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

    with torch.amp.autocast("cuda", dtype=torch.float16):
        outputs = model(dummy_batch["image"])
        loss, _, _ = total_loss(outputs, dummy_batch)

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
    prediction = outputs[f"{name}_logits"].squeeze(1)

    target = batch[f"{data_name}"]

    mae_sum = torch.abs(prediction - target).sum().item()

    count = target.numel()

    return mae_sum, count


def build_log_row(metrics: dict, headers: list[str]):
    row = []
    for h in headers:
        if h not in metrics:
            raise KeyError(f"Missing metric: {h}")
        row.append(metrics[h])
    return row


def evaluate(model, dataloader, device):
    model.eval()

    val_losses = {name: 0.0 for name in total_loss.loss_names}
    val_total_loss = 0.0
    val_batches = 0

    cls_correct = defaultdict(int)
    cls_total = defaultdict(int)
    cls_topk_correct = defaultdict(int)

    multilabel_preds = {name: [] for name in MULTILABEL_TASKS}
    multilabel_targets = {name: [] for name in MULTILABEL_TASKS}

    regression_mae_sum = {name: 0.0 for name, _ in REGRESSION_TASKS}
    regression_count = {name: 0 for name, _ in REGRESSION_TASKS}

    portion_weight_abs_sum = 0.0
    portion_weight_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = model(batch["image"])

            loss, loss_dict, _ = total_loss(outputs, batch)

            val_total_loss += loss.item()

            for k, v in loss_dict.items():
                val_losses[k] += v.item()

            val_batches += 1

            # classification
            for name, topk in CLASSIFICATION_TASKS:
                correct, total, correctk = val_single_class(outputs, batch, name, topk)

                cls_correct[name] += correct
                cls_total[name] += total
                cls_topk_correct[name] += correctk

            # regression
            for task_name, target_name in REGRESSION_TASKS:
                mae_sum, count = val_regression(outputs, batch, task_name, target_name)

                regression_mae_sum[task_name] += mae_sum
                regression_count[task_name] += count

            # multilabel
            for name in MULTILABEL_TASKS:
                pred = (torch.sigmoid(outputs[f"{name}_logits"]) > 0.5).cpu()
                target = batch[name].cpu()

                multilabel_preds[name].append(pred)
                multilabel_targets[name].append(target)

            # custom
            pred = outputs["portion_weight_logits"]
            target = batch["portion_weight"]
            mask = batch["portion_presence"].bool()

            portion_weight_abs_sum += torch.abs(pred[mask] - target[mask]).sum().item()
            portion_weight_count += mask.sum().item()

    return {
        "val_losses": val_losses,
        "val_total_loss": val_total_loss,
        "val_batches": val_batches,
        "cls_correct": cls_correct,
        "cls_total": cls_total,
        "cls_topk_correct": cls_topk_correct,
        "multilabel_preds": multilabel_preds,
        "multilabel_targets": multilabel_targets,
        "regression_mae_sum": regression_mae_sum,
        "regression_count": regression_count,
        "portion_weight_abs_sum": portion_weight_abs_sum,
        "portion_weight_count": portion_weight_count,
    }


def compute_metrics(state, epoch=None):
    val_losses = {k: v / state["val_batches"] for k, v in state["val_losses"].items()}

    val_total_loss = state["val_total_loss"] / state["val_batches"]

    classification_metrics = {
        name: state["cls_correct"][name] / state["cls_total"][name] for name, _ in CLASSIFICATION_TASKS
    }

    regression_metrics = {
        f"{name}_mae": state["regression_mae_sum"][name] / state["regression_count"][name]
        for name, _ in REGRESSION_TASKS
    }

    ingredients_pred = torch.cat(state["multilabel_preds"]["ingredients"]).cpu().numpy()
    ingredients_target = torch.cat(state["multilabel_targets"]["ingredients"]).cpu().numpy()

    multilabel_metrics = {
        "ingredients_micro_f1": f1_score(ingredients_target, ingredients_pred, average="micro"),
        "ingredients_macro_f1": f1_score(ingredients_target, ingredients_pred, average="macro"),
    }

    portion_weight_mae = state["portion_weight_abs_sum"] / max(state["portion_weight_count"], 1)

    metrics = {
        "epoch": epoch,
        "val_total_loss": val_total_loss,
        **{f"loss_{k}": v for k, v in val_losses.items()},
        **classification_metrics,
        **regression_metrics,
        **multilabel_metrics,
        "portion_weight_mae": portion_weight_mae,
    }

    return metrics


if __name__ == "__main__":
    CLASSIFICATION_TASKS = [
        ("food_type", 1),
        ("dish_name", 5),
    ]

    MULTILABEL_TASKS = [
        "ingredients",
        "portion_presence",
        "cooking_method",
    ]

    REGRESSION_TASKS = [
        ("calories", "calories_kcal"),
        ("fats", "fat_g"),
        ("carbohydrates", "carbohydrate_g"),
        ("proteins", "protein_g"),
        ("camera_or_phone", "camera_or_phone_prob"),
        ("food", "food_prob"),
    ]

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

    device = torch.device("cuda")

    x = torch.randn(2, 3, 448, 672, device=device)

    model = SwinModel(
        heads_ratio=16,
        dim=48,
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

    epochs = 30
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

    loss_dictionary = {
        "food_type": nn.CrossEntropyLoss(label_smoothing=0.05, weight=food_type_weights),  # single-label
        "ingredients": nn.BCEWithLogitsLoss(pos_weight=ingredient_weights),  # multi-label
        "portion_presence": nn.BCEWithLogitsLoss(pos_weight=portion_ingredient_weights),  # multi-label
        "cooking_method": nn.BCEWithLogitsLoss(pos_weight=cooking_method_weights),  # multi-label
        "dish_name": nn.CrossEntropyLoss(label_smoothing=0.05, weight=dish_name_weights),  # single-label
        "portion_weight": nn.HuberLoss(),  # vector regression
        "calories": nn.HuberLoss(),
        "fats": nn.HuberLoss(),
        "carbohydrates": nn.HuberLoss(),
        "proteins": nn.HuberLoss(),
        "binary": nn.BCEWithLogitsLoss(),  # binary classification
    }

    total_loss = LossCombiner(losses=loss_dictionary, ema_factor=0.99)

    METRIC_SCHEMA = {
        "meta": ["epoch", "model_name", "dataset", "val_total_loss"],
        "loss": [f"loss_{name}" for name in total_loss.loss_names],
        "classification": [
            "food_type_acc",
            "dish_name_top1_acc",
            "dish_name_top5_acc",
        ],
        "regression": [
            "calories_mae",
            "fats_mae",
            "carbohydrates_mae",
            "proteins_mae",
            "camera_or_phone_mae",
            "food_mae",
        ],
        "multilabel": [
            "ingredients_micro_f1",
            "ingredients_macro_f1",
        ],
        "custom": [
            "portion_weight_mae",
        ],
    }
    HEADERS = sum(METRIC_SCHEMA.values(), [])

    train_set = DataLoader(train_dataset, batch_size=micro_batch_size, collate_fn=collate_fn)
    val_set = DataLoader(val_dataset, batch_size=micro_batch_size, collate_fn=collate_fn)
    test_set = DataLoader(test_dataset, batch_size=micro_batch_size, collate_fn=collate_fn)

    log_path = os.path.join(ROOT_DIR, "stats", "losses.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            headers = (
                ["epoch", "step", "total_loss", "ema_moving_loss"]
                + list(total_loss.loss_names)
                + [f"{name}_ema" for name in total_loss.loss_names]
                + [f"{name}_w" for name in total_loss.loss_names]
            )
            f.write(",".join(headers) + "\n")

    val_log_path = os.path.join(ROOT_DIR, "stats", "validation_log.csv")

    test_log_path = os.path.join(ROOT_DIR, "stats", "test_log.csv")

    if not os.path.exists(val_log_path):
        with open(val_log_path, "w") as f:
            f.write(",".join(headers) + "\n")

    adam = torch.optim.Adam(list(model.parameters()) + list(total_loss.parameters()), lr=1e-4)

    scheduler = CosineAnnealingLR(adam, T_max=total_steps - warmup_steps, eta_min=1e-6)

    scaler = torch.amp.GradScaler("cuda")

    stress_test(
        model,
        total_loss,
        adam,
        device,
        batch_size=2,
        height=448,
        width=int(448 * 3),
    )

    assert effective_batch_size % micro_batch_size == 0, (
        "Batch size must be divisible by micro batch size, no trunkation allowed"
    )
    accumulation_steps = effective_batch_size // micro_batch_size
    training_step = 0
    for epoch in range(epochs):
        model.train()
        adam.zero_grad(set_to_none=True)
        pbar = tqdm(
            total=math.ceil(len(train_dataset) / effective_batch_size),
            desc=f"Epoch {epoch + 1}/{epochs}",
        )
        accum_counter = 0
        running_loss = 0.0
        running_all_losses = {name: 0.0 for name in total_loss.loss_names}

        ema_score = 0.0
        ema_scores = {name: 0.0 for name in total_loss.loss_names}

        for batch in train_set:
            found_inf = False
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(batch["image"])
                for i in outputs:
                    if not torch.isfinite(outputs[i]).all():
                        print(f"Invalid {i} logits at step", training_step)
                        found_inf = True

                loss, loss_dict, kendall_weights = total_loss(outputs, batch)
                if not torch.isfinite(loss):
                    print("Invalid global loss at step", training_step)
                    found_inf = True
                for i in loss_dict:
                    if not torch.isfinite(loss_dict[i]).all():
                        print(f"Invalid loss {i} at step", training_step)
                        found_inf = True

            running_loss += loss.item()

            for k, v in loss_dict.items():
                running_all_losses[k] += v.item()

            if found_inf:
                accum_counter = 0
                running_loss = 0.0
                running_all_losses = {name: 0.0 for name in total_loss.loss_names}
                adam.zero_grad(set_to_none=True)
                continue

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            """
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("Invalid gradients at step", training_step)
                    found_inf = True
            """  # use only if training collapses

            accum_counter += 1

            if accum_counter == accumulation_steps:
                scaler.unscale_(adam)
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(total_loss.parameters()),
                        max_norm=1.0,
                        error_if_nonfinite=False,  # if training collapses, return to True, currently it does not skip broken batches
                    )
                except RuntimeError as e:
                    print("Gradient error:", e)
                    print("Invalid gradients at step", training_step)

                    accum_counter = 0
                    running_loss = 0.0
                    running_all_losses = {name: 0.0 for name in total_loss.loss_names}

                    scaler.update()
                    adam.zero_grad(set_to_none=True)
                    continue
                scaler.step(adam)
                scaler.update()
                adam.zero_grad(set_to_none=True)

                effective_loss = running_loss / accumulation_steps
                effective_losses = {k: v / accumulation_steps for k, v in running_all_losses.items()}

                ema_score = ema_decay * ema_score + (1 - ema_decay) * effective_loss
                ema_scores = {k: v * ema_decay + (1 - ema_decay) * effective_losses[k] for k, v in ema_scores.items()}
                pbar.update(1)
                pbar.set_postfix(
                    {
                        "loss": f"{effective_loss:.4f}",
                        "ema_loss": f"{ema_score:.4f}",
                    }
                )

                with open(log_path, "a") as f:
                    row = [
                        epoch,
                        pbar.n,
                        f"{effective_loss:.4f}",
                        f"{ema_score:.4f}",
                        *[f"{effective_losses[name]:.4f}" for name in total_loss.loss_names],
                        *[f"{ema_scores[name]:.4f}" for name in total_loss.loss_names],
                        *[f"{kendall_weights[name]:.4f}" for name in total_loss.loss_names],
                    ]
                    f.write(",".join(map(str, row)) + "\n")

                accum_counter = 0
                running_loss = 0.0
                running_all_losses = {name: 0.0 for name in total_loss.loss_names}
                training_step += 1

        if accum_counter > 0:
            adam.step()
            adam.zero_grad(set_to_none=True)

            effective_loss = running_loss / accum_counter

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{effective_loss:.4f}",
                    "ema_loss": f"{ema_score:.4f}",
                }
            )
        val_state = evaluate(model, val_set, device)
        metrics = compute_metrics(val_state, epoch=epoch)

        row = build_log_row(metrics, HEADERS)

        with open(val_log_path, "a") as f:
            f.write(",".join(map(str, row)) + "\n")

        wandb.log(metrics, step=epoch)
    test_state = evaluate(model, test_set, device)
    test_metrics = compute_metrics(test_state)

    row = build_log_row(metrics, HEADERS)

    with open(test_log_path, "a") as f:
        f.write(",".join(map(str, row)) + "\n")
    wandb.log(metrics)
    wandb.finish()
