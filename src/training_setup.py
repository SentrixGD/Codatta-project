import os

from torchinfo import summary

from src.model import SwinModel, init_weights
import pandas as pd
import torch
from torchviz import make_dot
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math
from torchvision import transforms
from tqdm import tqdm
from typing import Tuple, Dict, List, Callable
import json
from collections import Counter
import ast
import numpy as np
from torch import nn
import torch.nn.functional as F

LossType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")


def normalize(s):
    return s.replace(" ", "_").lower()


def load_image(image_id: str) -> torch.Tensor:
    path = os.path.join(DATA_DIR, f"resized_images", image_id)
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

        targets["cooking_method"] = encode_multilabel(
            row["cooking_method"], self.cooking_method_vocab
        )

        targets["ingredients"] = encode_multilabel(
            row["ingredients"], self.ingredients_vocab
        )

        presence, weight = encode_portion(
            row["portion_size"], self.portion_ingredients_vocab
        )

        targets["portion_presence"] = presence
        weight = weight.to(torch.float32)

        mask = weight > 0

        targets["portion_weight"] = torch.zeros_like(weight)

        targets["portion_weight"][mask] = (
            np.log1p(weight[mask])
            - self.regression_normalization["portion_size"]["log_mean"]
        ) / (self.regression_normalization["portion_size"]["log_std"])

        targets["fat_g"] = torch.tensor(
            np.log1p(row["fat_g"])
            - self.regression_normalization["fat_g"]["log_mean"]
            / (self.regression_normalization["fat_g"]["log_std"]),
            dtype=torch.float32,
        )
        targets["protein_g"] = torch.tensor(
            np.log1p(row["protein_g"])
            - self.regression_normalization["protein_g"]["log_mean"]
            / (self.regression_normalization["protein_g"]["log_std"]),
            dtype=torch.float32,
        )
        targets["carbohydrate_g"] = torch.tensor(
            np.log1p(row["carbohydrate_g"])
            - self.regression_normalization["carbohydrate_g"]["log_mean"]
            / (self.regression_normalization["carbohydrate_g"]["log_std"]),
            dtype=torch.float32,
        )
        targets["calories_kcal"] = torch.tensor(
            np.log1p(row["calories_kcal"])
            - self.regression_normalization["calories_kcal"]["log_mean"]
            / (self.regression_normalization["calories_kcal"]["log_std"]),
            dtype=torch.float32,
        )

        targets["camera_or_phone_prob"] = torch.tensor(
            row["camera_or_phone_prob"], dtype=torch.float32
        )
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
        super().__init__()
        self.losses = losses

        self.loss_names = list(losses.keys())
        self.log_vars = nn.Parameter(torch.zeros(len(losses)))  # s = log(sigma^2)

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

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ):
        total_loss = 0.0
        loss_dict = {}

        for i, name in enumerate(self.loss_names):

            s = self.log_vars[i]
            task = self.tasks[name]

            if isinstance(task["target"], list):
                target = torch.stack(
                    [targets[t] for t in task["target"]], dim=1
                )  # shape [B, 2]
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

            if task["type"] == "reg":
                weighted_loss = torch.exp(-s) * base_loss + 0.5 * s
            else:
                weighted_loss = torch.exp(-s) * base_loss + s

            total_loss += weighted_loss
            loss_dict[name] = base_loss.detach()

        kendall_weights = {
            name: torch.exp(-self.log_vars[i]).item()
            for i, name in enumerate(self.loss_names)
        }

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


def build_class_weights(
    file_name: str, vocab: Dict[str, int], ignore_class: str = None
):
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


if __name__ == "__main__":

    train = pd.read_parquet(os.path.join(DATA_DIR, "train_labels_sorted.parquet"))
    val = pd.read_parquet(os.path.join(DATA_DIR, "val_labels_sorted.parquet"))
    test = pd.read_parquet(os.path.join(DATA_DIR, "test_labels_sorted.parquet"))

    image_norms = json.load(
        open(os.path.join(ROOT_DIR, "stats", "data", "image_normalization.json"), "r")
    )
    regression_norms = json.load(
        open(
            os.path.join(ROOT_DIR, "stats", "data", "regression_labels_stats.json"),
            "r",
        )
    )
    image_norm_mean = image_norms["mean"]
    image_norm_std = image_norms["std"]
    train_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(image_norm_mean, image_norm_std)]
    )

    food_type_vocab, food_type_inverse_vocab = build_vocab(train["food_type"])
    dish_name_vocab, dish_name_inverse_vocab = build_vocab(train["dish_name"])
    cooking_method_vocab, cooking_method_inverse_vocab = build_multilabel_vocab(
        train["cooking_method"]
    )

    train["portion_size"] = train["portion_size"].apply(ast.literal_eval)
    val["portion_size"] = val["portion_size"].apply(ast.literal_eval)
    test["portion_size"] = test["portion_size"].apply(ast.literal_eval)

    portion_ingredient_vocab, portion_ingredient_inverse_vocab = build_multilabel_vocab(
        train["portion_size"].apply(lambda x: list(i[0] for i in x))
    )
    ingredients_vocab, ingredients_inverse_vocab = build_multilabel_vocab(
        train["ingredients"]
    )

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
    ingredient_weights = build_class_weights(
        "ingredient_counter.csv", ingredients_vocab
    )
    portion_ingredient_weights = build_class_weights(
        "weight_ingredient_counter.csv", portion_ingredient_vocab
    )
    food_type_weights = food_type_weights.to(device)
    dish_name_weights = dish_name_weights.to(device)
    ingredient_weights = ingredient_weights.to(device)
    cooking_method_weights = cooking_method_weights.to(device)
    portion_ingredient_weights = portion_ingredient_weights.to(device)

    loss_dictionary = {
        "food_type": nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=food_type_weights
        ),  # single-label
        "ingredients": nn.BCEWithLogitsLoss(
            pos_weight=ingredient_weights
        ),  # multi-label
        "portion_presence": nn.BCEWithLogitsLoss(
            pos_weight=portion_ingredient_weights
        ),  # multi-label
        "cooking_method": nn.BCEWithLogitsLoss(
            pos_weight=cooking_method_weights
        ),  # multi-label
        "dish_name": nn.CrossEntropyLoss(
            label_smoothing=0.05, weight=dish_name_weights
        ),  # single-label
        "portion_weight": nn.HuberLoss(),  # vector regression
        "calories": nn.HuberLoss(),
        "fats": nn.HuberLoss(),
        "carbohydrates": nn.HuberLoss(),
        "proteins": nn.HuberLoss(),
        "binary": nn.BCEWithLogitsLoss(),  # binary classification
    }

    total_loss = LossCombiner(losses=loss_dictionary)

    train_set = DataLoader(
        train_dataset, batch_size=micro_batch_size, collate_fn=collate_fn
    )

    log_path = os.path.join(ROOT_DIR, "stats", "losses.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            headers = (
                ["epoch", "step", "total_loss"]
                + list(total_loss.loss_names)
                + [f"{name}_ema" for name in total_loss.loss_names]
                + [f"{name}_w" for name in total_loss.loss_names]
            )
            f.write(",".join(headers) + "\n")

    adam = torch.optim.Adam(
        list(model.parameters()) + list(total_loss.parameters()), lr=1e-4
    )

    scheduler = CosineAnnealingLR(adam, T_max=total_steps - warmup_steps, eta_min=1e-6)

    model.train()
    accumulation_steps = effective_batch_size // micro_batch_size
    for epoch in range(epochs):
        adam.zero_grad(set_to_none=True)
        pbar = tqdm(
            total=math.ceil(len(train_dataset) / effective_batch_size),
            desc=f"Epoch {epoch+1}/{epochs}",
        )
        accum_counter = 0
        running_loss = 0.0
        running_all_losses = {name: 0.0 for name in total_loss.loss_names}

        ema_score = 0.0
        ema_scores = {name: 0.0 for name in total_loss.loss_names}

        for batch in train_set:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }

            outputs = model(batch["image"])

            loss, loss_dict, kendall_weights = total_loss(outputs, batch)

            running_loss += loss.item()
            for k, v in loss_dict.items():
                running_all_losses[k] += v.item()
            loss = loss / accumulation_steps
            loss.backward()

            accum_counter += 1

            if accum_counter == accumulation_steps:
                adam.step()
                adam.zero_grad(set_to_none=True)

                effective_loss = running_loss / accumulation_steps
                effective_losses = {
                    k: v / accumulation_steps for k, v in running_all_losses.items()
                }

                ema_score = ema_decay * ema_score + (1 - ema_decay) * effective_loss
                ema_scores = {
                    k: v * ema_decay + (1 - ema_decay) * effective_losses[k]
                    for k, v in ema_scores.items()
                }
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
                        *[
                            f"{effective_losses[name]:.4f}"
                            for name in total_loss.loss_names
                        ],
                        *[f"{ema_scores[name]:.4f}" for name in total_loss.loss_names],
                        *[
                            f"{kendall_weights[name]:.4f}"
                            for name in total_loss.loss_names
                        ],
                    ]
                    f.write(",".join(map(str, row)) + "\n")

                accum_counter = 0
                running_loss = 0.0
                running_all_losses = {name: 0.0 for name in total_loss.loss_names}

        if accum_counter > 0:
            adam.step()
            adam.zero_grad(set_to_none=True)

            effective_loss = running_loss / accum_counter

            pbar.update(1)
            pbar.set_postfix(
                {
                    "loss": f"{effective_loss:.4f}",
                }
            )
