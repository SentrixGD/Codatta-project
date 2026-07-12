"""
File: dataset.py

Description:
    Dataset class, encoding functions, vocabulary builders, collate function,
    and class weight computation for food image training.

Purpose:
    Encapsulate all data loading and encoding logic so the training loop
    only deals with tensors.

Usage:
    from src.training.dataset import FoodDataset, collate_fn, build_vocab
"""

import os
from io import BytesIO

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import DATA_DIR, ROOT_DIR

# ──────────────────────────────────────────────
# Transforms (set by entry point before dataset creation)
# ──────────────────────────────────────────────

train_transform: transforms.Compose | None = None


def set_transform(transform: transforms.Compose) -> None:
    """Set the global transform used by load_image."""
    global train_transform  # noqa: PLW0603
    train_transform = transform


# ──────────────────────────────────────────────
# Image loading
# ──────────────────────────────────────────────


def load_image(image_id: str, image_dir: str = "resized_images") -> torch.Tensor:
    """Load an image from disk and apply the global transform."""
    path = os.path.join(DATA_DIR, image_dir, image_id)
    image = Image.open(path).convert("RGB")
    return train_transform(image)


def _load_image_bytes(image_id: str) -> bytes:
    """Load raw JPEG bytes from disk."""
    path = os.path.join(DATA_DIR, "resized_images", image_id)
    with open(path, "rb") as f:
        return f.read()


def _decode_image_bytes(raw: bytes) -> torch.Tensor:
    """Decode JPEG bytes and apply the global transform."""
    image = Image.open(BytesIO(raw)).convert("RGB")
    return train_transform(image)


# ──────────────────────────────────────────────
# Vocabulary builders
# ──────────────────────────────────────────────


def build_vocab(values: list) -> tuple[dict, dict]:
    """Build a string-to-int and int-to-string vocabulary from a list of values."""
    unique = sorted(set(values))
    stoi = {v: i for i, v in enumerate(unique)}
    itos = {i: v for v, i in stoi.items()}
    return stoi, itos


def build_multilabel_vocab(column: list) -> tuple[dict, dict]:
    """Build a vocabulary from a list of multi-label lists (flattened)."""
    unique: set = set()
    for labels in column:
        unique.update(labels)
    unique = sorted(unique)
    stoi = {v: i for i, v in enumerate(unique)}
    itos = {i: v for v, i in stoi.items()}
    return stoi, itos


# ──────────────────────────────────────────────
# Encoding functions
# ──────────────────────────────────────────────


def encode_single(label: str, vocab: dict[str, int]) -> torch.Tensor:
    """Encode a single label into a scalar tensor."""
    return torch.tensor(vocab[label], dtype=torch.long)


def encode_multilabel(labels: list, vocab: dict[str, int]) -> torch.Tensor:
    """Encode a list of labels into a multi-hot binary tensor."""
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for label in labels:
        if label in vocab:
            vec[vocab[label]] = 1.0
    return vec


def encode_portion(portion_list: list, vocab: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of (name, weight) tuples into presence and weight tensors."""
    presence = torch.zeros(len(vocab), dtype=torch.float32)
    weight = torch.zeros(len(vocab), dtype=torch.float32)
    for name, w in portion_list:
        if name in vocab:
            idx = vocab[name]
            presence[idx] = 1.0
            weight[idx] = float(w)
    return presence, weight


# ──────────────────────────────────────────────
# Regression normalization helper
# ──────────────────────────────────────────────


def normalize_regression(value: float, norm_params: dict[str, float]) -> torch.Tensor:
    """Apply log1p + z-score normalization to a regression target."""
    return torch.tensor(
        (np.log1p(value) - norm_params["log_mean"]) / norm_params["log_std"],
        dtype=torch.float32,
    )


# ──────────────────────────────────────────────
# Class weights
# ──────────────────────────────────────────────


def build_class_weights(
    file_name: str,
    vocab: dict[str, int],
    ignore_class: str | None = None,
) -> torch.Tensor:
    """Build inverse-frequency class weights with sqrt dampening."""
    df = pd.read_csv(os.path.join(ROOT_DIR, "stats", "data", file_name))
    key_col = df.columns[0]
    val_col = df.columns[1]

    counts = df[val_col].values.astype(np.float32)
    weights = 1 / np.sqrt(counts)
    weights = weights / weights.mean()

    df["weight"] = weights
    if ignore_class is not None:
        df.loc[df[key_col] == ignore_class, "weight"] = 0.0

    weight_tensor = torch.ones(len(vocab))
    for _, row in df.iterrows():
        cls = row[key_col]
        if cls in vocab:
            weight_tensor[vocab[cls]] = row["weight"]
    return weight_tensor


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────


class FoodDataset(Dataset):
    """Dataset for food image classification with multi-task targets."""

    def __init__(
        self,
        df: pd.DataFrame,
        food_type_vocab: dict[str, int],
        dish_name_vocab: dict[str, int],
        cooking_method_vocab: dict[str, int],
        ingredients_vocab: dict[str, int],
        portion_ingredients_vocab: dict[str, int],
        regression_normalization: dict[str, dict[str, float]],
        cache_images: bool = False,
        image_dir: str = "resized_images",
    ):
        df = df.reset_index(drop=True)
        self.food_type_vocab = food_type_vocab
        self.dish_name_vocab = dish_name_vocab
        self.cooking_method_vocab = cooking_method_vocab
        self.ingredients_vocab = ingredients_vocab
        self.portion_ingredients_vocab = portion_ingredients_vocab
        self.regression_normalization = regression_normalization
        self.image_dir = image_dir

        self.image_paths = df["image_path"].tolist()
        self._len = len(df)

        self._image_cache: list[bytes] | None = None
        if cache_images:
            self._cache_all_images()

        # Pre-compute all label tensors
        self.food_types = [encode_single(v, food_type_vocab) for v in df["food_type"]]
        self.dish_names = [encode_single(v, dish_name_vocab) for v in df["dish_name"]]
        self.cooking_methods = [encode_multilabel(v, cooking_method_vocab) for v in df["cooking_method"]]
        self.ingredients_list = [encode_multilabel(v, ingredients_vocab) for v in df["ingredients"]]

        portion_norm = regression_normalization["portion_size"]
        self.portion_presences = []
        self.portion_weights = []
        for portion_list in df["portion_size"]:
            presence, weight = encode_portion(portion_list, portion_ingredients_vocab)
            weight = weight.to(torch.float32)
            mask = weight > 0
            pw = torch.zeros_like(weight)
            if mask.any():
                log_weight = np.log1p(weight[mask].numpy())
                pw[mask] = torch.tensor(
                    (log_weight - portion_norm["log_mean"]) / portion_norm["log_std"],
                    dtype=torch.float32,
                )
            self.portion_presences.append(presence)
            self.portion_weights.append(pw)

        self.fat_g = [normalize_regression(v, regression_normalization["fat_g"]) for v in df["fat_g"]]
        self.protein_g = [normalize_regression(v, regression_normalization["protein_g"]) for v in df["protein_g"]]
        self.carbohydrate_g = [normalize_regression(v, regression_normalization["carbohydrate_g"]) for v in df["carbohydrate_g"]]
        self.calories_kcal = [normalize_regression(v, regression_normalization["calories_kcal"]) for v in df["calories_kcal"]]
        self.camera_or_phone_prob = [torch.tensor(v, dtype=torch.float32) for v in df["camera_or_phone_prob"]]
        self.food_prob = [torch.tensor(v, dtype=torch.float32) for v in df["food_prob"]]

    def _cache_all_images(self) -> None:
        """Load all image bytes into memory to avoid disk I/O during training."""
        print(f"Caching {self._len} images in memory...")
        self._image_cache = []
        for image_id in self.image_paths:
            self._image_cache.append(_load_image_bytes(image_id))
        print(f"Cached {len(self._image_cache)} images ({sum(len(b) for b in self._image_cache) / 1e6:.0f} MB)")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._image_cache is not None:
            image = _decode_image_bytes(self._image_cache[idx])
        else:
            image = load_image(self.image_paths[idx], self.image_dir)

        return {
            "image": image,
            "food_type": self.food_types[idx],
            "dish_name": self.dish_names[idx],
            "cooking_method": self.cooking_methods[idx],
            "ingredients": self.ingredients_list[idx],
            "portion_presence": self.portion_presences[idx],
            "portion_weight": self.portion_weights[idx],
            "fat_g": self.fat_g[idx],
            "protein_g": self.protein_g[idx],
            "carbohydrate_g": self.carbohydrate_g[idx],
            "calories_kcal": self.calories_kcal[idx],
            "camera_or_phone_prob": self.camera_or_phone_prob[idx],
            "food_prob": self.food_prob[idx],
        }


# ──────────────────────────────────────────────
# Collate
# ──────────────────────────────────────────────


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-size images with padding; stack all other tensors."""
    images = [item["image"] for item in batch]
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        c, h, w = img.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded = F.pad(img, (0, pad_w, 0, pad_h))
        padded_images.append(padded)

    images_tensor = torch.stack(padded_images)

    collated: dict[str, torch.Tensor] = {}
    for key in batch[0].keys():
        if key == "image":
            collated[key] = images_tensor
        else:
            collated[key] = torch.stack([item[key] for item in batch])

    return collated


# ──────────────────────────────────────────────
# Batch device move helper
# ──────────────────────────────────────────────


def move_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch to the given device."""
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
