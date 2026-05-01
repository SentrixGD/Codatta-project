"""
File: data_analysis.py

Description:
    Gather various statistics about the images and metadata. Creates both tables and visualizations.

Purpose:
    Gathter the data to analyze the dataset and visualize the distribution of labels.

Inputs:
    - Resized images and normalized labels.

Outputs:
    - Tables and visualizations in the /stats/ directory.

Dependencies:
    - matplotlib
    - numpy
    - pandas
    - PIL
    - seaborn
    - torch
    - tqdm
    - torchvision

Usage:
    python -m src.data_analysis
"""

import json
import os
from collections import Counter
from itertools import chain
from typing import Any, Dict, List, Tuple

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    def __init__(self, image_dir: str, parquet_path: str) -> None:
        """
        Loads the parquet file with the dataset and all the necessary variables.

        Args:
            image_dir (str): Directory where the images are stored.
            parquet_path (str): Path to the parquet file with the dataset.
        """
        self.image_dir = image_dir
        df = pd.read_parquet(parquet_path)
        self.rows = df.to_dict("records")
        self.image_files = df["image_path"].tolist()

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any], int]:
        """
        Loads an image from disk, converts to a tensor and returns it with metadata.

        Args:
            idx (int): Index of the dataset sample.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any], int]: Image, metadata, index
        """
        row = self.rows[idx]

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image = transforms.ToTensor()(image)

        # Convert numeric fields explicitly (important for training stability)
        sample = {
            "dish_name": row["dish_name"],
            "food_type": row["food_type"],
            "ingredients": row["ingredients"],
            "portion_size": row["portion_size"],
            "fat_g": row["fat_g"],
            "protein_g": row["protein_g"],
            "carbohydrate_g": row["carbohydrate_g"],
            "calories_kcal": row["calories_kcal"],
            "cooking_method": row["cooking_method"],
            "camera_or_phone_prob": float(row["camera_or_phone_prob"]),
            "food_prob": float(row["food_prob"]),
        }

        return image, sample, idx


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any], int]],
) -> Tuple[Tuple[torch.Tensor], Tuple[Dict[str, Any]], torch.Tensor]:
    """
    Custom collate function for batching dataset samples. Images are not stacked into a single tensor due to varying sizes.

    Args:
        batch (list[tuple]):
            List of samples, where each sample is:
            (image, metadata, idx)

    Returns:
        tuple:
            - images (tuple[Tensor]): Batch of image tensors
            - samples (tuple[dict]): Batch of metadata dictionaries
            - idxs (Tensor): Sample indices
    """
    images, samples, idxs = zip(*batch)
    return images, samples, torch.tensor(idxs)


def flatten(series: List[List[Any]]) -> List[Any]:
    """
    Flattens a list of lists into a single list.

    Args:
        series (list[list]): List of lists to flatten.

    Returns:
        list: Flattened list.
    """
    return list(chain.from_iterable(series))


if __name__ == "__main__":

    # ----------------------------
    # CONFIGURATION
    # ----------------------------

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    dataset = ImageDataset(
        image_dir=os.path.join(DATA_DIR, "resized_images"),
        parquet_path=os.path.join(DATA_DIR, "labels_processed.parquet"),
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    max_i = dataset.__len__()
    done = False
    image_stats = {
        "red_mean": [],
        "green_mean": [],
        "blue_mean": [],
        "red_std": [],
        "green_std": [],
        "blue_std": [],
        "entropy": [],
        "brightness": [],
        "sharpness": [],
        "contrast": [],
        "luminance": [],  # 0.299 * R + 0.587 * G + 0.114 * B
        "saturation": [],
    }
    target_stats = {
        "dish_name": [],
        "food_type": [],
        "ingredients": [],
        "portion_size": [],
        "fat_g": [],
        "protein_g": [],
        "calories_kcal": [],
        "carbohydrate_g": [],
        "cooking_method": [],
        "camera_or_phone_prob": [],
        "food_prob": [],
    }
    kernel = (
        torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # ----------------------------
    # ANALYSIS
    # ----------------------------

    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    n_pixels = 0
    for images, targets, idxs in tqdm(loader, total=len(loader), dynamic_ncols=True):
        for image, target, idx in zip(images, targets, idxs):

            # ----------------------------
            # IMAGE ANALYSIS
            # ----------------------------

            r, g, b = image[0], image[1], image[2]
            r_mean = r.mean().item()
            g_mean = g.mean().item()
            b_mean = b.mean().item()
            image_stats["red_mean"].append(r_mean)
            image_stats["green_mean"].append(g_mean)
            image_stats["blue_mean"].append(b_mean)
            r_std = r.std().item()
            g_std = g.std().item()
            b_std = b.std().item()
            image_stats["red_std"].append(r_std)
            image_stats["green_std"].append(g_std)
            image_stats["blue_std"].append(b_std)
            pixels = image.view(3, -1)  # shape: (C, H*W)

            sum_ += pixels.sum(dim=1)
            sum_sq += (pixels**2).sum(dim=1)
            n_pixels += pixels.shape[1]

            gray = image.mean(dim=0)
            hist = torch.histc(gray, bins=256, min=0.0, max=1.0)
            p = hist / hist.sum()
            entropy = -(p * torch.log2(p + 1e-12)).sum()
            image_stats["entropy"].append(entropy.item())
            image_stats["brightness"].append(gray.mean().item())
            image_stats["contrast"].append(gray.std().item())
            gray_unsqueezed = gray.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

            lap = F.conv2d(gray_unsqueezed, kernel, padding=1)

            image_stats["sharpness"].append(lap.var().item())
            image_stats["luminance"].append(
                0.299 * r_mean + 0.587 * g_mean + 0.114 * b_mean
            )
            maxc = torch.maximum(torch.maximum(r, g), b)
            minc = torch.minimum(torch.minimum(r, g), b)
            sat = (maxc - minc) / (maxc + 1e-8)
            image_stats["saturation"].append(sat.mean().item())

            # ----------------------------
            # LABELS ANALYSIS
            # ----------------------------

            target_stats["dish_name"].append(target["dish_name"])
            target_stats["food_type"].append(target["food_type"])
            target_stats["ingredients"].append(target["ingredients"])
            target_stats["portion_size"].append(target["portion_size"])
            target_stats["cooking_method"].append(target["cooking_method"])
            target_stats["camera_or_phone_prob"].append(target["camera_or_phone_prob"])
            target_stats["food_prob"].append(target["food_prob"])

            target_stats["fat_g"].append(target["fat_g"])
            target_stats["protein_g"].append(target["protein_g"])
            target_stats["carbohydrate_g"].append(target["carbohydrate_g"])
            target_stats["calories_kcal"].append(target["calories_kcal"])
            if idx == max_i:
                done = True
                break
        if done:
            break
    true_mean = sum_ / n_pixels
    true_std = torch.sqrt(sum_sq / n_pixels - true_mean**2)

    save_path = os.path.join(ROOT_DIR, "stats", "data", "image_normalization.json")

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = {"mean": true_mean.tolist(), "std": true_std.tolist()}

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    df_img = pd.DataFrame(image_stats)
    image_describe = df_img.describe().T
    image_correlation = df_img.corr()

    target_data = pd.DataFrame(
        {
            "food_type": target_stats["food_type"],
            "dish_name": target_stats["dish_name"],
            "cooking_method": target_stats["cooking_method"],
            "portion_size": target_stats["portion_size"],
            "ingredients": target_stats["ingredients"],
            "fat_g": target_stats["fat_g"],
            "protein_g": target_stats["protein_g"],
            "carbohydrate_g": target_stats["carbohydrate_g"],
            "calories_kcal": target_stats["calories_kcal"],
            "camera_or_phone_prob": target_stats["camera_or_phone_prob"],
            "food_prob": target_stats["food_prob"],
        }
    )

    # ----------------------------
    # ADVANCED FEATURE ANALYSIS
    # ----------------------------

    target_data["portion_size"] = target_data["portion_size"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    weights = target_data["portion_size"].tolist()
    meal_total_weight = []
    meal_weight_max_ratio = []
    num_weight_ingredients = []
    meal_weight_std_ratio = []
    normalized_meal_weight_entropy = []
    ingredient_counter = Counter()
    for meal in weights:
        total = sum(g for _, g in meal)
        for ingredient, _ in meal:
            ingredient_counter[ingredient] += 1
        meal_total_weight.append(total)
        num_weight_ingredients.append(len(meal))

        if total > 0:
            ratios = np.array([g / total for _, g in meal], dtype=np.float32)
            ent = -(ratios * np.log(ratios + 1e-8)).sum()
            max_ent = np.log(len(ratios)) if len(ratios) > 1 else 1.0
            normalized_meal_weight_entropy.append(ent / max_ent)
            meal_weight_max_ratio.append(ratios.max())
            meal_weight_std_ratio.append(ratios.std())
        else:
            meal_weight_max_ratio.append(0.0)
            meal_weight_std_ratio.append(0.0)
            normalized_meal_weight_entropy.append(0.0)

    target_data["meal_weight_entropy"] = normalized_meal_weight_entropy
    target_data["meal_total_weight"] = meal_total_weight
    target_data["meal_weight_max_ratio"] = meal_weight_max_ratio
    target_data["num_weight_ingredients"] = num_weight_ingredients
    target_data["meal_weight_std_ratio"] = meal_weight_std_ratio
    target_data["num_ingredients"] = target_data["ingredients"].apply(lambda x: len(x))
    portion_ingredients = pd.DataFrame(
        ingredient_counter.items(), columns=["ingredient", "count"]
    ).sort_values("count", ascending=False)

    # ----------------------------
    # SAVE CSV STATS
    # ----------------------------

    os.makedirs(os.path.join(ROOT_DIR, "stats", "data"), exist_ok=True)
    target_describe = target_data.describe().T
    image_describe.to_csv(os.path.join(ROOT_DIR, "stats", "data", "image_describe.csv"))
    target_describe.to_csv(
        os.path.join(ROOT_DIR, "stats", "data", "target_describe.csv")
    )
    image_correlation.to_csv(
        os.path.join(ROOT_DIR, "stats", "data", "image_correlation.csv")
    )

    # ----------------------------
    # PLOTS
    # ----------------------------

    corr = image_correlation

    features = pd.concat(
        [
            target_data.reset_index(drop=True)[
                [
                    "fat_g",
                    "protein_g",
                    "carbohydrate_g",
                    "calories_kcal",
                    "camera_or_phone_prob",
                    "food_prob",
                    "meal_total_weight",
                    "num_weight_ingredients",
                    "num_ingredients",
                    "meal_weight_entropy",
                ]
            ],
            pd.DataFrame(image_stats).reset_index(drop=True)[
                ["entropy", "luminance", "brightness", "contrast", "sharpness"]
            ],
        ],
        axis=1,
    )

    colors = sns.color_palette("husl", len(features.columns))

    # ----------------------------
    # INDIVIDUAL FEATURE PLOTS
    # ----------------------------

    for i, feat in enumerate(features.columns):
        data = features[feat].dropna()
        n_unique = data.nunique()

        plt.figure(figsize=(6, 4))

        # ----------------------------
        # CASE 1: low-cardinality -> bar
        # ----------------------------

        if n_unique <= 15:
            counts = data.value_counts().sort_index()

            plt.bar(
                counts.index.astype(str),
                counts.values,
                color=colors[i],
                alpha=0.95,
            )

            plt.xticks(rotation=45)
            plt.xlabel(feat)
            plt.ylabel("count")

        # ----------------------------
        # CASE 2: continuous -> histogram
        # ----------------------------

        else:
            bins = min(50, int(np.sqrt(len(data))))  # adaptive bins

            plt.hist(
                data,
                bins=bins,
                color=colors[i],
                alpha=0.95,
            )

            plt.xlabel(feat)
            plt.ylabel("count")

        plt.title(f"{feat} (unique={n_unique})")
        plt.grid(True, alpha=0.2)

        plt.savefig(
            os.path.join(ROOT_DIR, "stats", "data", f"{feat}_hist.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # ----------------------------
    # COMPILATION OF PLOTS
    # ----------------------------

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    labels = [f"({chr(97 + i)})" for i in range(len(features.columns))]

    for i, feat in enumerate(features.columns):
        ax = axes[i]

        data = features[feat].dropna()
        n_unique = data.nunique()

        # ----------------------------
        # DISCRETE FEATURES → BAR PLOT
        # ----------------------------
        if n_unique <= 15:
            counts = data.value_counts().sort_index()

            ax.bar(
                counts.index.astype(str),
                counts.values,
                color=colors[i],
                alpha=0.95,
            )
            ax.tick_params(axis="x", rotation=45)

        # ----------------------------
        # CONTINUOUS FEATURES → HISTOGRAM
        # ----------------------------
        else:
            bins = min(50, int(np.sqrt(len(data))))

            ax.hist(
                data,
                bins=bins,
                color=colors[i],
                alpha=0.95,
            )

        # ----------------------------
        # COMMON STYLING
        # ----------------------------
        ax.set_title(feat)

        ax.text(
            0.02,
            0.95,
            labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
        )

        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        os.path.join(ROOT_DIR, "stats", "data", "data_hist_compilation.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7, 5))

    # ----------------------------
    # BRIGHTNESS VS LUMINANCE PLOT
    # ----------------------------

    bins = 50

    plt.hist(
        features["brightness"],
        bins=bins,
        alpha=0.5,
        label="brightness",
        density=True,
        color="red",
    )

    plt.hist(
        features["luminance"],
        bins=bins,
        alpha=0.5,
        label="luminance",
        density=True,
        color="blue",
    )

    plt.title("Brightness vs Luminance Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.savefig(
        os.path.join(ROOT_DIR, "stats", "data", "brightness_vs_luminance.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(8, 4))

    # ----------------------------
    # FOOD TYPE AND COOKING METHOD PLOTS
    # ----------------------------

    food_counts = target_data["food_type"].value_counts()
    food_type_variety = pd.Series(food_counts).nunique()

    sns.barplot(x=food_counts.index, y=food_counts.values, palette="tab10")

    plt.title("Food Type Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.2)

    plt.savefig(
        os.path.join(ROOT_DIR, "stats", "data", "food_type_dist.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plt.figure(figsize=(8, 4))

    methods = flatten(target_data["cooking_method"])
    method_counts = pd.Series(methods).value_counts()
    cooking_method_variety = pd.Series(method_counts).nunique()

    sns.barplot(x=method_counts.index, y=method_counts.values, palette="tab10")

    plt.title("Cooking Method Distribution")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.xlabel("Cooking Method")
    plt.grid(True, alpha=0.2)

    plt.savefig(
        os.path.join(ROOT_DIR, "stats", "data", "cooking_method_dist.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    TOP_K = 30

    # ----------------------------
    # DISH NAMES
    # ----------------------------

    plt.figure(figsize=(10, 5))

    dish_counts = target_data["dish_name"].value_counts()
    dish_name_variety = pd.Series(dish_counts).nunique()
    dish_counts = dish_counts[dish_counts.index != "other"].head(TOP_K)

    values = dish_counts.values

    # normalize values → [0,1]
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap("viridis")

    colors = [cmap(norm(v)) for v in values]

    sns.barplot(x=values, y=dish_counts.index, palette=colors)

    plt.title(f"Top {TOP_K} Dish Names")
    plt.xlabel("Count")
    plt.ylabel("Dish Name")
    plt.grid(True, alpha=0.2)

    plt.savefig(
        os.path.join(ROOT_DIR, "stats", "data", "dish_name_topk.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ----------------------------
    # INGREDIENTS PLOT
    # ----------------------------
    plt.figure(figsize=(10, 6))

    ingredients = flatten(target_data["ingredients"])
    ingredient_variety = pd.Series(ingredients).nunique()
    ingredient_counts = pd.Series(ingredients).value_counts().head(TOP_K)

    values = ingredient_counts.values

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.get_cmap("magma")

    colors = [cmap(norm(v)) for v in values]

    sns.barplot(x=values, y=ingredient_counts.index, palette=colors)

    plt.title(f"Top {TOP_K} Ingredients")
    plt.xlabel("Count")
    plt.ylabel("Ingredient")
    plt.grid(True, alpha=0.2)

    plt.savefig(
        os.path.join(ROOT_DIR, "stats", "data", "ingredients_topk.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    all_weighed_names = [name for row in target_data["portion_size"] for name, _ in row]
    unique_weighed_count = len(set(all_weighed_names))

    with open(
        os.path.join(ROOT_DIR, "stats", "data", "label_diversity_data.json"), "w"
    ) as f:
        json.dump(
            {
                "ingredients": ingredient_variety,
                "dish_names": dish_name_variety,
                "weighed_ingredients": unique_weighed_count,
                "food_types": food_type_variety,
                "cooking_methods": cooking_method_variety,
            },
            f,
        )

    print("Done!")
