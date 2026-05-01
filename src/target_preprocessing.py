"""
File: target_preprocessing.py

Description:
    Resize images with smaller side to 672px while preserving the aspect ratio. Save them into resized_images directory. Also normalize target labels.

Purpose:
    In order to avoid online preprocessing, the data is preprocessed offline and stored locally. Noisy targets are normalized to contain a limited number of labels in a convenient format.

Inputs:
    - Raw images and labels are stored in the /data/images directory.
    - Labels are stored in the /data/labels.parquet file.

Outputs:
    - Preprocessed images are stored in the /data/resized_images directory.
    - Labels are stored in the /data/targets.parquet file.

Dependencies:
    - pandas
    - torch
    - torchvision
    - tqdm
    - PIL

Usage:
    python -m src.target_preprocessing
"""

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Set, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class ImageDataset(Dataset):
    """
    Loads and preprocesses images from disk and saves the resized versions to disk
    """

    def __init__(self, image_dir: str, parquet_path: str, resize: int = 0) -> None:
        """
        Loads the parquet file with the dataset and all the necessary variables.

        Args:
            image_dir (str): Directory where the images are stored.
            parquet_path (str): Path to the parquet file with the dataset.
            resize (int, optional): Size to which the shorter side of the image is resized. Defaults to 0 (no resizing).
        """
        self.image_dir = image_dir
        df = pd.read_parquet(parquet_path)
        self.rows = df.to_dict("records")
        self.image_files = df["image_path"].tolist()
        self.resize = resize

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.rows)

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, Dict[str, Any], int, int, int, str]:
        """
        Loads an image from disk, resizes it, converts to a tensor and returns it with metadata.

        Args:
            idx (int): Index of the dataset sample.

        Returns:
            Tuple[torch.Tensor, Dict[str, Any], int, int, int, str]: Resized image, metadata, index, width, height, filename
        """
        row = self.rows[idx]

        img_path = os.path.join(self.image_dir, self.image_files[idx])

        image = Image.open(img_path).convert("RGB")

        image_width, image_height = image.size

        # Resize according to patches (4x4)
        patch = 4
        if self.resize > 0:
            scale = self.resize / min(image_height, image_width)
            new_h = int(round(image_height * scale / patch) * patch)
            new_w = int(round(image_width * scale / patch) * patch)
            image = transforms.Resize((new_h, new_w))(image)

        filename = os.path.basename(self.image_files[idx])

        save_path = os.path.join(NEW_IMG_DIR, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        image.save(save_path)

        image = transforms.ToTensor()(image)

        # Convert numeric fields explicitly (important for training stability)
        sample = {
            "dish_name": row["dish_name"],
            "food_type": row["food_type"],
            "ingredients": row["ingredients"],
            "portion_size": row["portion_size"],
            "nutritional_profile": row["nutritional_profile"],
            "cooking_method": row["cooking_method"],
            "camera_or_phone_prob": float(row["camera_or_phone_prob"]),
            "food_prob": float(row["food_prob"]),
        }

        return (image, sample, idx, image_height, image_width, filename)


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, Any], int, int, int, str]],
) -> Tuple[
    Tuple[torch.Tensor],
    Tuple[Dict[str, Any]],
    torch.Tensor,
    Tuple[int],
    Tuple[int],
    Tuple[str],
]:
    """
    Custom collate function for batching dataset samples. Images are not stacked into a single tensor due to varying sizes.

    Args:
        batch (list[tuple]):
            List of samples, where each sample is:
            (image, metadata, idx, height, width, image_path)

    Returns:
        tuple:
            - images (tuple[Tensor]): Batch of image tensors
            - samples (tuple[dict]): Batch of metadata dictionaries
            - idxs (Tensor): Sample indices
            - heights (tuple[int]): Original image heights
            - widths (tuple[int]): Original image widths
            - image_paths (tuple[str]): Paths to saved images
    """
    images, samples, idxs, heights, widths, image_paths = zip(*batch)
    return images, samples, torch.tensor(idxs), heights, widths, image_paths


def parse_portion(entry: List[str]) -> List[Tuple[str, float]]:
    """
    Weight of the ingredients is written as a list of strings, each divided by ":". This function parses it.

    Args:
        - entry (List[str]): List of strings, each divided by ":"

    Returns:
        - List[Tuple[str, float]]: List of tuples, where each tuple contains the name of the ingredient and its weight
    """
    if not entry:
        return []
    out = []
    for item in entry:
        try:
            name, val = item.split(":")
            grams = int(val.replace("g", ""))
            out.append((name, grams))
        except Exception:
            continue  # skip malformed entries safely
    return out


def normalize_methods(text: str) -> List[Set[str]]:
    """
    Normalization of the cooking methods function. The unnormalized methods are normalized and split if several are listed in one sample.

    Args:
        - text (str): Unnormalized cooking methods

    Returns:
        - List[Set[str]]: List of normalized cooking methods
    """
    # case 1: NaN
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ["unknown"]

    methods = parse_methods(text)

    out = []

    # normalize, very rare or special methods are marked as "other"
    for m in methods:
        if m in CANON:
            out.append(CANON[m])
        elif m.strip():
            out.append("other")

    # case 2: empty after parsing
    if len(out) == 0:
        return ["missing"]

    return list(set(out))


def parse_methods(text: str) -> List[str]:
    """
    Parsing of the cooking methods. The string of methods is split into individual methods.

    Args:
        - text (str): Unnormalized cooking methods

    Returns:
        - List[str]: List of normalized cooking methods
    """
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = text.replace(".", "")

    # normalize separators
    text = re.sub(r"[/,]| and | or ", "|", text)

    parts = [p.strip() for p in text.split("|")]

    return [p for p in parts if p]


def normalize_ingredients(text: str, counter: Counter, threshold: int) -> str:
    """
    Normalization of ingredients function. If the ingredient is rare, it is marked as "other".

    Args:
        - text (str): Raw ingredients
        - counter (Counter): Counter of ingredients
        - threshold (int): Threshold for rare ingredients

    Returns:
        - Str: Normalized ingredients
    """
    text = text.lower()
    if counter[text] <= threshold:
        return "other"
    return text


def safe_parse(x: List[str] | str) -> List[str]:
    """
    Ensure that the output is a list.

    Args:
        - x (List[str] | str): Ingredients

    Returns:
        - List[str]: List of ingredients
    """
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return json.loads(x)
    return []


def normalize_name(s: str) -> str:
    return s.lower().replace(" ", "_")


def normalize_portion_size(
    portion: List[Tuple[str, float]], counter: Counter, threshold: int
) -> List[Tuple[str, float]]:
    """
    Args:
        portion: list[tuple(str, number)] OR list[list[str, number]]
        counter: frequency of ingredient names (precomputed on normalized names)
        threshold: rare cutoff

    Returns:
        list[(normalized_name, weight)]
    """
    if not isinstance(portion, list):
        return []

    out = []
    for item in portion:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        name, weight = item
        name = normalize_name(str(name))

        if counter[name] <= threshold:
            name = "other"

        out.append((name, float(weight)))

    return out


if __name__ == "__main__":
    CANON = {
        # --- BOIL FAMILY ---
        "boiling": "boil",
        "boiled": "boil",
        "simmered": "boil",
        "simmering": "boil",
        "boil": "boil",
        # --- FRY FAMILY ---
        "frying": "fry",
        "fried": "fry",
        "deep frying": "fry",
        "deep-frying": "fry",
        "pan-fried": "fry",
        "pan-frying": "fry",
        "air frying": "fry",
        # --- STIR FRY ---
        "stir-frying": "stir_fry",
        "stir-fried": "stir_fry",
        "stir fry": "stir_fry",
        "stir-fry": "stir_fry",
        # --- GRILL ---
        "grilling": "grill",
        "grilled": "grill",
        "grill": "grill",
        "pan-searing": "grill",
        # --- STEAM ---
        "steamed": "steam",
        "steaming": "steam",
        "steam": "steam",
        "lightly steamed": "steam",
        # --- BAKE ---
        "baked": "bake",
        "baking": "bake",
        # --- ROAST ---
        "roasting": "roast",
        "roasted": "roast",
        # --- BRAISE / STEW ---
        "braising": "braise",
        "braised": "braise",
        "stewing": "stew",
        "stewed": "stew",
        # --- SAUTE ---
        "sautéed": "saute",
        "sautéing": "saute",
        "sauteed": "saute",
        # --- RAW / NO COOK ---
        "raw": "raw",
        "uncooked": "raw",
        "raw preparation": "raw",
        "no cooking involved": "raw",
        "no cooking required": "raw",
        # --- ASSEMBLY ---
        "assembled": "assemble",
        "assembled directly": "assemble",
        "assembled as is": "assemble",
        "prepared": "assemble",
        "mixed": "assemble",
        "served": "assemble",
        "rolled": "assemble",
        # --- PRESERVATION ---
        "chilled": "preserve",
        "frozen": "preserve",
        "dried": "preserve",
        "dehydration": "preserve",
        "packaged": "preserve",
        # --- OTHER / NOISE ---
        "seasoned": "other",
        "sauced": "other",
        "tossed in sauce": "other",
        "cooked": "other",
        "lightly cooked": "other",
    }

    # -----------------------
    # CONFIGURATION
    # -----------------------

    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    OLD_IMG_DIR = os.path.join(DATA_DIR, "images")
    NEW_IMG_DIR = os.path.join(DATA_DIR, "resized_images")
    tqdm.pandas(dynamic_ncols=True)

    # -----------------------
    # LOAD DATA
    # -----------------------

    # Resize chosen to match transformer spatial design (patching + window stages) and reduce padding artifacts
    dataset = ImageDataset(
        image_dir=OLD_IMG_DIR,
        parquet_path=os.path.join(DATA_DIR, "labels.parquet"),
        resize=448,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    # 8 images have never successfully materialized
    max_i = dataset.__len__()
    done = False

    # -----------------------
    # Data Stats
    # -----------------------

    image_stats = {"height": [], "width": [], "size": [], "aspect_ratio": []}
    image_files = []
    target_stats = {
        "dish_name": [],
        "food_type": [],
        "ingredients": [],
        "portion_size": [],
        "nutritional_profile": {
            "fat_g": [],
            "protein_g": [],
            "calories_kcal": [],
            "carbohydrate_g": [],
        },
        "cooking_method": [],
        "camera_or_phone_prob": [],
        "food_prob": [],
    }
    for images, targets, idxs, original_height, original_width, image_ids in tqdm(
        loader, total=len(loader), dynamic_ncols=True
    ):
        for image, target, idx, h, w, image_id in zip(
            images, targets, idxs, original_height, original_width, image_ids
        ):
            image_files.append(image_id)
            image_stats["height"].append(h)
            image_stats["width"].append(w)
            image_stats["size"].append(h * w)
            image_stats["aspect_ratio"].append(w / h)

            nut = json.loads(target["nutritional_profile"])

            target_stats["dish_name"].append(target["dish_name"])
            target_stats["food_type"].append(target["food_type"])
            target_stats["ingredients"].append(target["ingredients"])

            # portion size is a list in a string, requires parsing
            target_stats["portion_size"].append(
                [parse_portion(json.loads(target["portion_size"]))]
            )
            target_stats["cooking_method"].append(target["cooking_method"])
            target_stats["camera_or_phone_prob"].append(target["camera_or_phone_prob"])
            target_stats["food_prob"].append(target["food_prob"])

            target_stats["nutritional_profile"]["fat_g"].append(nut["fat_g"])
            target_stats["nutritional_profile"]["protein_g"].append(nut["protein_g"])
            target_stats["nutritional_profile"]["calories_kcal"].append(
                nut["calories_kcal"]
            )
            target_stats["nutritional_profile"]["carbohydrate_g"].append(
                nut["carbohydrate_g"]
            )
            if idx == max_i:
                done = True
                break
        if done:
            break
    image_describe = pd.DataFrame(image_stats).describe().T
    full_data = pd.DataFrame(
        {
            "image_path": image_files,
            "food_type": target_stats["food_type"],
            "dish_name": target_stats["dish_name"],
            "cooking_method": target_stats["cooking_method"],
            "portion_size": [x[0] for x in target_stats["portion_size"]],
            "ingredients": target_stats["ingredients"],
            "fat_g": target_stats["nutritional_profile"]["fat_g"],
            "protein_g": target_stats["nutritional_profile"]["protein_g"],
            "carbohydrate_g": target_stats["nutritional_profile"]["carbohydrate_g"],
            "calories_kcal": target_stats["nutritional_profile"]["calories_kcal"],
            "camera_or_phone_prob": target_stats["camera_or_phone_prob"],
            "food_prob": target_stats["food_prob"],
        }
    )

    # -----------------------
    # Data Normalization
    # -----------------------

    # many cooking methods are non-canonical strings, normalize them
    full_data["cooking_method"] = full_data["cooking_method"].progress_apply(
        normalize_methods
    )

    ingredients = Counter()
    for idx, row in full_data.iterrows():
        for i in json.loads(row["ingredients"]):
            ingredients[i.lower()] += 1
    full_data["ingredients"] = full_data["ingredients"].apply(safe_parse)

    # threshold limits the number of unique ingredients
    full_data["ingredients"] = full_data["ingredients"].progress_apply(
        lambda lst: [
            normalize_ingredients(item, counter=ingredients, threshold=24)
            for item in lst
        ]
    )

    # threshold limits the number of unique dish names
    dish_name = Counter()
    for idx, row in full_data.iterrows():
        dish_name[str(row["dish_name"]).lower()] += 1
    full_data["dish_name"] = full_data["dish_name"].progress_apply(
        lambda line: normalize_ingredients(str(line), counter=dish_name, threshold=24)
    )
    full_data["portion_size"] = full_data["portion_size"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    counter = Counter()

    for row in full_data["portion_size"]:
        if isinstance(row, list):
            for name, _ in row:
                counter[normalize_name(str(name))] += 1
    full_data["portion_size"] = full_data["portion_size"].apply(
        lambda x: normalize_portion_size(x, counter, threshold=24)
    )

    # -----------------------
    # Save to disk
    # -----------------------

    full_data["portion_size"] = full_data["portion_size"].apply(json.dumps)
    full_data.to_parquet(
        os.path.join(ROOT_DIR, "data", "labels_processed.parquet"), index=False
    )
    image_describe.to_csv(
        os.path.join(ROOT_DIR, "stats", "data", "image_size_stats.csv"), index=True
    )
    print("Done!")
