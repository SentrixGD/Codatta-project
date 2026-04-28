"""
File: materialization.py

Description:
    Download all the data from the cloud storage. If run again, will try to download all missing data. The data is stored in the data directory.

Purpose:
    Despite its size, it is faster to use the local offline-preprocessed data rather than stream it from the cloud and preprocess it online. Such possibility to store the dataset locally is present, therefore we do it.

Inputs:
    - No inputs needed, the path to the data is defined in the code.

Outputs:
    - Raw data is stored as jpeg images in the /data/images/ directory.
    - Metadata is stored in the /data/labels.parquet file.

Dependencies:
    - pandas
    - requests
    - PIL
    - datasets

Usage:
    python -m src.materialization
"""

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import requests
from datasets import load_dataset
from PIL import Image


# -----------------------
# DOWNLOAD
# -----------------------
def download_and_save(i: int, item: dict) -> dict | None:
    """
    Downloads an image from a URL and saves it locally with retry logic.
    If the image already exists, returns cached metadata without re-downloading.

    This function is designed for parallel dataset materialization with resume support.

    Args:
        i (int): Index of the dataset sample, used for filename generation.
        item (dict): Dataset entry containing image URL and metadata fields:
            - image_url (str)
            - dish_name (str)
            - food_type (str)
            - ingredients (list)
            - portion_size (str | list)
            - nutritional_profile (dict)
            - cooking_method (any)
            - camera_or_phone_prob (float)
            - food_prob (float)
            - sub_dt (any)

    Returns:
        dict | None:
            Dictionary containing saved image path and metadata if successful,
            otherwise None if all download attempts fail.
    """

    img_name = f"{i:06d}.jpg"
    url = item["image_url"]

    # resume behavior: skip download if already cached locally
    if os.path.exists(os.path.join(IMG_DIR, img_name)):
        return {
            "image_path": img_name,
            "dish_name": item["dish_name"],
            "food_type": item["food_type"],
            "ingredients": item["ingredients"],
            "portion_size": item["portion_size"],
            "nutritional_profile": item["nutritional_profile"],
            "cooking_method": item["cooking_method"],
            "camera_or_phone_prob": item["camera_or_phone_prob"],
            "food_prob": item["food_prob"],
            "sub_dt": item["sub_dt"],
        }

    # retry loop improves robustness against network instability
    for _ in range(10):
        try:
            r = requests.get(url, timeout=TIMEOUT)

            # retry on non-success HTTP responses (e.g., 4xx/5xx)
            if r.status_code != 200:
                continue

            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.save(os.path.join(IMG_DIR, img_name), format="JPEG")

            return {
                "image_path": img_name,
                "dish_name": item["dish_name"],
                "food_type": item["food_type"],
                "ingredients": item["ingredients"],
                "portion_size": item["portion_size"],
                "nutritional_profile": item["nutritional_profile"],
                "cooking_method": item["cooking_method"],
                "camera_or_phone_prob": item["camera_or_phone_prob"],
                "food_prob": item["food_prob"],
                "sub_dt": item["sub_dt"],
            }

        # in case of unexpected errors, not connected to network timeouts, also try to fetch the data again
        except Exception:
            pass

    return None


# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # -----------------------
    # CONFIG
    # -----------------------

    # paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMG_DIR = os.path.join(ROOT_DIR, "data", "images")
    PARQUET_PATH = os.path.join(ROOT_DIR, "data", "labels.parquet")

    # create image directory if not present
    os.makedirs(IMG_DIR, exist_ok=True)

    # constants
    MAX_WORKERS = 32
    TIMEOUT = 10

    # load dataset, train is the only key in the DatasetDict object
    ds = load_dataset("Codatta/MM-Food-100K")
    split = ds["train"]
    N = len(split)

    # -----------------------
    # RESUME LOGIC
    # -----------------------
    if os.path.exists(PARQUET_PATH):
        # resume
        print("Loading existing dataset...")
        df_existing = pd.read_parquet(PARQUET_PATH)

        # load the set of completed images
        done_set = set(df_existing["image_path"])
        print(f"Already completed: {len(done_set)}")
    else:
        # else start from scratch
        df_existing = pd.DataFrame()
        done_set = set()

    # -----------------------
    # DETERMINE MISSING
    # -----------------------
    missing_indices = [
        i for i in range(N) if os.path.join(IMG_DIR, f"{i:06d}.jpg") not in done_set
    ]

    print(f"To process: {len(missing_indices)}")

    # -----------------------
    # RUN DOWNLOAD
    # -----------------------
    results = []
    lock = threading.Lock()
    failed = 0

    # use multi-threaded download
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # submit tasks
        futures = {
            ex.submit(download_and_save, i, split[i]): i for i in missing_indices
        }

        for f in as_completed(futures):
            res = f.result()

            # use lock to protect shared variables
            with lock:
                if res is not None:
                    results.append(res)
                else:
                    failed += 1

                print(f"Failed: {failed}, New: {len(results)}")

    # -----------------------
    # MERGE + SAVE
    # -----------------------
    df_new = pd.DataFrame(results)

    # merge the existing samples with newly downloaded ones
    df_final = pd.concat([df_existing, df_new], ignore_index=True)

    # deduplicate purely by image_path
    df_final = df_final.drop_duplicates(subset=["image_path"])

    df_final.to_parquet(PARQUET_PATH, index=False)

    print("Done.")
