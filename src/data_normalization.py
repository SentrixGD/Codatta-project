"""
File: data_normalization.py

Description:
    Sort images by height and width and store them in a new parquet file for each set.

Purpose:
    In order to avoid online preprocessing, the data is preprocessed offline and stored locally. Sorted data enables less padding due to similar sizes of consecutive images.

Inputs:
    - Resized images are stored in the /data/resized_images directory.
    - Labels are stored in the /data/{train, val, test}_labels.parquet files.

Outputs:
    - Sorted labels are stored in the /data/{train, val, test}_labels_sorted.parquet files.

Dependencies:
    - pandas
    - PIL

Usage:
    python -m src.data_normalization
"""

import os

import pandas as pd
from PIL import Image

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    for i in ["train", "val", "test"]:
        data = pd.read_parquet(os.path.join(DATA_DIR, f"{i}_labels.parquet"))
        stretched_h = []
        stretched_w = []
        for row in data.itertuples():
            path = os.path.join(DATA_DIR, "resized_images", row.image_path)
            with Image.open(path) as img:
                w, h = img.size
            # Split images by aspect ratio orientation to group similar shapes together (used to reduce padding variance in batching)
            if w < h:
                stretched_h.append({"w": w, "h": h, "path": row.image_path})
            else:
                stretched_w.append({"w": w, "h": h, "path": row.image_path})
        # Sort each group by its dominant spatial dimension to cluster similarly shaped images together
        stretched_h.sort(key=lambda x: x["h"])
        stretched_w.sort(key=lambda x: x["w"])
        ordered_paths = [x["path"] for x in stretched_h] + [
            x["path"] for x in stretched_w
        ]
        data = data.set_index("image_path").loc[ordered_paths].reset_index()
        data.to_parquet(
            os.path.join(ROOT_DIR, "data", f"{i}_labels_sorted.parquet"), index=False
        )
    print("Done!")
