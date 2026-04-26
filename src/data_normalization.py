import os

import pandas as pd
from PIL import Image

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, "data")
    data = pd.read_parquet(os.path.join(DATA_DIR, "labels_processed.parquet"))
    stretched_h = []
    stretched_w = []
    for row in data.itertuples():
        path = row.image_path
        with Image.open(path) as img:
            w, h = img.size
        if w<h:
            stretched_h.append({"w": w, "h": h, "path": path})
        else:
            stretched_w.append({"w": w, "h": h, "path": path})
    stretched_h.sort(key = lambda x: x["h"])
    stretched_w.sort(key = lambda x: x["w"])
    ordered_paths = (
        [x["path"] for x in stretched_h] +
        [x["path"] for x in stretched_w]
    )
    data = data.set_index("image_path").loc[ordered_paths].reset_index()
    data.to_parquet(
        os.path.join(ROOT_DIR, "data", "labels_sorted.parquet"), index=False
    )
    print("Done!")