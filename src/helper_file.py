import pandas as pd
import os

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

data = pd.read_parquet(
    os.path.join(
        DATA_DIR,
        "train_labels_sorted.parquet",
    )
)
pd.set_option("display.max_columns", None)
print(data)
