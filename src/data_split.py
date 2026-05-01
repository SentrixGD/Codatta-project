from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np
import os
import pandas as pd

from collections import Counter
import json


def clean_portion(x):
    if isinstance(x, float):  # NaN case
        return []
    if x is None:
        return []
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except:
            return []
    return x if isinstance(x, list) else []


DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

data = pd.read_parquet(
    os.path.join(
        DATA_DIR,
        "labels_processed.parquet",
    )
)

all_ing = []
for row in data["ingredients"]:
    all_ing.extend(row)

vocab_ing = {k: i for i, (k, _) in enumerate(Counter(all_ing).items())}

N = len(data)
D_ing = len(vocab_ing)

Y_ing = np.zeros((N, D_ing), dtype=np.float32)

for i, row in enumerate(data["ingredients"]):
    for ing in row:
        if ing in vocab_ing:
            Y_ing[i, vocab_ing[ing]] = 1.0


def parse_portion(x):
    return json.loads(x) if isinstance(x, str) else x


portion_sizes = data["portion_size"].apply(parse_portion)


def normalize(s):
    return s.replace(" ", "_").lower()


all_portion = []
for row in portion_sizes:
    for name, _ in row:
        all_portion.append(normalize(name))

vocab_portion = {k: i for i, (k, _) in enumerate(Counter(all_portion).items())}

D_portion = len(vocab_portion)

Y_portion_presence = np.zeros((N, D_portion), dtype=np.float32)
Y_portion_weight = np.zeros((N, D_portion), dtype=np.float32)

for i, row in enumerate(portion_sizes):
    for name, weight in row:
        name = normalize(name)
        if name in vocab_portion:
            j = vocab_portion[name]
            Y_portion_presence[i, j] = 1.0

methods = data["dish_name"].astype(str)

vocab_method = {k: i for i, k in enumerate(methods.unique())}
D_method = len(vocab_method)
methods = data["dish_name"].astype(str)

vocab_method = {k: i for i, k in enumerate(methods.unique())}
D_method = len(vocab_method)
Y_method = np.zeros((N, D_method), dtype=np.float32)

for i, m in enumerate(methods):
    if m in vocab_method:
        Y_method[i, vocab_method[m]] = 1.0

Y_strat = np.concatenate([Y_ing, Y_portion_presence, Y_method], axis=1)

X = np.arange(len(Y_strat))
val_size = 10_000

msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)

train_idx, val_idx = next(msss.split(X, Y_strat))

# If val is slightly larger → move overflow back to train
if len(val_idx) > val_size:
    overflow = val_idx[val_size:]
    val_idx = val_idx[:val_size]
    train_idx = np.concatenate([train_idx, overflow])

val_X = X[val_idx]
val_Y = Y_strat[val_idx]

msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

val_sub_idx, test_sub_idx = next(msss2.split(val_X, val_Y))

val_final_idx = val_idx[val_sub_idx]
test_idx = val_idx[test_sub_idx]

base_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
data.iloc[train_idx].to_parquet(os.path.join(DATA_DIR, "train_labels.parquet"))
data.iloc[val_final_idx].to_parquet(os.path.join(DATA_DIR, "val_labels.parquet"))
data.iloc[test_idx].to_parquet(os.path.join(DATA_DIR, "test_labels.parquet"))

print("Done!")
