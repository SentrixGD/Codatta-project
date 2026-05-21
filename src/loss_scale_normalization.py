import os

import pandas as pd

if __name__ == "__main__":
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = pd.read_csv(os.path.join(ROOT_DIR, "stats", "losses.csv"))
    print(data)
