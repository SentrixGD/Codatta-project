import os

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(ROOT_DIR, "stats", "validation_log.csv")

df = pd.read_csv(csv_path)

df = df.tail(15)

epochs = df["epoch"] + 1
losses = df["total_loss"]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    epochs,
    losses,
    marker="o",
    linewidth=1,
)

ax.set_title("Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Total Loss")
ax.grid(True)

plt.tight_layout()

save_dir = os.path.join(ROOT_DIR, "stats", "results")
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "validation_loss_last15.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved plot to: {save_path}")
