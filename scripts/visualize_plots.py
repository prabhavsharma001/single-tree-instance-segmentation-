import rasterio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ORTHO_DIR = Path("/home/cdac/Desktop/detectree2/data/orthomosaics")

def normalize(arr):
    valid = arr[arr > 0]
    if len(valid) == 0:
        return arr
    p2, p98 = np.percentile(valid, [2, 98])
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

fig, axes = plt.subplots(3, 3, figsize=(18, 18))
fig.suptitle("Channel Combinations — One Plot per Split", fontsize=14)

samples = [
    ("plot_B03", "TRAIN (B03 - smallest)"),
    ("plot_B06", "VAL (B06)"),
    ("plot_B09", "TEST (B09)"),
]

combos = ["rgb", "cir", "renir"]
titles = ["RGB", "CIR (NIR/Red/Blue)", "RENIR (RE/NIR/Red)"]

for row, (plot_name, split_label) in enumerate(samples):
    for col, (combo, title) in enumerate(zip(combos, titles)):
        path = ORTHO_DIR / plot_name / f"{combo}.tif"
        with rasterio.open(path) as src:
            r = src.read(1).astype(float)
            g = src.read(2).astype(float)
            b = src.read(3).astype(float)
        img = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
        axes[row, col].imshow(img)
        axes[row, col].set_title(f"{split_label}\n{title}", fontsize=9)
        axes[row, col].axis("off")

plt.tight_layout()
plt.savefig("/home/cdac/Desktop/detectree2/orthomosaics_preview.png",
            dpi=120, bbox_inches="tight")
print("Saved: orthomosaics_preview.png")
