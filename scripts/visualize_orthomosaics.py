import rasterio
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = "/home/cdac/Desktop/detectree2/data/orthomosaics"

def normalize(arr):
    p2, p98 = np.percentile(arr[arr > 0], [2, 98])
    return np.clip((arr - p2) / (p98 - p2 + 1e-8), 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle("Dry Season — Channel Combinations (Orthomosaics)", fontsize=14)

titles = ["RGB (Red/Green/Blue)", "CIR (NIR/Red/Blue)", "RENIR (RedEdge/NIR/Red)"]
files  = ["dry_rgb.tif", "dry_cir.tif", "dry_renir.tif"]

for ax, title, fname in zip(axes, titles, files):
    with rasterio.open(f"{OUT_DIR}/{fname}") as src:
        r = src.read(1).astype(float)
        g = src.read(2).astype(float)
        b = src.read(3).astype(float)

    rgb = np.stack([normalize(r), normalize(g), normalize(b)], axis=-1)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    ax.axis("off")

plt.tight_layout()
plt.savefig("orthomosaics_preview.png", dpi=150, bbox_inches="tight")
print("Saved: orthomosaics_preview.png")
