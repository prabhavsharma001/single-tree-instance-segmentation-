import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

DATASET_ROOT = "/home/cdac/Downloads/RGB and Multispectral images dataset ( Álamos, Sonora)"
DRY_RGB   = os.path.join(DATASET_ROOT, "Dry Season",        "Dry Season RGB")
DRY_MULTI = os.path.join(DATASET_ROOT, "Dry Season",        "Dry Season Multi")
RAIN_RGB  = os.path.join(DATASET_ROOT, "First Rain Season", "Rain Season RPG")
RAIN_MULTI= os.path.join(DATASET_ROOT, "First Rain Season", "Rain Season Multi")

def count_files(folder, ext):
    if not os.path.exists(folder):
        return 0, []
    files = sorted([f for f in os.listdir(folder)
                    if f.upper().endswith(ext.upper()) and not f.endswith(".aux.xml")])
    return len(files), files

dry_rgb_count,    dry_rgb_files    = count_files(DRY_RGB,    ".JPG")
dry_multi_count,  dry_multi_files  = count_files(DRY_MULTI,  ".TIF")
rain_rgb_count,   rain_rgb_files   = count_files(RAIN_RGB,   ".JPG")
rain_multi_count, rain_multi_files = count_files(RAIN_MULTI, ".TIF")

print("=" * 60)
print("DATASET FILE COUNTS")
print("=" * 60)
print(f"Dry Season   RGB:           {dry_rgb_count} JPG images")
print(f"Dry Season   Multispectral: {dry_multi_count} TIF files")
print(f"  → {dry_multi_count // 5} image sets x 5 bands each")
print(f"Rain Season  RGB:           {rain_rgb_count} JPG images")
print(f"Rain Season  Multispectral: {rain_multi_count} TIF files")
print(f"  → {rain_multi_count // 5} image sets x 5 bands each")

# ── Understand naming convention ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("NAMING CONVENTION CHECK (first 10 TIF files)")
print("=" * 60)
for f in dry_multi_files[:10]:
    print(f"  {f}  → image={f[4:7]}, band={f[7]}")

# ── Inspect one complete image set (all 5 bands) ───────────────────────────────
print("\n" + "=" * 60)
print("INSPECTING IMAGE SET 001 - ALL 5 BANDS")
print("=" * 60)

band_labels = {
    "1": "Blue    (450nm)",
    "2": "Green   (560nm)",
    "3": "Red     (650nm)",
    "4": "RedEdge (730nm)",
    "5": "NIR     (840nm)",
}

bands = {}
for band_num in ["1", "2", "3", "4", "5"]:
    fname = f"DJI_0011.TIF".replace("1.TIF", f"{band_num}.TIF")
    fpath = os.path.join(DRY_MULTI, fname)
    with rasterio.open(fpath) as src:
        data = src.read(1).astype(np.float32)
        bands[band_num] = data
        print(f"  Band {band_num} {band_labels[band_num]}: "
              f"shape={data.shape}, min={int(data.min())}, "
              f"max={int(data.max())}, mean={data.mean():.0f}")

# ── Visualize ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING VISUALIZATION → dataset_preview.png")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle("Dataset Preview — Álamos Sonora Tropical Dry Forest", fontsize=14)

cmaps = ["Blues", "Greens", "Reds", "YlGn", "inferno"]

# Row 1: Dry season - all 5 bands
for i, (band_num, cmap) in enumerate(zip(["1","2","3","4","5"], cmaps)):
    fname = f"DJI_0011.TIF".replace("1.TIF", f"{band_num}.TIF")
    fpath = os.path.join(DRY_MULTI, fname)
    with rasterio.open(fpath) as src:
        data = src.read(1)
    axes[0, i].imshow(data, cmap=cmap)
    axes[0, i].set_title(f"Dry — {band_labels[band_num].strip()}")
    axes[0, i].axis("off")

# Row 2: Rain season - all 5 bands
for i, (band_num, cmap) in enumerate(zip(["1","2","3","4","5"], cmaps)):
    fname = f"DJI_0121.TIF".replace("1.TIF", f"{band_num}.TIF")
    fpath = os.path.join(RAIN_MULTI, fname)
    with rasterio.open(fpath) as src:
        data = src.read(1)
    axes[1, i].imshow(data, cmap=cmap)
    axes[1, i].set_title(f"Rain — {band_labels[band_num].strip()}")
    axes[1, i].axis("off")

plt.tight_layout()
plt.savefig("dataset_preview.png", dpi=150, bbox_inches="tight")
print("Saved: dataset_preview.png")

# ── NDVI quick check ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NDVI QUICK CHECK (Dry vs Rain season)")
print("=" * 60)

def compute_ndvi(multi_dir, image_id):
    nir_path = os.path.join(multi_dir, f"DJI_{image_id}5.TIF")
    red_path = os.path.join(multi_dir, f"DJI_{image_id}3.TIF")
    with rasterio.open(nir_path) as s: nir = s.read(1).astype(float)
    with rasterio.open(red_path) as s: red = s.read(1).astype(float)
    ndvi = (nir - red) / (nir + red + 1e-8)
    return ndvi

ndvi_dry  = compute_ndvi(DRY_MULTI,  "001")
ndvi_rain = compute_ndvi(RAIN_MULTI, "012")

print(f"  Dry  season NDVI: mean={ndvi_dry.mean():.3f},  "
      f"min={ndvi_dry.min():.3f},  max={ndvi_dry.max():.3f}")
print(f"  Rain season NDVI: mean={ndvi_rain.mean():.3f}, "
      f"min={ndvi_rain.min():.3f}, max={ndvi_rain.max():.3f}")
print("\n  (Higher NDVI in rain season = more green vegetation)")
print("\nDONE.")
