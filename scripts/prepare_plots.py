import os
import shutil
from pathlib import Path

RAW_DIR = Path("/home/cdac/Desktop/detectree2/data/raw")
ODM_DIR = Path("/home/cdac/Desktop/detectree2/data/odm_projects")

# Updated split including B10
SPLIT = {
    "train": ["plot_B01", "plot_B02", "plot_B03", "plot_B04", "plot_B05"],
    "val":   ["plot_B06", "plot_B07"],
    "test":  ["plot_B08", "plot_B09", "plot_B10"],
}

def collect_band_files(plot_dir, band_num):
    files = []
    for subfolder in sorted(plot_dir.iterdir()):
        if subfolder.is_dir() and "multi" in subfolder.name.lower():
            for f in sorted(subfolder.glob(f"*{band_num}.TIF")):
                files.append(f)
    return files

def collect_rgb_files(plot_dir):
    files = []
    for subfolder in sorted(plot_dir.iterdir()):
        if subfolder.is_dir() and (
            "rgb" in subfolder.name.lower() or
            "optique" in subfolder.name.lower()
        ):
            for f in sorted(subfolder.glob("*.JPG")):
                files.append(f)
    return files

print("=" * 60)
print("UPDATED DATASET SPLIT (including B10)")
print("=" * 60)

for split_name, plots in SPLIT.items():
    total_rgb = 0
    total_multi = 0
    for plot_name in plots:
        plot_dir = RAW_DIR / plot_name
        if not plot_dir.exists():
            print(f"  WARNING: {plot_name} not found")
            continue
        rgb   = collect_rgb_files(plot_dir)
        multi = collect_band_files(plot_dir, "1")
        total_rgb   += len(rgb)
        total_multi += len(multi)
        print(f"  {plot_name}: RGB={len(rgb)}, Multi sets={len(multi)}")
    print(f"  → {split_name.upper()} total: RGB={total_rgb}, "
          f"Multi sets={total_multi}\n")

# Prepare B10 ODM folder (others already done)
print("=" * 60)
print("PREPARING B10 ODM FOLDER")
print("=" * 60)

plot_dir    = RAW_DIR / "plot_B10"
odm_project = ODM_DIR / "plot_B10_band3"
images_dir  = odm_project / "images"
images_dir.mkdir(parents=True, exist_ok=True)

band3_files = collect_band_files(plot_dir, "3")
for f in band3_files:
    dest = images_dir / f.name
    if not dest.exists():
        shutil.copy2(f, dest)

print(f"  plot_B10: {len(band3_files)} band-3 images → {odm_project.name}")
print("\nDone.")
