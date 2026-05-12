import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
from pathlib import Path
import os

ODM_DIR  = Path("/home/cdac/Desktop/detectree2/data/odm_projects")
OUT_DIR  = Path("/home/cdac/Desktop/detectree2/data/orthomosaics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS = [f"plot_B{i:02d}" for i in range(1, 11)]

SPLIT = {
    "train": ["plot_B01","plot_B02","plot_B03","plot_B04","plot_B05"],
    "val":   ["plot_B06","plot_B07"],
    "test":  ["plot_B08","plot_B09","plot_B10"],
}

def stack_plot(plot_name):
    band_paths = {
        1: ODM_DIR / f"{plot_name}_band1/odm_orthophoto/odm_orthophoto.tif",
        2: ODM_DIR / f"{plot_name}_band2/odm_orthophoto/odm_orthophoto.tif",
        3: ODM_DIR / f"{plot_name}_band3/odm_orthophoto/odm_orthophoto.tif",
        4: ODM_DIR / f"{plot_name}_band4/odm_orthophoto/odm_orthophoto.tif",
        5: ODM_DIR / f"{plot_name}_band5/odm_orthophoto/odm_orthophoto.tif",
    }

    # Use band 3 as reference grid
    with rasterio.open(band_paths[3]) as ref:
        profile    = ref.profile.copy()
        ref_crs    = ref.crs
        ref_trans  = ref.transform
        ref_h      = ref.height
        ref_w      = ref.width

    # Reproject all bands onto reference grid
    stacked = np.zeros((5, ref_h, ref_w), dtype=np.float32)
    for i, (band_num, path) in enumerate(band_paths.items()):
        with rasterio.open(path) as src:
            data = np.zeros((ref_h, ref_w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_trans,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
            stacked[i] = data

    # Save 5-band stack
    plot_dir = OUT_DIR / plot_name
    plot_dir.mkdir(exist_ok=True)

    profile.update(count=5, dtype="float32", nodata=0)
    with rasterio.open(plot_dir / "5band.tif", "w", **profile) as dst:
        dst.write(stacked)

    # Extract channel combos
    combos = {
        "rgb":   [2, 1, 0],   # Red, Green, Blue
        "cir":   [4, 2, 0],   # NIR, Red, Blue
        "renir": [3, 4, 2],   # RedEdge, NIR, Red
    }
    profile.update(count=3)
    for name, indices in combos.items():
        with rasterio.open(plot_dir / f"{name}.tif", "w", **profile) as dst:
            dst.write(stacked[indices])

    return ref_w, ref_h

print("=" * 60)
print("STACKING ALL PLOTS")
print("=" * 60)

for split_name, plots in SPLIT.items():
    print(f"\n--- {split_name.upper()} ---")
    for plot_name in plots:
        w, h = stack_plot(plot_name)
        size = sum(
            os.path.getsize(OUT_DIR / plot_name / f)
            for f in ["5band.tif","rgb.tif","cir.tif","renir.tif"]
        ) / 1e6
        print(f"  {plot_name}: {w}x{h} → {size:.0f} MB total")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
for split_name, plots in SPLIT.items():
    print(f"\n{split_name.upper()} plots: {plots}")

print(f"\nOutput directory: {OUT_DIR}")
print("Done.")
