import os
import rasterio
import numpy as np
from rasterio.transform import from_bounds

SRC_DIR = "/home/cdac/Downloads/RGB and Multispectral images dataset ( Álamos, Sonora)/Dry Season/Dry Season Multi"
OUT_DIR = "/home/cdac/Desktop/detectree2/odm_projects/dry_multi/images"

os.makedirs(OUT_DIR, exist_ok=True)

# Get all unique image IDs (001, 002, ... 098)
tif_files = sorted([f for f in os.listdir(SRC_DIR)
                    if f.endswith(".TIF") and not f.endswith(".aux.xml")])
image_ids = sorted(set(f[4:7] for f in tif_files))

print(f"Found {len(image_ids)} image sets to stack")
print("Stacking bands: Blue(1) Green(2) Red(3) RedEdge(4) NIR(5)")
print("-" * 50)

success = 0
for img_id in image_ids:
    band_files = [os.path.join(SRC_DIR, f"DJI_{img_id}{b}.TIF")
                  for b in ["1", "2", "3", "4", "5"]]

    # Check all 5 bands exist
    missing = [f for f in band_files if not os.path.exists(f)]
    if missing:
        print(f"  SKIP {img_id}: missing {missing}")
        continue

    out_path = os.path.join(OUT_DIR, f"DJI_{img_id}_multi.TIF")

    # Read all bands
    bands = []
    with rasterio.open(band_files[0]) as src:
        profile = src.profile.copy()
        profile.update(count=5, dtype="uint16")
        # Copy EXIF geotag if present
        tags = src.tags()

    for bf in band_files:
        with rasterio.open(bf) as src:
            bands.append(src.read(1))

    # Write stacked 5-band TIF
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, band in enumerate(bands, 1):
            dst.write(band, i)
        # Write band descriptions
        dst.update_tags(1, name="Blue_450nm")
        dst.update_tags(2, name="Green_560nm")
        dst.update_tags(3, name="Red_650nm")
        dst.update_tags(4, name="RedEdge_730nm")
        dst.update_tags(5, name="NIR_840nm")

    success += 1
    if success % 10 == 0:
        print(f"  Stacked {success}/{len(image_ids)}...")

print(f"\nDone. {success} stacked TIFs saved to {OUT_DIR}")

# Verify one output
sample = os.path.join(OUT_DIR, f"DJI_{image_ids[0]}_multi.TIF")
with rasterio.open(sample) as src:
    print(f"\nSample check: {os.path.basename(sample)}")
    print(f"  Bands: {src.count}")
    print(f"  Size:  {src.width} x {src.height}")
    print(f"  Dtype: {src.dtypes[0]}")
