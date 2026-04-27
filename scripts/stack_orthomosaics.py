import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling
import os

BASE_DIR = "/home/cdac/Desktop/detectree2/odm_projects"
OUT_DIR  = "/home/cdac/Desktop/detectree2/data/orthomosaics"
os.makedirs(OUT_DIR, exist_ok=True)

band_paths = {
    1: f"{BASE_DIR}/dry_band1/odm_orthophoto/odm_orthophoto.tif",  # Blue  450nm
    2: f"{BASE_DIR}/dry_band2/odm_orthophoto/odm_orthophoto.tif",  # Green 560nm
    3: f"{BASE_DIR}/dry_band3/odm_orthophoto/odm_orthophoto.tif",  # Red   650nm
    4: f"{BASE_DIR}/dry_band4/odm_orthophoto/odm_orthophoto.tif",  # RE    730nm
    5: f"{BASE_DIR}/dry_band5/odm_orthophoto/odm_orthophoto.tif",  # NIR   840nm
}

# ── Use band 3 (Red) as the reference grid ────────────────────────────────────
print("Reading reference grid from Band 3 (Red)...")
with rasterio.open(band_paths[3]) as ref:
    ref_profile = ref.profile.copy()
    ref_crs     = ref.crs
    ref_transform = ref.transform
    ref_height  = ref.height
    ref_width   = ref.width
    print(f"  Reference size: {ref_width} x {ref_height}")
    print(f"  CRS: {ref_crs}")
    print(f"  Resolution: {ref.res[0]:.4f} m/px")

# ── Reproject all bands onto reference grid ───────────────────────────────────
print("\nReprojecting and stacking all 5 bands...")
stacked = np.zeros((5, ref_height, ref_width), dtype=np.float32)

for i, (band_num, path) in enumerate(band_paths.items()):
    with rasterio.open(path) as src:
        data = np.zeros((ref_height, ref_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear
        )
        stacked[i] = data
        print(f"  Band {band_num}: min={data.min():.0f}, max={data.max():.0f}, "
              f"mean={data[data>0].mean():.0f}")

# ── Save 5-band stacked orthomosaic ───────────────────────────────────────────
stack_path = f"{OUT_DIR}/dry_5band.tif"
profile = ref_profile.copy()
profile.update(count=5, dtype="float32", nodata=0)
with rasterio.open(stack_path, "w", **profile) as dst:
    dst.write(stacked)
    dst.update_tags(1, name="Blue_450nm")
    dst.update_tags(2, name="Green_560nm")
    dst.update_tags(3, name="Red_650nm")
    dst.update_tags(4, name="RedEdge_730nm")
    dst.update_tags(5, name="NIR_840nm")
print(f"\nSaved 5-band stack: {stack_path}")

# ── Extract 3-channel combos ──────────────────────────────────────────────────
print("\nExtracting channel combinations...")

combos = {
    "rgb":    [2, 1, 0],  # Red, Green, Blue
    "cir":    [4, 2, 0],  # NIR, Red, Blue
    "renir":  [3, 4, 2],  # RedEdge, NIR, Red
}

for name, band_indices in combos.items():
    out_path = f"{OUT_DIR}/dry_{name}.tif"
    combo_data = stacked[band_indices]
    profile_3 = ref_profile.copy()
    profile_3.update(count=3, dtype="float32", nodata=0)
    with rasterio.open(out_path, "w", **profile_3) as dst:
        dst.write(combo_data)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  {name.upper()}: {out_path} ({size_mb:.1f} MB)")

print("\nDone. All channel combinations ready for tiling.")
