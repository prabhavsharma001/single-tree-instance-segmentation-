import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

ODM_DIR   = "/home/cdac/Desktop/detectree2/odm_projects/dry_rgb"
DSM_PATH  = f"{ODM_DIR}/odm_dem/dsm.tif"
DTM_PATH  = f"{ODM_DIR}/odm_dem/dtm.tif"
CHM_PATH  = f"{ODM_DIR}/odm_dem/chm.tif"
ORTHO_PATH= f"{ODM_DIR}/odm_orthophoto/odm_orthophoto.tif"

# ── 1. Inspect orthophoto ──────────────────────────────────────────────────────
print("=" * 60)
print("ORTHOPHOTO INFO")
print("=" * 60)
with rasterio.open(ORTHO_PATH) as src:
    print(f"Bands:      {src.count}")
    print(f"Size:       {src.width} x {src.height} px")
    print(f"CRS:        {src.crs}")
    print(f"Resolution: {src.res[0]:.4f} m/px")
    print(f"Bounds:     {src.bounds}")

# ── 2. Generate CHM = DSM - DTM ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING CHM (DSM - DTM)")
print("=" * 60)

with rasterio.open(DSM_PATH) as dsm_src:
    dsm  = dsm_src.read(1).astype(np.float32)
    profile = dsm_src.profile
    nodata  = dsm_src.nodata or -9999

with rasterio.open(DTM_PATH) as dtm_src:
    dtm = dtm_src.read(
        1,
        out_shape=(dsm_src.height, dsm_src.width),
        resampling=Resampling.bilinear
    ).astype(np.float32)

# Mask nodata
mask = (dsm == nodata) | (dtm == nodata)
chm  = dsm - dtm
chm[mask]  = 0
chm[chm < 0] = 0   # remove negative noise
chm[chm > 50] = 50 # cap unrealistic heights

print(f"CHM stats:")
valid = chm[~mask]
print(f"  Min height:  {valid.min():.2f} m")
print(f"  Max height:  {valid.max():.2f} m")
print(f"  Mean height: {valid.mean():.2f} m")
print(f"  Pixels > 2m (likely trees): {(valid > 2).sum():,} ({100*(valid>2).mean():.1f}%)")
print(f"  Pixels > 5m (tall trees):   {(valid > 5).sum():,} ({100*(valid>5).mean():.1f}%)")

# Save CHM
profile.update(dtype=rasterio.float32, count=1, nodata=0)
with rasterio.open(CHM_PATH, 'w', **profile) as dst:
    dst.write(chm, 1)
print(f"\nSaved CHM to: {CHM_PATH}")

# ── 3. Visualize ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("GENERATING VISUALIZATION → odm_outputs_preview.png")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("ODM Outputs — Dry Season RGB", fontsize=14)

with rasterio.open(ORTHO_PATH) as src:
    r = src.read(1)
    g = src.read(2)
    b = src.read(3)
    rgb = np.stack([r, g, b], axis=-1)
    # Normalize for display
    rgb = np.clip(rgb / np.percentile(rgb, 98) * 255, 0, 255).astype(np.uint8)

axes[0].imshow(rgb)
axes[0].set_title("RGB Orthomosaic")
axes[0].axis("off")

with rasterio.open(DSM_PATH) as src:
    dsm_vis = src.read(1)
    dsm_vis = np.where(dsm_vis == nodata, np.nan, dsm_vis)

axes[1].imshow(dsm_vis, cmap="terrain")
axes[1].set_title("DSM (Digital Surface Model)")
axes[1].axis("off")

axes[2].imshow(chm, cmap="YlGn", vmin=0, vmax=15)
axes[2].set_title("CHM (Canopy Height Model)")
axes[2].axis("off")
plt.colorbar(axes[2].images[0], ax=axes[2], label="Height (m)", shrink=0.8)

plt.tight_layout()
plt.savefig("odm_outputs_preview.png", dpi=150, bbox_inches="tight")
print("Saved: odm_outputs_preview.png")
print("\nDONE. Orthomosaic + CHM ready for next step.")
