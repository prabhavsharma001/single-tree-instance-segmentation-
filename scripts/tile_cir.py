import geopandas as gpd
from pathlib import Path
from detectree2.preprocessing.tiling import tile_data, to_traintest_folders
import json

ORTHO_DIR = Path("/home/cdac/Desktop/detectree2/data/orthomosaics")
LABEL_DIR = Path("/home/cdac/Desktop/detectree2/data/labels")
TILES_DIR = Path("/home/cdac/Desktop/detectree2/data/tiles/cir")

TRAIN_PLOTS = ["plot_B03", "plot_B04"]

for plot in TRAIN_PLOTS:
    label_file = LABEL_DIR / f"tree_crowns_{plot.replace('plot_','')}.gpkg"
    ortho_file = ORTHO_DIR / plot / "cir.tif"
    out_dir    = TILES_DIR / plot
    out_dir.mkdir(parents=True, exist_ok=True)

    crowns = gpd.read_file(label_file)
    print(f"  {plot}: {len(crowns)} crowns → tiling CIR...")

    tile_data(
        img_path=str(ortho_file),
        out_dir=str(out_dir),
        buffer=5,
        tile_width=40,
        tile_height=40,
        crowns=crowns,
        threshold=0.1,
        nan_threshold=0.5,
        mode="rgb",
        tile_placement="adaptive",
    )

    # Fix geojson imagePaths
    for gj in out_dir.glob("*.geojson"):
        with open(gj) as f:
            data = json.load(f)
        data["imagePath"] = str(out_dir / f"{gj.stem}.png")
        with open(gj, "w") as f:
            json.dump(data, f)

    to_traintest_folders(
        tiles_folder=str(out_dir),
        out_folder=str(out_dir),
        test_frac=0.1,
    )

    # Copy images into fold_1
    for gj in (out_dir / "train" / "fold_1").glob("*.geojson"):
        stem = gj.stem
        for ext in [".png", ".tif"]:
            src = out_dir / f"{stem}{ext}"
            if src.exists():
                import shutil
                shutil.copy2(src, out_dir / "train" / "fold_1" / f"{stem}{ext}")

    # Fix fold_1 geojson paths
    for gj in (out_dir / "train" / "fold_1").glob("*.geojson"):
        with open(gj) as f:
            data = json.load(f)
        data["imagePath"] = str(out_dir / "train" / "fold_1" / f"{gj.stem}.png")
        with open(gj, "w") as f:
            json.dump(data, f)

    n = len(list((out_dir / "train" / "fold_1").glob("*.geojson")))
    print(f"  {plot}: {n} train tiles ready")

print("CIR tiling complete!")
