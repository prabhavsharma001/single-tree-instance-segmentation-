import os
from pathlib import Path
from detectree2.preprocessing.tiling import tile_data
from detectree2.models.train import setup_cfg, predictions_on_data
from detectron2.engine import DefaultPredictor

ORTHO_DIR = Path("/home/cdac/Desktop/detectree2/data/orthomosaics")
TILES_DIR = Path("/home/cdac/Desktop/detectree2/data/tiles/rgb")
MODEL_DIR = Path("/home/cdac/Desktop/detectree2/data/models/rgb")
PRED_DIR  = Path("/home/cdac/Desktop/detectree2/data/predictions/rgb")
PRED_DIR.mkdir(parents=True, exist_ok=True)

TEST_PLOTS = ["plot_B08", "plot_B09", "plot_B10"]

# Tile test plots for prediction
print("=== Tiling test plots ===")
for plot in TEST_PLOTS:
    out_dir = TILES_DIR / plot / "pred"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.tif"))
    if existing:
        print(f"  {plot}: {len(existing)} tiles already exist, skipping")
        continue
    tile_data(
        img_path=str(ORTHO_DIR / plot / "rgb.tif"),
        out_dir=str(out_dir),
        buffer=5,
        tile_width=40,
        tile_height=40,
        mode="rgb",
        full_coverage=True,
    )
    n = len(list(out_dir.glob("*.tif")))
    print(f"  {plot}: {n} prediction tiles created")

# Setup predictor
print("\n=== Setting up predictor ===")
cfg = setup_cfg(
    trains=("trees_train_dummy",),
    tests=(),
    out_dir=str(MODEL_DIR),
    max_iter=3000,
    imgmode="rgb",
)
cfg.MODEL.WEIGHTS = str(MODEL_DIR / "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.DEVICE = "cuda"

from detectron2.data import MetadataCatalog, DatasetCatalog
# Register a dummy dataset to avoid errors
if "trees_train_dummy_full" not in DatasetCatalog:
    DatasetCatalog.register("trees_train_dummy_full", lambda: [])
    MetadataCatalog.get("trees_train_dummy_full").set(thing_classes=["tree"])

predictor = DefaultPredictor(cfg)
print("  Predictor ready")

# Run predictions
print("\n=== Running predictions ===")
for plot in TEST_PLOTS:
    pred_dir = TILES_DIR / plot / "pred"
    out_dir  = PRED_DIR / plot
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tiles = len(list(pred_dir.glob("*.tif")))
    print(f"  {plot}: predicting on {n_tiles} tiles...")
    predictions_on_data(
        geos_exist=False,
        directory=str(pred_dir),
        predictor=predictor,
        save=True,
    )
    print(f"  {plot}: done")

print("\nPredictions complete!")
