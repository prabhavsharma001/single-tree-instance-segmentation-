from pathlib import Path
from detectree2.models.train import setup_cfg, predictions_on_data
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog

MODEL_DIR = Path("/home/cdac/Desktop/detectree2/data/models/rgb")
TILES_DIR = Path("/home/cdac/Desktop/detectree2/data/tiles/rgb")
TEST_PLOTS = ["plot_B08", "plot_B09", "plot_B10"]

cfg = setup_cfg(out_dir=str(MODEL_DIR), max_iter=3000, imgmode="rgb")
cfg.MODEL.WEIGHTS = str(MODEL_DIR / "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.MODEL.DEVICE = "cuda"

if "trees_train_dummy_full" not in DatasetCatalog:
    DatasetCatalog.register("trees_train_dummy_full", lambda: [])
    MetadataCatalog.get("trees_train_dummy_full").set(thing_classes=["tree"])

predictor = DefaultPredictor(cfg)

for plot in TEST_PLOTS:
    pred_dir = TILES_DIR / plot / "pred"
    print(f"Predicting {plot}...")
    predictions_on_data(
        directory=str(pred_dir),
        predictor=predictor,
        save=True,
        geos_exist=False,
    )
    n = len(list((pred_dir / "predictions").glob("*")))
    print(f"  {plot}: {n} prediction files saved")
