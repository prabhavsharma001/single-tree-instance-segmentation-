import geopandas as gpd
import numpy as np
from pathlib import Path
from shapely import union
from shapely.geometry import box
from torch import gt

PRED_DIR  = Path("/home/cdac/Desktop/detectree2/data/predictions/rgb")
LABEL_DIR = Path("/home/cdac/Desktop/detectree2/data/labels")

def evaluate(pred_path, gt_path, iou_threshold=0.5):

    preds = gpd.read_file(pred_path)
    gt = gpd.read_file(gt_path)

    print("Checking prediction geometries...")
    print(f"Invalid predictions before fix: {(~preds.is_valid).sum()}")

    print("Checking ground truth geometries...")
    print(f"Invalid ground truth before fix: {(~gt.is_valid).sum()}")

    preds["geometry"] = preds.geometry.buffer(0)
    gt["geometry"] = gt.geometry.buffer(0)

    if preds.crs != gt.crs:
        preds = preds.to_crs(gt.crs)

    print(f"  Predictions: {len(preds)}")
    print(f"  Ground truth: {len(gt)}")

    matched_gt  = set()
    matched_pred = set()

    for pi, pred in preds.iterrows():
        best_iou = 0
        best_gi  = None
        # Only check gt crowns that intersect
        candidates = gt[gt.intersects(pred.geometry)]
        for gi, g in candidates.iterrows():
            inter = pred.geometry.intersection(g.geometry).area
            union = (pred.geometry.area +g.geometry.area -inter)
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_gi  = gi

        if best_iou >= iou_threshold and best_gi not in matched_gt:
            matched_gt.add(best_gi)
            matched_pred.add(pi)

    TP = len(matched_pred)
    FP = len(preds) - TP
    FN = len(gt) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n  IoU threshold: {iou_threshold}")
    print(f"  TP={TP}  FP={FP}  FN={FN}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    return {"precision": precision, "recall": recall, "f1": f1, "TP": TP, "FP": FP, "FN": FN}

print("=" * 50)
print("RGB Model Evaluation — plot_B08")
print("=" * 50)

results = evaluate(
    pred_path  = PRED_DIR / "plot_B08_predictions.gpkg",
    gt_path    = LABEL_DIR / "tree_crowns_B08.gpkg",
    iou_threshold = 0.5
)
