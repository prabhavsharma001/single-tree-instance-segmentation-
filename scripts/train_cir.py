import os
from pathlib import Path
from detectree2.models.train import setup_cfg, MyTrainer, register_train_data, register_test_data

TILES_DIR = Path("/home/cdac/Desktop/detectree2/data/tiles/cir")
MODEL_DIR = Path("/home/cdac/Desktop/detectree2/data/models/cir")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PLOTS = ["plot_B03", "plot_B04"]

# Register train data
train_names = []
for plot in TRAIN_PLOTS:
    train_dir = TILES_DIR / plot / "train"
    if train_dir.exists():
        name = f"trees_{plot}"
        register_train_data(str(train_dir), name=name)
        train_names.append(f"{name}_full")
        print(f"  Train: {name}_full")

# Register test data — use plot_B03 test tiles
test_name = "trees_test"
test_dir = str(TILES_DIR / "plot_B03" / "test")
register_test_data(test_dir, test_name)
print(f"  Test:  {test_name}")

print(f"\nRegistered train: {train_names}")

print("\n=== Training (CPU) ===")
cfg = setup_cfg(
    trains=tuple(train_names),
    tests=("trees_test_test",),
    out_dir=str(MODEL_DIR),
    max_iter=3000,
    workers=2,
    eval_period=500,
    imgmode="rgb",
)
cfg.MODEL.DEVICE = "cuda"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = MyTrainer(cfg, patience=10)
trainer.resume_or_load(resume=False)
trainer.train()
print("\nTraining complete!")
