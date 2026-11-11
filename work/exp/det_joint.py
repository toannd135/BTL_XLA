"""Thử nghiệm sau train lại 2026-10-18: train detection MỘT lần trên public + ảnh tự quay (nhân bản x OWN_REP),
thay cho baseline -> fine-tune (fine-tune mới chọn best.pt ở epoch 1 vì val chủ yếu là public)."""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "local_dataset" / "det"
model, name, own_rep, epochs, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

pub = sorted((DET / "train" / "images").iterdir())
own = sorted(p for p in (DET / "train_ft" / "images").iterdir() if p.name.startswith("own-"))
lst = ROOT / "work" / "exp" / f"{name}_train.txt"
lst.write_text("\n".join(str(p) for p in pub + own * own_rep) + "\n")
yaml = ROOT / "work" / "exp" / f"{name}.yaml"
yaml.write_text(f"path: {DET}\ntrain: {lst}\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: banana\n  1: apple\n  2: orange\n")
print(f"train: {len(pub)} public + {len(own)} own x{own_rep}")

YOLO(model).train(data=str(yaml), epochs=epochs, imgsz=640, batch=batch, optimizer="SGD", lr0=0.01, seed=42,
                  deterministic=True, patience=30, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, workers=4,
                  project=str(ROOT / "runs" / "exp"), name=name, exist_ok=True, plots=False)
