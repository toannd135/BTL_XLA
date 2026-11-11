"""Thí nghiệm D1: fine-tune từ C (exports/det_C_neg.pt) với danh sách train của C + ảnh ghép work/synth/train (mỗi ảnh 1 lần).
Cần đã chạy work/gen_synth.py và work/exp/det_neg.py (tạo work/exp/det_C_neg_train.txt, work/exp/neg/test). Dùng: python work/exp/det_synth.py [epochs=20]"""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]; DET = ROOT / "local_dataset" / "det"; name = "det_D_synth"
epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
base = (ROOT / "work/exp/det_C_neg_train.txt").read_text().split()
syn = sorted(str(p) for p in (ROOT / "work/synth/train/images").glob("*.jpg"))
lst = ROOT / "work/exp" / f"{name}_train.txt"; lst.write_text("\n".join(base + syn) + "\n")
yaml = ROOT / "work/exp" / f"{name}.yaml"
yaml.write_text(f"path: {DET}\ntrain: {lst}\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: banana\n  1: apple\n  2: orange\n")
print(f"train: {len(base)} (cấu hình C) + {len(syn)} ảnh ghép", flush=True)
YOLO(str(ROOT / "exports/det_C_neg.pt")).train(data=str(yaml), epochs=epochs, imgsz=640, batch=16, optimizer="SGD", lr0=0.002, warmup_epochs=1,
    seed=42, deterministic=True, patience=0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, workers=4, project=str(ROOT / "runs/exp"), name=name, exist_ok=True, plots=False)
