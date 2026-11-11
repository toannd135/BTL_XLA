"""Thí nghiệm B: giống baseline det_joint_s (yolo11s, public + own x4, 80 epoch, seed 42) nhưng LOẠI các ảnh train bị cắm cờ bunch/nested
(work/audit/quarantine_train.txt). Không xoá file; val/test giữ nguyên. Dùng: python work/exp/det_expB.py [name=det_B_clean]"""
import csv, sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DET, name = ROOT / "local_dataset" / "det", (sys.argv[1] if len(sys.argv) > 1 else "det_B_clean")
quar = {r["image"] for r in csv.DictReader(open(ROOT / "work" / "audit" / "det_label_audit.csv"))
        if r["split"] == "train" and any(f.split(":")[0] in ("bunch", "nested") for f in r["flags"].split(";"))}
(ROOT / "work" / "audit" / "quarantine_train.txt").write_text("\n".join(sorted(quar)) + "\n")
pub = [p for p in sorted((DET / "train" / "images").iterdir()) if p.name not in quar]
own = sorted(p for p in (DET / "train_ft" / "images").iterdir() if p.name.startswith("own-"))
lst = ROOT / "work" / "exp" / f"{name}_train.txt"; lst.write_text("\n".join(str(p) for p in pub + own * 4) + "\n")
yaml = ROOT / "work" / "exp" / f"{name}.yaml"
yaml.write_text(f"path: {DET}\ntrain: {lst}\nval: val/images\ntest: test/images\nnc: 3\nnames:\n  0: banana\n  1: apple\n  2: orange\n")
print(f"train: {len(pub)} public (loại {len(quar)} ảnh nghi nhãn nải/lồng) + {len(own)} own x4", flush=True)
YOLO("yolo11s.pt").train(data=str(yaml), epochs=80, imgsz=640, batch=16, optimizer="SGD", lr0=0.01, seed=42, deterministic=True, patience=30,
                         hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, workers=4, project=str(ROOT / "runs" / "exp"), name=name, exist_ok=True, plots=False)
