"""Train detection trên public + own x4 + COCO (chuối/táo/cam, ảnh đa dạng, đếm nhiều quả). Val = val cũ + nửa COCO val; test_coco = nửa còn lại."""
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DET = ROOT / "local_dataset" / "det"
COCO = ROOT / "raw" / "coco"
model, name, own_rep, epochs, batch = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
lab = lambda d: {p.stem for p in (COCO / d / "labels").glob("*.txt")}

pub = sorted((DET / "train" / "images").iterdir())
own = sorted(p for p in (DET / "train_ft" / "images").iterdir() if p.name.startswith("own-"))
ctr = sorted(p for p in (COCO / "train" / "images").iterdir() if p.stem in lab("train"))
cva = sorted(p for p in (COCO / "val" / "images").iterdir() if p.stem in lab("val"))
cval, ctest = cva[0::2], cva[1::2]
E=ROOT / "work" / "exp"
(E / f"{name}_train.txt").write_text("\n".join(map(str, pub + own * own_rep + ctr)) + "\n")
(E / f"{name}_val.txt").write_text("\n".join(map(str, sorted((DET / "val" / "images").iterdir()) + cval)) + "\n")
(E / "coco_test.txt").write_text("\n".join(map(str, ctest)) + "\n")
(E / f"{name}.yaml").write_text(f"path: {DET}\ntrain: {E/(name+'_train.txt')}\nval: {E/(name+'_val.txt')}\nnc: 3\nnames:\n  0: banana\n  1: apple\n  2: orange\n")
(E / "coco_test.yaml").write_text(f"path: {DET}\ntrain: {E/'coco_test.txt'}\nval: {E/'coco_test.txt'}\nnc: 3\nnames:\n  0: banana\n  1: apple\n  2: orange\n")
print(f"train: {len(pub)} public + {len(own)} own x{own_rep} + {len(ctr)} coco; val +{len(cval)} coco; test_coco {len(ctest)}", flush=True)
if epochs:
    YOLO(model).train(data=str(E / f"{name}.yaml"), epochs=epochs, imgsz=640, batch=batch, optimizer="SGD", lr0=0.01, seed=42,
                      deterministic=True, patience=25, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, workers=4,
                      project=str(ROOT / "runs" / "exp"), name=name, exist_ok=True, plots=False)
