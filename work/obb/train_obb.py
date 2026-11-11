"""Train YOLO11s-OBB trên work/obb/data/obb.yaml (dữ liệu thật đổi sang OBB + ảnh ghép).
Dùng: python work/obb/train_obb.py [name=obb_s1] [epochs=60] [batch=8] [project=runs/obb] [init=yolo11s-obb.pt]
- project: thư mục lưu kết quả (Colab: đặt trên Google Drive để không mất khi ngắt kết nối).
- init: trọng số khởi tạo; mặc định work/weights/yolo11s-obb.pt nếu có, không thì Ultralytics tự tải yolo11s-obb.pt.
- Nếu <project>/<name>/weights/last.pt đã có thì tự train tiếp (resume) thay vì train lại từ đầu.
Máy 6GB: batch 8. Colab T4 16GB: batch 16-24."""
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
a = sys.argv[1:] + [None] * 5
name = a[0] or "obb_s1"
epochs = int(a[1] or 60)
batch = int(a[2] or 8)
project = Path(a[3] or ROOT / "runs/obb").resolve()
local_w = ROOT / "work/weights/yolo11s-obb.pt"
init = a[4] or (str(local_w) if local_w.exists() else "yolo11s-obb.pt")

last = project / name / "weights" / "last.pt"
if last.exists():
    print(f"Resume từ {last}")
    YOLO(str(last)).train(resume=True)
else:
    YOLO(init).train(
        data=str(ROOT / "work/obb/data/obb.yaml"), epochs=epochs, imgsz=640, batch=batch, seed=42, deterministic=False,
        degrees=20, flipud=0.3, fliplr=0.5, mosaic=1.0, close_mosaic=10, patience=25, workers=4,
        project=str(project), name=name, exist_ok=True, plots=True)
