# Bàn giao: train lại 2026-10-18 + thu thập dữ liệu tự động

## Trạng thái
- `fruit_yolo_retrain_20261018.ipynb` chạy xong cục bộ (RTX 3060 6GB), thêm `raw/rf_v7` + `raw_cls/rf_v3i`. Sửa để chạy được trên 6GB: Ultralytics 8.4.180 bỏ qua `batch=` khi `predict()` nhận list -> tự chia lô (ô evaluate_det, đánh giá classifier). CONFIG cuối ô 3 hiện chỉ OVERWRITE det_finetune/cls_finetune.
- Bản cũ: `backup_20261018_before_retrain/` (exports, runs, work split).
- det_finetune của notebook kém trên ảnh tự quay (best.pt chọn ở epoch 1 vì val chủ yếu public) -> thay bằng thí nghiệm `work/exp/det_joint.py` (yolo11s, public + own x4, 80 epoch) -> `runs/exp/det_joint_s`.
- `exports/det_finetune_best.pt/.onnx` = **yolo11s joint**; `exports/cls_finetune_best.pt` = classifier mới của notebook.

## Kết quả test (mAP50 / mAP50-95)
| model | test | test_public (321 box) | test_own (28 box) |
|---|---|---|---|
| det_finetune cũ | 0.778/0.576 | 0.767/0.555 | 0.937/0.832 |
| det_finetune notebook mới | 0.852/0.667 | 0.878/0.702 | 0.621/0.302 |
| **yolo11s joint (đang dùng)** | 0.903/0.758 | 0.904/0.751 | 0.805/0.740 |
Classifier mới: public acc 0.999 (cũ 0.993), own 27/28 (cũ 28/28). test_own quá nhỏ, chỉ số dao động mạnh.

## Thu thập dữ liệu tự động (mới)
- Web tab "Camera và video": công tắc "Thu thập dữ liệu" + chu kỳ 1/2/5/10 s -> `POST /api/collect` (`backend/collect.py`), `GET /api/collect/stats`.
- Server lưu ảnh gốc + nhãn YOLO 6 class vào `own_cam/` (env `COLLECT_DIR` để đổi) chỉ khi: không mờ, không trùng (dHash), detector + YOLO11l COCO khớp nhau, classifier >= 0.8, không có box bao chùm quả.
- Notebook ô 1.1c chép `own_cam/` -> `own/`; có ảnh mới thì tự OVERWRITE gộp/chia/4 lần train.
- Thử 40 ảnh test: lưu 15, 14 box đúng; sai chủ yếu ở cà chua/dưa chuột bị nhận là táo/cam.

## Lưu ý
- Không chạy đánh giá GPU khi đang train (đã OOM giết 1 lần train; resume bằng `YOLO(last.pt).train(resume=True)`).
- Notebook vẫn dùng `MODEL_SIZE="yolo11n"` + baseline->finetune; chạy lại notebook sẽ GHI ĐÈ `exports/det_finetune_best.pt` bằng yolo11n. Cần đưa cách train joint/yolo11s vào notebook trước lần train sau.
- `pkill -f "<mẫu>"` trong lệnh Bash tự giết shell của chính nó: dùng `pgrep -f "[u]vicorn ..." | xargs kill`.

## Đang chờ
- Người dùng thu thập ảnh bằng camera thật, rồi train lại.
- (XONG 2026-10-24) Ô 4.3b của notebook train joint: `JOINT_MODEL=yolo11s`, `EPOCHS_JOINT=80`, `OWN_REP=4` -> `runs/det_joint`; không nằm trong danh sách OVERWRITE của CONFIG nên không ghi đè khi chạy lại; đã thêm vào bước xuất. Chưa chạy thử trên máy. `det_finetune` (yolo11n) vẫn còn, nhưng `exports/det_finetune_best.*` hiện là bản joint: đừng chạy lại 5.1 mà không đổi tên.
- Đã commit git 1749d2d (dữ liệu, model, runs, configs/kaggle.json nằm trong .gitignore).
