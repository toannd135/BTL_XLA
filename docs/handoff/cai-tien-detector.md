# Bàn giao: cải tiến detector (2026-11-08)

## Mục tiêu
Giảm nhận sai của detector (chuối/táo/cam) trên webcam, đặc biệt nhận nhầm vật khác thành quả và box chuối thừa. Đã triển khai model B; đang tạm dừng train.

## Nhánh/trạng thái git
- Không xác định (lệnh git không trả output ở `BTL_XLA/`). Chưa commit gì trong phiên này; mọi thay đổi chưa lưu vào git.

## Đã xong
- Baseline A ghi cố định: `work/audit/eval_A_baseline.json`; thí nghiệm B (`runs/exp/det_B_clean`): `eval_B_clean.json`.
- Rà nhãn (`work/audit_det_labels.py` -> `work/audit/det_label_audit.csv`), phân loại box thừa (`fp_*.csv`).
- Triển khai B vào `exports/det_finetune_best.pt` (A backup: `exports/det_A_baseline_backup.pt/.onnx`), FP16, `NMS_IOU` 0.6.
- Tải 484 ảnh nền Wikimedia, lọc YOLO11l: 358 giữ ở `own_neg/`, 126 chuyển `own_neg_flagged/`.
- Tạo `CLAUDE.md` (trạng thái + kế hoạch đầy đủ).

## Quyết định đã chốt
- Quy ước nhãn: gán từng quả. Quarantine 50 ảnh train nghi nhãn nải/lồng (không xoá file).
- Giữ nguyên val/test gốc; clean test là bản sao riêng do người xác minh.
- Hard negative: không nhân >2x; giữ 20% ảnh nền làm bộ kiểm tra độc lập.
- Không đổi kiến trúc (yolo11s + classifier + teacher yolo11l).

## Tệp quan trọng
- `CLAUDE.md` (kế hoạch), `work/eval_det.py`, `work/exp/det_expB.py`, `work/exp/det_neg.py`
- `backend/main.py:29` (NMS_IOU), `backend/collect.py:29`

## Việc còn lại
1. Thí nghiệm C theo `CLAUDE.md` mục "Kế hoạch tiếp theo" bước 1 (chỉnh `det_neg.py`: loại quarantine, `neg_rep`<=2).
2. Clean test (người xác minh ~30 ảnh test cắm cờ); thí nghiệm D (ảnh webcam).

## Đang chờ
- Người dùng: ảnh webcam chuối/táo/cam đã kiểm tra nhãn; xác minh nhãn ảnh test cắm cờ.
- Máy nóng (CPU ~97–100°C): kê cao máy trước khi train.

## Lệnh kiểm tra
- `curl -s http://localhost:8000/api/health | jq -c .`
- `python3 work/eval_det.py exports/det_finetune_best.pt <tag>`
