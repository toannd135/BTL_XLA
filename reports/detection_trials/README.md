# Hai lượt detector mới ngày 16/11/2026

Kết quả gốc do người dùng chạy trên Colab, sao chép vào repo và kiểm tra SHA256.
Danh mục model: [exports/trial_models.json](../../exports/trial_models.json).
Model mặc định của app vẫn là `exports/det_obb_s.pt`; hai ứng viên chưa được tự động nhận.

| Lượt | File trọng số dự kiến | Báo cáo |
|---|---|---|
| V1 | `exports/det_obb_trial_v1.pt` — đã có, hash khớp | [Validation640](trial_v1/validation_640.json) |
| V2 | `exports/det_obb_trial_v2.pt` — đã có, hash khớp | [Validation1280](trial_v2/validation_1280.json) |

Không tạo file giả hoặc đổi tên trọng số V1 thành V2 để lấp chỗ thiếu.

## Kết quả theo đúng cấu hình đo

| Báo cáo | Model | imgsz | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|
| Validation V1 | baseline | 640 | .8957 | .7418 |
| Validation V1 | trial_v1 | 640 | .8990 | .7498 |
| Validation V2 | baseline | 1280 | .6321 | .3014 |
| Validation V2 | trial_v1 | 1280 | .5387 | .2523 |
| Validation V2 | trial_v2 | 1280 | .7909 | .5385 |

Các dòng640/1280 không phải phép so sánh cùng giao thức. V2 tốt hơn hai model kia ở1280,
nhưng chưa kết luận tốt nhất trên mọi độ phân giải. Nhãn cũ còn gộp nải chuối; chưa phải
benchmark độc lập đủ nhãn từng quả cho cả3loại. CLIP/teacher/classifier không nằm trong phép đo này.

## Ảnh và phân tích

- `trial_v1/diagnostics_640/`, `trial_v1/diagnostics_1280/`: nhãn / baseline / V1 và tọa độ chẩn đoán.
- `trial_v2/validation/`: toàn bộ59file đã nhận, gồm PR/F1/confusion matrix, previews và JSON từng ảnh.
- [Phân tích V1](trial_v1/KET_QUA_SO_SANH.md), [phân tích V2](trial_v2/PHAN_TICH.md).
- Notebook/input tái lập nằm tại `out/accuracy_trial_20261116/`, `out/all_fruits_v2_20261116/`;
  các thư mục này hiện được phép thêm vào Git. ZIP dùng Git LFS.

## Trọng số V2 đã bổ sung

Đã kiểm tra file V2 trên máy ngày16/11/2026 và sửa tên bị dư `.pt.pt` thành `.pt`.
Hiện đã có đủ trọng số V1/V2, SHA256 khớp hai báo cáo. Phần dưới ghi lại nguồn và tên file
để tải lại khi cần.

Tải file gốc từ Drive:

```text
My Drive/fruit_all3_v2_20261116/runs/all3_balanced_v2/weights/best.pt
```

Lưu đúng tại:

```text
BTL_XLA/exports/det_obb_trial_v2.pt
```

SHA256 phải là:

```text
d0abaa79cc3784849dc3bf7c178033c56df2990b71a2f69397dfb94fc463796d
```

Kiểm tra báo cáo và trọng số thực tế:

```powershell
.\.venv\Scripts\python.exe tools\check_trial_artifacts.py --require-models
```

Nếu thiếu trọng số trên máy khác, lệnh báo thiếu và trả mã1; không coi việc có JSON kết quả là đã có trọng số.
Tắt tùy chọn `--require-models` nếu chỉ kiểm tra hai báo cáo và các trọng số sẵn có.
