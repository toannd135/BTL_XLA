# Đếm chuối trên nải — 2026-11-11

Script: `work/eval_banana_count.py` (chạy đúng `backend/detection.py::detect`, FP16, conf 0.4, IoU 0.6) và `work/viz_banana_count.py` (lưới ảnh ở `work/diag/banana/`, xanh lá = nhãn, đỏ = dự đoán chuối). Kết quả: `work/audit/banana_count_<tag>.{csv,json}`.
Tập đo: 106 ảnh test có chuối; "clean" = 95 ảnh không bị cờ bunch/nested/missing:banana; "multi" = 19 ảnh clean có nhãn >= 2 quả.

## 1. Hiện trạng (tập clean; multi chỉ ghi MAE)

| Cấu hình | Pred/nhãn | MAE | Đúng hẳn | ±1 | P50/R50 | MAE multi | Nền báo sai |
|---|---|---:|---:|---:|---|---:|---:|
| app mặc định `det_finetune_best.pt`, 640 | 123/127 | 0,38 | 70% | 94% | 0,80/0,77 | 1,00 | 1/71 |
| `det_C_candidate.pt`, 640 | 139/127 | 0,38 | 72% | 93% | 0,78/0,85 | 0,79 | 0/71 |
| C_candidate, chế độ careful (960 + xoay) | 305/127 | 1,98 | 17% | 47% | 0,27/0,64 | 1,68 | 3/71 |
| C_candidate, chỉ 960 | 241/127 | 1,39 | 27% | 60% | 0,31/0,59 | 1,11 | 3/71 |
| C_candidate, xoay ở 640 | 154/127 | 0,50 | 66% | 90% | 0,71/0,87 | 1,05 | 0/71 |

- Chế độ careful của app đếm thừa gần gấp 2,4 lần; nguyên nhân chính là imgsz 960 (model học ở 640), xoay chỉ thêm một phần.
- Ở 640 MAE thấp chủ yếu vì 80% ảnh chỉ có 1 nhãn; trên ảnh nhiều quả model đếm thiếu (bias âm).

## 2. Lỗi theo nhóm (xem mắt 16 ảnh sai nhất, `candC_fast_grid*`, `candC_960_grid*`)

| Nhóm | Ví dụ | Ghi chú |
|---|---|---|
| Nhãn test là box cả nải (gt=1 nhưng thật có 4–6 quả) | rf_v7 b_f068, b_f078, b_f009 | 76 ảnh clean có gt=1; 14 ảnh trong đó model ra >= 2 box. Cờ audit không bắt được loại này. |
| Model ra box "nửa nải" (2–3 quả/box), không phải nải cũng không phải từng quả | b_f068, b_f078 ở cả 640 và 960 | Hệ quả trực tiếp của nhãn train trộn nải/từng quả. NMS hay lọc box nhóm không sửa được. |
| Bỏ sót cả nải dày, quả sát nhau | mbk__banana_10 (gt4, pred0), banana_17 (gt4, pred1) | Đếm thiếu. |
| Đếm trùng một quả | r089 (1 quả, 4–5 box), banana_77 | Box lệch/chồng nhau IoU < 0,6 nên NMS không gộp. |

=> Nhãn test hiện tại không đo đúng được việc đếm từng quả (điều kiện dừng). Chưa thử cải tiến Bước 3 vì tối ưu theo số liệu này dễ thành tối ưu nhầm.

## 3. Cải tiến không cần train
Chưa thử (dừng ở trên). Riêng imgsz 960 của chế độ careful đã có số liệu rõ là xấu hơn.

## 4. Khuyến nghị
1. Ngay (không train): đổi `QUALITY_IMGSZ` 960 -> 640 trong `backend/main.py` (careful = xoay ở 640: MAE 0,50, nền 0/71 thay cho 1,98, 3/71). Có thể chuyển app sang `det_C_candidate.pt` (đếm multi tốt hơn: MAE 0,79 so với 1,00). Cả hai thay đổi đều đổi mặc định nên cần người dùng duyệt.
2. Hướng chính là (a): sửa nhãn train/val theo từng quả rồi fine-tune từ C. Lỗi gốc nằm ở nhãn. (c) chỉ sửa hậu xử lý thì không tạo ra được box từng quả. (b) thêm class nải cần detector thứ hai, vẫn phải gán nhãn từng quả trên crop, nên công sức không ít hơn (a).
3. Trước khi train: dựng bộ "đếm thật" khoảng 40 ảnh chuối nhiều quả (chỉ ghi số quả nhìn thấy, cộng box từng quả nếu được), tách khỏi train, để có thước đo.

## Đã làm sau khi người dùng duyệt (2026-11-11)
- `backend/main.py`: `QUALITY_IMGSZ` 960 -> 640; `MODEL_PATH` mặc định -> `exports/det_C_candidate.pt`. `backend/test_detection.py` 16/16 OK. Chưa khởi động lại backend để thử.
- Dựng lại bộ duyệt: `python tools/instance_review.py prepare` -> `out/instance_review_v1/review.html` (3.039 ảnh, test của tools OK khi chạy trong `tools/`).

## Đang chờ
- Người duyệt nhãn từng quả trong `out/instance_review_v1/review.html`, ưu tiên ảnh chuối nhiều quả ở val/test (~40 ảnh) để làm bộ đếm thật; tải `instance_reviews.json`.
- Sau đó: `python tools/evaluate_instance_count.py --reviews instance_reviews.json --roles val test`, rồi `python tools/instance_review.py build --reviews instance_reviews.json` và fine-tune từ C trên bản sạch.

## Ảnh naichuoi.jpeg (2026-11-11)
- Thật ~20 quả (một nải xòe hình quạt). Model C (conf 0.4, NMS 0.6): 12 chuối + 1 "orange" (mặt cắt cuống nải). Ảnh vẽ box: `work/diag/naichuoi/`.
- Đổi NMS chỉ đổi kiểu lỗi: iou 0.8 -> 28 box, 0.9 -> 57 box; TTA 18; imgsz 1024 13; A/B tương tự; YOLO11l COCO 2-17.
- Nguyên nhân gốc: chuối xòe chéo nên box thẳng (axis-aligned) của mỗi quả rất to và chồng lên quả bên cạnh (IoU > 0.6) -> NMS xoá quả thật; nâng ngưỡng thì giữ cả box trùng. Không có ngưỡng nào đúng; đây là giới hạn của box thẳng, không chỉ do nhãn.
- Đề xuất: chuyển chuối sang box xoay (YOLO11-obb, NMS dùng IoU xoay), nhãn từng quả làm bán tự động bằng SAM2 (điểm nhắp -> mask -> minAreaRect). Lưu ý: YOLO-seg của Ultralytics vẫn NMS theo box thẳng nên không giải quyết được.
