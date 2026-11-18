# Kết quả fine-tune mAP50–95 — S1/S2

Nhận ngày 17/11/2026. Báo cáo gốc: [comparison_val.json](comparison_val.json).
Đối chiếu file, SHA256, metadata checkpoint: [artifacts.json](artifacts.json).

## Kết luận

**Hai lượt fine-tune không vượt checkpoint V1 đã dùng để khởi tạo về mAP50–95.**
Giữ V1 làm mốc OBB trong giao thức này; không thay model app bằng S1/S2.
Model mặc định của app vẫn là `exports/det_obb_s.pt`, không tự chuyển sang V1.
Tiếp tục hướng thử nghiệm instance segmentation đã thống nhất thay vì tăng epoch/S3 tùy tiện.

## So sánh cùng giao thức

297 ảnh validation kế thừa all3 v2; imgsz640, batch2, AP conf0.001, NMS IoU0.7,
agnostic=True, max_det300, rect=True, FP32, không augmentation lúc đánh giá.
Ultralytics8.4.180, Torch2.14.1+cu130, TeslaT4 theo báo cáo.

| Model | mAP50 | mAP75 | mAP50–95 | AP50–95 chuối | AP50–95 táo | AP50–95 cam |
|---|---:|---:|---:|---:|---:|---:|
| baseline app | 0.8748 | 0.7517 | 0.7243 | 0.3866 | 0.8942 | 0.8921 |
| **V1 — checkpoint khởi tạo** | 0.8791 | 0.7507 | **0.7375** | **0.4066** | 0.8974 | 0.9084 |
| V2 | 0.8648 | 0.7252 | 0.7028 | 0.3376 | 0.8842 | 0.8865 |
| S1 — lr0.0001 | 0.8743 | 0.7472 | 0.7269 | 0.3633 | **0.8997** | **0.9176** |
| S2 — lr0.0003 | **0.8844** | 0.7492 | 0.7257 | 0.3788 | 0.8967 | 0.9016 |

- S1 so V1: mAP50–95 giảm **0.01061**, tương đương **1.06 điểm phần trăm**.
  Chuối giảm4.33 điểm; táo tăng0.23 và cam tăng0.92 điểm phần trăm.
- S2 so V1: mAP50–95 giảm **0.01178**, tương đương **1.18 điểm phần trăm**.
  AP50–95 cả ba loại giảm: chuối2.78, táo0.07, cam0.68 điểm phần trăm.
- S2 có mAP50 cao hơn V1 nhưng không cải thiện mAP75/mAP50–95.
  Không dùng riêng mAP50 để tuyên bố đạt mục tiêu cải thiện độ khít box.
- Táo/cam đã có AP50–95 khoảng0.90 trong bộ này, còn chuối khoảng0.36–0.41.
  Đây là chênh lệch quan sát được; chỉ từ bảng số không xác định được tỉ lệ lỗi do nhãn,
  thiếu dữ liệu, box gộp hay cách train. Cần ảnh và nhãn từng trường hợp để phân tích nguyên nhân.

## Đếm và phát hiện ở conf0.4

Các số sau dựa trên nhãn cũ của cùng297ảnh, có nhãn cả nải; không phải số đếm thực tế đã duyệt lại.

| Model | Macro F1 | MAE tổng số đếm theo lớp | Đếm đúng cả ba lớp |
|---|---:|---:|---:|
| V1 | 0.8598 | **0.3232** | **224/297 (75.42%)** |
| S1 | 0.8523 | 0.3502 | 220/297 (74.07%) |
| S2 | 0.8427 | 0.3771 | 216/297 (72.73%) |

MAE thấp hơn tốt hơn. S1 cải thiện nhẹ táo/cam ở operating point nhưng giảm recall chuối
từ0.6646 xuống0.6013; S2 recall chuối0.6329, đồng thời F1 cam giảm.
Không có kết quả ảnh nền giữ riêng hoặc test độc lập mới trong file đã nhận.

## Trọng số đã xác minh và lưu riêng

| File người dùng gửi | Xác định theo SHA256 | Bản lưu trong repo |
|---|---|---|
| `best.pt` | S1 | `exports/det_obb_map95_s1.pt` |
| `best (1).pt` | S2 | `exports/det_obb_map95_s2.pt` |

Cả hai là OBB, classes0banana/1apple/2orange, khởi tạo từ `models/trial_v1.pt`,
imgsz640, batch2, AdamW, cấu hình tối đa60epoch/patience12. Metadata xác nhận lr0 tương ứng.
Không suy ra đã chạy đủ60epoch từ trường cấu hình này.
Checkpoint cuối đã strip optimizer, epoch=-1; dùng để inference hoặc khởi tạo fine-tune mới,
không dùng `resume=True` để phục hồi đầy đủ lượt train đã kết thúc.

Hash input contract và runner trong báo cáo khớp gói Colab đã chuẩn bị tại repo.
Hash ba checkpoint cũ và hai checkpoint nhận mới khớp báo cáo.
Đã kiểm tra trung bình AP từng lớp, P/R/F1 từ TP/FP/FN và tổng MAE trong JSON.
Chỉ đọc metadata trên CPU, không chạy lại inference/validation hoặc train để tạo số mới.
File gốc trong Downloads và checkpoint app được giữ nguyên.

## Cách dùng kết quả cho bước tiếp

- Giữ các model và báo cáo này làm đối chứng OBB. Không có bằng chứng để triển khai S1/S2.
- V1 hiện là mốc tốt nhất **trong bảng cùng giao thức này**, không phải khẳng định tốt nhất
  trên mọi kích thước hoặc mọi bộ dữ liệu. Không so trực tiếp0.7375 với0.7498 của báo cáo V1 cũ
  hay0.7587 test box thẳng của B, vì khác giao thức/tập nhãn/loại khung.
- Hướng segmentation cần mask riêng từng quả cho cả ba loại, kiểm tra nhãn SAM đề xuất,
  giữ tập ảnh nguồn độc lập. SAM/segmentation không tự bảo đảm tách được từng quả.
- Đánh giá segmentation bằng box mAP và mask mAP riêng; so với OBB bằng sai số đếm
  trên cùng ảnh có số đếm đã duyệt. Không dùng mask mAP mới để tuyên bố vượt mAP OBB cũ.

Chỉ có hai checkpoint và JSON tổng hợp được gửi. Chưa có `selection.json` với sáu phép đo
anchors, lịch sử train đầy đủ, JSON box từng ảnh hoặc previews của lượt này;
chưa thể xác nhận độc lập lựa chọn giữa640/1024 hoặc giải thích nguyên nhân từng lỗi ảnh.
