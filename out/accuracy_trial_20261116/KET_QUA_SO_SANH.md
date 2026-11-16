# So sánh detector mới ngày 16/11/2026

Model mới: `exports/det_obb_trial_v1.pt`, SHA256
`750cadadaf9b8c80354d7c22aba332794300f75b54d295bc71a02faf79a83095`.
Khớp chính xác candidate trong `comparison_val.json` người dùng gửi.
Baseline: `exports/det_obb_s.pt`, SHA256
`e2ae26714bdcb883d4650fdc39d4843b221fdb29b2829681cbcddc8b2af2643e`.

## Cách đo

Chỉ inference 4 ảnh validation bổ sung, mỗi model ở imgsz 640 và 1280.
CPU 2 luồng; FP32; conf 0.40; NMS IoU 0.70; agnostic_nms=True.
Lưu thêm proposal từ conf 0.10 để chẩn đoán, không chọn ngưỡng mới từ 4 ảnh.
Ảnh và nhãn trích đúng từ ZIP thí nghiệm, kiểm tra SHA256 trước khi chạy.
Không train tại máy, không thay checkpoint gốc hoặc cấu hình mặc định của app.
Không chạy CLIP/teacher/classifier: đây chưa phải đánh giá toàn pipeline.

Ở 640, số đếm của cả hai model trên cả 4 ảnh khớp kết quả Colab đã gửi.

## Số đếm tại conf 0.40

| Ảnh | Nhãn đã duyệt | Cũ 640 | Mới 640 | Cũ 1280 | Mới 1280 |
|---|---:|---:|---:|---:|---:|
| 1: mbk__banana_42 | 1 | 2 | 3 | 2 | 2 |
| 2: rf_v7__b_f054 | 4 | 2 | 2 | 1 | 4 |
| 3: rf_v7__b_f064 | 4 | 1 | 1 | 1 | 4 |
| 4: rf_v7__b_f072 | 1 | 1 | 1 | 2 | 2 |
| MAE (quả/ảnh) | | 1.50 | 1.75 | 2.00 | 0.50 |
| Đếm đúng hoàn toàn | | 1/4 | 1/4 | 0/4 | 2/4 |

Ở 640, box model mới vẫn gộp nhiều quả trên hai ảnh nải chuối.
Ở 1280, model mới có 4 box trên mỗi ảnh nải. Với probIoU >= 0.50,
có thể ghép một-một đầy đủ 4 box với 4 nhãn trên từng ảnh này:
không chỉ trùng tổng số đếm. Vị trí/hướng khung vẫn chưa thật sát mọi quả cong.
Hai ảnh chuối bóc vỏ vẫn có box thừa, tách bộ phận của cùng một vật thể.
Hạ conf xuống 0.25 trên candidate 1280 làm số box bốn ảnh thành 4/9/5/3;
không phải cách giải quyết chung cho các ảnh này.

## Kết luận và giới hạn

Candidate có ích trên hai ảnh nhiều chuối khi chạy imgsz1280, dù kết quả
640 không cho thấy cải thiện đếm. Đáng thử có kiểm soát ở 1280, giữ baseline
để đối chiếu. Chưa đủ căn cứ thay mặc định hay khẳng định tốt hơn trên mọi ảnh.

Chỉ có 4 ảnh chuối phát triển; không phải test độc lập, chưa có bằng chứng
về nền báo sai/nho/táo/cam ở 1280 hoặc tác động của CLIP/teacher.
Không train thêm hay đổi ngưỡng chỉ để khớp 4 ảnh này.

## Hình và dữ liệu

- `comparison_750cadad/overview.png`: nhãn / cũ / mới ở 640.
- `comparison_750cadad_1280/overview.png`: nhãn / cũ / mới ở 1280.
- Mỗi thư mục có `scene_1.png` đến `scene_4.png`, `diagnostics.json`
  (tọa độ, confidence, probIoU, số đếm theo ngưỡng) và bản sao JSON Colab.
- Script tái lập: `compare_reviewed.py`; thêm `--imgsz 1280` cho lần đo thứ hai.
  Script từ chối ghi đè thư mục kết quả đã tồn tại.
