# Kế hoạch fine-tune để tăng mAP50–95 cho chuối, táo, cam

Ngày lập: 17/11/2026. Đây là kế hoạch thí nghiệm; chưa thực hiện các lượt train bên dưới.
Không có cấu hình bảo đảm đạt một mức điểm cố định.

**Cập nhật triển khai theo yêu cầu chạy ngay:** đã chuẩn bị gói Colab S1/S2 tại
[`out/map95_finetune_20261117/BAT_DAU.md`](../out/map95_finetune_20261117/BAT_DAU.md).
Gói gộp bước đo/chọn checkpoint và kích thước vào ACTION='anchors', sau đó smoke → S1 → S2 → compare.
Dùng nhãn train/val hiện có để thực hiện trước phép thử hyperparameter có kiểm soát;
bước sửa nhãn dưới đây vẫn cần nếu muốn cải thiện đáng tin cậy khả năng khoanh từng quả.
Chưa tạo benchmark sạch mới hoặc chạy GPU; không coi kết quả lượt này là bằng chứng đã sửa nhãn.

## 1. Điểm xuất phát và mục tiêu

| Báo cáo đã nhận | imgsz | baseline mAP50–95 | Ứng viên mAP50–95 |
|---|---:|---:|---:|
| V1 | 640 | 0,7418 | V1: 0,7498 |
| V2 | 1280 | 0,3014 | V2: 0,5385 |

Hai hàng khác giao thức, không dùng để kết luận V2 kém V1 bao nhiêu.
Trong báo cáo V2, cả ba model được đo cùng giao thức tại 1280 nên có thể so với nhau.
Nguồn: `reports/detection_trials/README.md` và các JSON liên kết tại đó.

Mục tiêu chính là mAP50–95 của **detector OBB**, kèm AP50–95 riêng từng loại.
Điểm này đòi hỏi đúng loại và khung khớp nhãn ở nhiều mức chồng lấp; không phải tỉ lệ đếm đúng.
Ultralytics đo chồng lấp OBB bằng probIoU. Đếm từng quả, box trùng và ảnh nền báo sai
được theo dõi riêng để tránh tăng điểm tổng nhưng làm ứng dụng kém đi.

## 2. Chốt phép đo trước khi train tiếp

Chạy notebook sẵn có:

```text
out/all_fruits_v2_20261116/kiem_tra_kich_thuoc_all3_colab.ipynb
```

Notebook chỉ đánh giá baseline, V1, V2 ở 640 và 1024, không train.
Giữ `BATCH=2` giữa các phép đo mới, dùng T4 GPU; nếu phải giảm batch thì đo lại các model cùng batch.
Kết quả nằm trên Drive:

```text
MyDrive/fruit_all3_v2_20261116/runs/all3_balanced_v2/
  resolution_check_d0abaa79_batch2/resolution_comparison.json
```

Chưa có kết quả notebook này trong các báo cáo đã nhận tính đến lúc lập kế hoạch.
Không sửa `EVAL_IMGSZ` của runner cũ rồi ghi đè kết quả cũ.

Sau bước này, chọn một kích thước chính 640 hoặc 1024 để dùng xuyên suốt các lượt mới.
1280 chỉ thử thêm nếu validation cho thấy lợi ích. Ảnh lớn hơn không mặc nhiên tăng AP.
Chọn checkpoint khởi tạo bằng validation, ghi rõ tên file và SHA256; không mặc định V2 tốt nhất.

Giao thức chính cần ghi trong mỗi báo cáo:

- Ultralytics 8.4.180; cùng phiên bản thư viện giữa các lượt.
- Cùng danh sách ảnh/nhãn và hash manifest; cùng imgsz, batch, `rect`, FP32/FP16.
- AP: `conf=0.001`, `iou=0.7`, `max_det=300`, `agnostic_nms=True`.
- Đếm và ảnh nền: `conf=0.4`, NMS IoU 0.7; tách khỏi bảng AP.
- Giữ CLIP, teacher và classifier tươi/hỏng ngoài phép đo detector.

Đây là giao thức cho nhóm thí nghiệm mới, không phải tuyên bố mọi báo cáo lịch sử đã dùng các cờ này.

## 3. Sửa dữ liệu ảnh hưởng trực tiếp đến định vị

Đã thấy nhãn một box bao bốn quả chuối, trong khi biến thể cùng ảnh nguồn lại có bốn box riêng.
Đây là mục tiêu học và đánh giá mâu thuẫn. Tăng epoch không giải quyết được mâu thuẫn nhãn.

Tạo bản dữ liệu mới có phiên bản, giữ nguyên val/test gốc. Quy tắc dùng chung cho cả ba loại:

1. Mỗi quả phân biệt được là một instance; không dùng box cụm/nải thay cho box từng quả.
2. OBB ôm phần quả theo cùng một quy tắc về cuống và phần bị che. Với quả bị che,
   quy định trước dùng phần nhìn thấy hay ước lượng toàn quả; không trộn hai cách tùy ảnh.
3. Gán đủ các quả mục tiêu nhìn thấy và xác định được. Cảnh quá mơ hồ cần cách ly để duyệt,
   không tự biến thành ảnh nền hoặc bỏ vài quả rồi dùng làm nhãn đầy đủ.
4. Nho/lê/vật khác không mang nhãn táo. Ảnh hỗn hợp vẫn phải có nhãn cho chuối/táo/cam hiện diện.
5. Chia tập theo ảnh nguồn, video hoặc phiên chụp trước khi tạo ảnh tăng cường.
   Không đưa các biến thể của cùng nguồn vào cả train và val/test.

Để giảm công duyệt, bắt đầu bằng **một đợt 60 ảnh khó**: khoảng 20 cảnh có mỗi loại quả,
có thể có nhiều loại trong một cảnh. Phân bố giữa box gộp, box lệch, trùng, bỏ sót và sai loại.
Model chỉ đề xuất nhãn; người duyệt xác nhận và sửa. Sau mỗi lượt mới, chọn tiếp ảnh lỗi thay vì
vẽ lại hàng nghìn ảnh ngay từ đầu. Số ảnh này là phạm vi đợt đầu, không bảo đảm đủ để tăng điểm.

Validation phải được kiểm tra cả ảnh dễ và ảnh khó, không chỉ sửa các ảnh model đang làm sai.
Nếu sửa nhãn validation, đặt tên benchmark mới và đo lại cả ba checkpoint cũ trên benchmark đó.
**Điểm trên nhãn mới không phải mức tăng trực tiếp so với 0,7498 trên nhãn cũ.**

Tạo thêm test thực tế độc lập theo nguồn chụp nếu muốn kết luận khả năng tổng quát hóa.
Các model cũ có thể đã thấy nguồn dữ liệu lịch sử; loại trùng ở lượt train mới không xóa lịch sử đó.
Test cũ đã được xem nhiều lần nên chỉ còn thích hợp làm bộ kiểm tra hồi quy.

Train ưu tiên ảnh thực đã duyệt, gồm cả quả riêng, nhiều quả, che khuất, táo xanh/đỏ,
cam/quýt theo định nghĩa lớp đã thống nhất, chuối nhiều hướng và quả tươi/hỏng.
Giữ ảnh ghép và ảnh nền ở mức có kiểm soát, báo cáo số ảnh và số box từng loại.
Không coi tỉ lệ 654 ảnh thực/60 ảnh ghép/93 ảnh nền của V2 là công thức tối ưu đã chứng minh.

## 4. Các lượt fine-tune có giới hạn

Chỉ bắt đầu khi bước 2 hoàn tất và dữ liệu bước 3 đã được chốt.
Đo lại checkpoint khởi tạo trên benchmark mới để có mốc công bằng.
Tất cả lượt S bắt đầu từ **cùng checkpoint**, cùng dữ liệu, seed và cấu hình đo.

| Lượt | Model/khởi tạo | Thay đổi chính | Ngân sách tối đa |
|---|---|---|---:|
| S1 | YOLO11s-OBB checkpoint quả được chọn | AdamW, lr0=0.0001 | 60 epoch, patience=12 |
| S2 | Giống S1 | Chỉ đổi lr0=0.0003 | 60 epoch, patience=12 |
| M1, có điều kiện | `yolo11m-obb.pt` pretrained chính thức | Thử sức chứa lớn hơn | 80 epoch, patience=15 |
| P1, có điều kiện | Best của lượt được chọn | Fine-tune nhẹ để kiểm tra định vị | 15 epoch, patience=5 |

Các giá trị learning rate là điểm thử có cơ sở để so sánh, chưa phải giá trị tối ưu đã đo.
M1 khởi tạo khác S1/S2 nên là phép thử một phương án tổng thể, không tách riêng được tác động
của kích thước model. Không tải checkpoint s vào kiến trúc m rồi giả định đã chuyển đầy đủ.
YOLO11m-OBB pretrained chưa phải model ba loại quả; phải train và đo lại.

Cấu hình chung khởi đầu cho S1/S2:

```yaml
optimizer: AdamW
epochs: 60
patience: 12
lr0: 0.0001  # S2: 0.0003
lrf: 0.05
cos_lr: true
warmup_epochs: 3
seed: 42
imgsz: 640  # thay bằng 1024 nếu được chọn ở bước 2; cố định giữa S1/S2
batch: 2
device: 0
amp: true
workers: 2
cache: false
degrees: 10
fliplr: 0.5
flipud: 0.1
mosaic: 0.2
close_mosaic: 10
mixup: 0.0
box: 7.5
cls: 0.5
dfl: 1.5
```

Giữ các tham số khác giống nhau và lưu toàn bộ `args.yaml`, phiên bản môi trường,
manifest dữ liệu, checkpoint đầu vào, `results.csv` và trọng số. Không chỉ lưu những cờ liệt kê trên.
Đặt AdamW rõ ràng vì `optimizer=auto` có thể tự chọn optimizer và learning rate.
Giữ trọng số loss mặc định trong hai lượt đầu để biết khác biệt có đến từ learning rate hay không.

Không dùng epoch 16/30 để kết luận chắc chắn một lượt sẽ thất bại; xem đường validation,
best checkpoint và dấu hiệu overfit. Patience tính theo số epoch không cải thiện fitness.
Trong bản Ultralytics 8.4.180 cài tại repo, fitness dùng mAP50–95; vẫn đối chiếu `best.pt`
với báo cáo validation cuối dưới giao thức đã chốt.

Chỉ chạy M1 nếu nhãn đã ổn và S1/S2 vẫn còn lỗi định vị/bỏ sót đáng kể.
Khởi đầu AdamW lr0=0.0003, warmup=5, giữ cách tăng cường như trên;
đây cũng là cấu hình cần đo, không bảo đảm tốt hơn S.
Smoke trên T4 trước; batch 1–2 tùy VRAM. Nếu đổi batch khi so model,
ghi nhận đây là khác biệt huấn luyện; đánh giá cuối vẫn dùng cùng giao thức.

P1 là thử nghiệm tùy chọn: mở một run mới từ best của model được chọn,
lr0=0.00003, warmup=1, mosaic=0, mixup=0, degrees=5, giữ imgsz và dữ liệu.
So lại với checkpoint trước P1; nếu giảm AP thì giữ checkpoint cũ.
Không bật `resume=True` để thay recipe: resume phục hồi lượt bị ngắt,
còn fine-tune theo recipe mới cần NAME mới. Checkpoint đã bỏ optimizer vẫn có thể
làm trọng số khởi tạo, nhưng không phục hồi đầy đủ trạng thái train cũ.

## 5. Điều kiện chọn model và dừng thử

- Chỉ tiêu chính: mAP50–95 chung trên validation đã chốt; báo cáo AP50–95 từng loại.
- Ghi thêm AP75 và AP ở ngưỡng cao nếu có để xem chất lượng định vị;
  đếm đúng, MAE và ảnh nền báo sai là các kiểm tra bổ sung.
- Xem ảnh có box gộp/trùng; không nhận model chỉ vì điểm tổng tăng nhưng một loại suy giảm mạnh.
  Báo cáo rõ đánh đổi nếu phải chọn giữa các mục tiêu.
- Có thể lấy tăng 0,01 tuyệt đối làm mốc thực dụng để ưu tiên ứng viên
  (ví dụ 0,75 → 0,76 là một điểm phần trăm), không coi đó là bằng chứng thống kê.
- Nếu mức tăng nhỏ, chạy thêm seed 43 và 44 cho hai phương án cuối khi đủ GPU;
  báo cáo trung bình/độ phân tán, không chỉ chọn seed có điểm cao nhất.
- Sau S1/S2 nếu vẫn kém mốc và lỗi chủ yếu là nhãn, quay lại duyệt dữ liệu;
  không tự chạy thêm hàng loạt cấu hình hoặc tăng model lên l/x.
- Đánh giá test độc lập sau khi chốt toàn bộ cấu hình; không dùng test để chỉnh epoch,
  learning rate, kích thước ảnh hoặc NMS.

Không đặt trước cam kết 0,85/0,90. Trước tiên phải tăng được điểm trên cùng bộ nhãn sạch,
đồng thời quan sát được box chính xác hơn trên ảnh thực chưa thấy.

## 6. Việc cần làm ngay và phân công

1. Người dùng chạy notebook kiểm tra kích thước ở bước 2 trên Colab T4, gửi
   `resolution_comparison.json`. Không cần train lại V1/V2 lúc này.
2. Tiếp theo chuẩn bị bộ ảnh/nhãn ưu tiên để duyệt và benchmark có phiên bản.
   Phần cần người dùng là xác nhận từng quả, đúng lớp và quy tắc phần bị che.
3. Sau khi dữ liệu được xác nhận, mới đóng gói runner/notebook S1/S2 có log Drive,
   tự tiếp tục khi ngắt, khóa giao thức và bảo vệ kết quả cũ. Bảng cấu hình trên là kế hoạch,
   chưa phải runner mới có thể thay trực tiếp vào notebook cũ.
4. GPU dùng trên Colab; lưu last/best và log lên Drive. Không chuyển train sang CPU laptop
   khi hết hạn mức GPU. Không thay model mặc định của app trước khi đánh giá xong.

## Nguồn kỹ thuật

- Ultralytics Train: https://docs.ultralytics.com/modes/train/
- YOLO11 và các checkpoint OBB pretrained: https://docs.ultralytics.com/models/yolo11/
- OBB validation: https://docs.ultralytics.com/reference/models/yolo/obb/val/
- Giao thức và kết quả của repo: `reports/detection_trials/README.md`.
- Fitness tại bản cài local: `.venv/Lib/site-packages/ultralytics/utils/metrics.py`, `Metric.fitness`.
