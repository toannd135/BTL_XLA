# Hướng tối ưu chung chuối, táo, cam

Chọn một lượt thử có giới hạn: YOLO11s-OBB hiện có + dữ liệu cân bằng ba loại + fine-tune 1024;
đánh giá tại 1280 như cấu hình detector của app. Giữ classifier tươi/hỏng hiện tại.
Đây là ứng viên cần đo, chưa được chứng minh là cấu hình tối ưu toàn cục hay bảo đảm vượt 0,76.

## Gói đã chuẩn bị

- `fruit_all3_v2_20261116.zip`: khoảng 167 MiB, tự chứa ảnh/nhãn/models/scripts.
- `finetune_all3_v2_colab.ipynb`: notebook mới. Không dùng lại notebook accuracy_trial/rotation cũ cho lượt này.
- Không cần clone GitHub, token hoặc tài nguyên trong Drive của bạn khác.
- Không sửa source app, model mặc định, dataset gốc hoặc các kết quả cũ. Toàn bộ nằm dưới `out/`.

Thành phần train:

| Nguồn | Ảnh riêng biệt |
|---|---:|
| Ảnh thực có nhãn quả được giữ lại | 654 |
| Ảnh ghép được chọn có giới hạn | 60 |
| Ảnh nền từ nguồn nền đã có | 93 |

Có 927 lượt ảnh trong danh sách train sau khi lặp có giới hạn.
Tổng box tính theo lượt xuất hiện: **chuối 636, táo 625, cam 713**.
Ảnh thực đã duyệt được ưu tiên; tỉ lệ ảnh ghép được giới hạn theo từng loại.

Các ảnh nền rỗng không rõ độ tin cậy và các nhãn dương tính đáng ngờ ở nguồn vật khác được
cách ly khỏi lượt train mới, không tự sửa thành nhãn đúng. Những frame cùng video/nhóm nguồn
của val/test được loại khỏi train mới. Dữ liệu cũ vẫn cần kiểm tra chất lượng thêm.

## Bạn làm lần lượt

1. Upload **thẳng vào My Drive** file `fruit_all3_v2_20261116.zip`. Giữ nguyên tên, không giải nén.
2. Mở Colab → File → Upload notebook → chọn `finetune_all3_v2_colab.ipynb`.
3. Runtime → Change runtime type → **T4 GPU** → Save.
4. Chạy các ô từ trên xuống. Cấp quyền mount chính Drive vừa upload. Để mặc định:

   ```python
   ACTION = 'smoke'
   NAME = 'all3_balanced_v2'
   EPOCHS = 30
   BATCH = 2
   IMGSZ = 1024
   EVAL_IMGSZ = 1280
   AUTO_RESUME = True
   ```

5. Smoke xong: sửa `ACTION = 'train'`, **chạy lại ô cấu hình mục 3 rồi ô thực thi mục 4**.
   Lượt đầy đủ bắt đầu từ baseline, không bắt đầu từ checkpoint smoke.
6. Train xong: sửa `ACTION = 'val'`, chạy lại ô 3 rồi 4. Không đổi NAME.
7. Gửi file:

   ```text
   My Drive/fruit_all3_v2_20261116/runs/all3_balanced_v2/comparison_val.json
   ```

   Bảng dễ đọc nằm ngay cạnh: `comparison_val.md`.
   Ảnh nhãn/dự đoán nằm ở `evaluation/val/{baseline,trial_v1,trial_v2}_previews/`.

Không chạy test trước khi xem validation. Không tự thay model app sau khi train xong.

## Nếu Colab ngắt hoặc có lỗi

- Runtime mới: chọn lại GPU, chạy lại setup, giữ NAME/EPOCHS/IMGSZ; đặt ACTION='train', AUTO_RESUME=True.
- Nếu last.pt còn optimizer: runner tự tiếp tục từ epoch đã lưu, không chạy lại từ đầu.
- Nếu lượt đã hoàn tất: runner báo xong và bỏ qua, không gây lỗi Run exists; chuyển sang val.
- Nếu CUDA out of memory: giảm BATCH xuống 1. Nếu đã có checkpoint, giữ NAME để tiếp tục.
- Nếu thư mục đã tạo nhưng chưa kịp ghi last.pt: chọn NAME mới, ví dụ `all3_balanced_v2_b1`.
  Giữ nguyên thư mục cũ để xem lỗi; không xóa tùy tiện.
- Nếu GPU hết hạn mức: dừng ở checkpoint trên Drive; không train bằng CPU laptop.
- Log lưu tại `My Drive/fruit_all3_v2_20261116/logs/`. Khi lỗi, gửi đoạn 60 dòng cuối notebook in ra.

## Đọc kết quả và bước sau

Val so cả 3 model trên cùng 297 ảnh, cùng imgsz1280 và ngưỡng:

- `baseline`: model app cũ.
- `trial_v1`: model bạn vừa train trước đó.
- `trial_v2`: ứng viên cân bằng cả 3 loại + train 1024.

AP đo ở conf0.001 / NMS0.7. Precision/Recall/F1 và đếm đo ở conf0.4 / NMS0.7.
Ghép OBB một-một, cùng loại quả, probIoU >=0.5; box trùng không được tính thành hai quả đúng.
Đây là matching riêng cho bảng operating point, không thay cách Ultralytics tính AP.
MAE thấp hơn tốt hơn; xem riêng từng loại và ảnh sai, không chỉ số mAP tổng.
Không đối chiếu trực tiếp AP1280 với báo cáo640 cũ.

Sau khi chốt model bằng val, đặt CHECKPOINT thành tên đã chọn và ACTION='test'.
Test đo model đã chọn với baseline trên 373 ảnh test cũ và 71 ảnh nền cũ.
71 ảnh nền có thể trùng một phần với nền trong test373, không cộng thành 444 ảnh độc lập.
Giữ nguyên cấu hình sau khi xem test, không dùng test để tìm ngưỡng.

File candidate: `runs/all3_balanced_v2/weights/best.pt`. Chỉ tải/chạy thử sau khi xem so sánh.

## Phần bạn cần bổ sung để đánh giá thực tế tốt hơn

Lượt hiện tại chạy được bằng dữ liệu sẵn có. Để đánh giá triển khai đáng tin hơn, thu thêm một đợt
ảnh thực mới, ưu tiên khoảng 15–20 ảnh cho mỗi nhóm sau (mục tiêu khởi đầu, không phải bảo đảm accuracy):

- Chuối riêng/nải, táo đỏ/xanh, cam: mỗi nhóm có ảnh ít quả và nhiều quả, sáng/tối, che khuất.
- Ảnh nhiều loại quả chung khung.
- Nho, lê, cà chua, bóng, đồ vật tròn: cả ảnh chỉ có vật khác và ảnh lẫn quả mục tiêu.

Chia nhóm theo cảnh/chùm ảnh/video **trước khi** gán nhãn, giữ một nhóm chụp khác làm kiểm tra cuối.
Không lấy các ảnh đã liên tục dùng chẩn đoán làm bộ test độc lập.
Bạn không cần tự gán lại toàn bộ dữ liệu cũ: có thể dùng nhãn nháp và chỉ sửa các ca sai,
nhưng phải xác nhận đủ mọi quả mục tiêu trước khi dùng ảnh vào train.

Giữ 3 nhãn `banana/apple/orange`; chưa thêm nhãn nho nếu phạm vi vẫn chỉ đếm ba loại này.
Ảnh chỉ có nho có thể là nền sau khi xác nhận không có quả mục tiêu; ảnh nho lẫn táo phải gán đủ táo.
Bộ gói hiện tại **chưa có nguồn nho mới đã xác nhận**, không cam kết đã giải quyết lỗi nho.

## Giới hạn và kiểm tra đã làm

- Nhãn val/test cũ còn box nải; val đã duyệt bổ sung chỉ có chuối. Không gọi đây là test độc lập đủ ba loại.
- Model khởi tạo đã được train bằng dữ liệu lịch sử; loại video trùng khỏi lượt mới không xóa lịch sử này.
- Báo cáo chỉ đo detector. CLIP/teacher/classifier và tốc độ app CPU phải kiểm tra riêng sau đó.
- Đã kiểm tra CRC/hash toàn bộ ZIP, đường dẫn ảnh/nhãn, nhóm nguồn và hash train/eval, cân bằng box,
  code notebook, matching box trùng/nhầm loại, OBB float32, và chặn train CPU.
- Chưa thực thi train GPU/Colab tại máy này. App/model gốc giữ nguyên.

Tham khảo chính thức:
[Thiết lập train](https://docs.ultralytics.com/modes/train/),
[Thu thập và gán nhãn dữ liệu](https://docs.ultralytics.com/guides/data-collection-and-annotation/).
