# Sửa mục tiêu: nhận diện và đếm từng quả

## Trạng thái thực tế

Model hiện tại chưa đạt yêu cầu đếm riêng từng quả ở cảnh chuối chồng nhau. Bộ lọc NMS/box bao cụm chỉ xử lý một số giả thuyết trùng; nó không tạo được nhãn từng quả mà detector chưa học đúng. Lượt rotation_v1 đã giảm chất lượng trên val/test và không được nhận vào app.

Đếm đúng tổng số chưa đủ: model có thể vừa đếm trùng một quả vừa bỏ sót một quả khác. Cần kiểm tra tương ứng một–một giữa nhãn và dự đoán, cùng loại quả và đủ độ khớp vị trí.

Khung chữ nhật của một quả chuối cong vẫn có thể bao phần ảnh của quả bên cạnh. Mục tiêu là mỗi khung đại diện đúng một quả, không yêu cầu mọi khung chữ nhật tách rời. Muốn viền bám sát đúng phần thịt/vỏ của từng quả cần nhãn và mô hình instance segmentation riêng; công cụ dưới đây chuẩn hóa nhãn box cho detector hiện tại.

## Những gì đã kiểm tra trong dữ liệu

| Tập gốc | Số ảnh | Ảnh có nhãn quả | Nhãn rỗng | Ảnh có >=2 đối tượng trong nhãn | Ảnh bị cắm cờ trong audit cũ |
|---|---:|---:|---:|---:|---:|
| train | 2.365 | 886 | 1.479 | 217 | 86 |
| val | 301 | 301 | 0 | 36 | 15 |
| test | 373 | 274 | 99 | 45 | 26 |

Đây là số lượng theo nhãn, chưa phải số quả thật đã xác minh. Nhãn rỗng có thể là ảnh nền hợp lệ hoặc ảnh bị thiếu nhãn; không tự kết luận tất cả đều sai. Các cờ bunch/nested/missing cũng chưa phải kết luận: chúng dựa một phần vào dự đoán model và có thể nhầm box chuối cong với box cả nải.

Đã xem contact sheet nhãn gốc tại `out/instance_audit/train_gt_examples.jpg`: có trường hợp gán box bao cụm và box quanh cuống. Cần kiểm tra/sửa theo ảnh, không tự chia đều box hoặc lấy dự đoán model làm nhãn thật.

## Bộ duyệt đã chuẩn bị

- Mở `out/instance_review_v1/review.html` bằng Chrome/Edge. Không cần backend hay GPU; ảnh và nhãn nằm trong bản sao local.
- Có 3.046 ảnh: train2.365, val301, test373 và 7 ảnh diagnostic bạn đang thử. Các ảnh có cờ/ảnh nhiều đối tượng được ưu tiên. Không cần duyệt hết cùng lúc; phần xuất dữ liệu chỉ dùng các ảnh thực sự đã xác nhận.
- `out/instance_review_v1/assistant_draft_reviews.json` chứa nhãn nháp vẽ theo từng quả cho `bananas.webp`, dự kiến 7 quả nhìn thấy. Đây là nhãn tham chiếu do trợ lý vẽ, không phải dự đoán mới của detector, chưa được đánh dấu approved. Hình minh họa: `out/instance_audit/bananas_individual_draft.jpg`.
- Ảnh diagnostic luôn bị loại khỏi tập train khi build dataset, kể cả nếu được duyệt. Các ảnh này đã dùng trong quá trình phát triển nên không phải bộ test độc lập cuối cùng.

Mở công cụ từ PowerShell nếu liên kết HTML mở bằng trình soạn thảo:

```powershell
cd D:\privateProject\xuLyAnh\BTL_XLA
Invoke-Item .\out\instance_review_v1\review.html
```

## Cách duyệt từng quả

1. Với ảnh đang thử, chọn Tập = **Ảnh bạn đang thử**. Bấm **Nạp bản duyệt JSON**, chọn `assistant_draft_reviews.json` để xem nhãn nháp 7 quả chuối. Kiểm tra lại từng khung trước khi xác nhận.
2. Chọn Train/Validation để sửa dữ liệu. Bật **Chỉ ảnh có cờ** để ưu tiên; cờ không phải yêu cầu phải xóa box. Nhìn toàn bộ ảnh và đếm số quả nhìn thấy, gồm quả ở mép ảnh nếu xác định được là một quả riêng.
3. Chọn khung bằng số trong bảng để đối chiếu. Xóa box bao cả nải/cuống bị gán thành quả. Chọn **Vẽ box mới**, chọn Chuối/Táo/Cam rồi kéo quanh phần nhìn thấy của từng quả. Một quả chỉ có một box; không đoán quả bị che hoàn toàn. Ảnh không xác định được số quả hoặc chưa thống nhất cách đếm quả cắt miếng: chọn **Loại ảnh**.
4. Nhập **Số quả đếm bằng mắt**. Bấm **Đã kiểm tra từng quả** sau khi kiểm tra từng box. Công cụ yêu cầu số này bằng số box; điều kiện này là kiểm tra nhất quán, không tự chứng minh nhãn đúng. Chỉ nhập 0 cho ảnh thật sự không có quả.
5. Duyệt cả chuối, táo, cam và các cảnh cần dùng: đơn lẻ, sát nhau, nhiều quả, che khuất, nhiều loại chung khung; có ảnh nền đã xác minh. Giữ nguyên vai trò train/val/test của nguồn, không chuyển các khung gần giống sang tập khác.
6. Bấm **Tải bản duyệt JSON** để sao lưu ra file `instance_reviews.json`. Dữ liệu tạm trong trình duyệt có thể bị mất nếu xóa dữ liệu trình duyệt; nạp JSON để tiếp tục. Mọi lần sửa box/count sẽ đưa ảnh về trạng thái chưa duyệt.

## Xuất bản dữ liệu đã duyệt

Chép file JSON đã tải vào thư mục dự án rồi chạy:

```powershell
.\.venv\Scripts\python.exe tools\instance_review.py build --reviews .\instance_reviews.json
```

Đầu ra mặc định: `out/instance_clean_v1/`, gồm `train`, `val`, `test` nếu có nhãn test đã duyệt, `data.yaml` và `reviewed_manifest.json`. Thư mục đầu ra phải mới để tránh ghi đè. Lưu JSON review bên ngoài `out/` hoặc sao lưu riêng trước khi dọn kết quả.

Các kiểm tra khi build:

- Chỉ ảnh approved được xuất; ảnh pending/excluded không bị biến thành ảnh nền. Diagnostic không được đưa vào train/val/test đã xuất.
- Kiểm tra số quả bằng số box, lớp, tọa độ, box trùng hoàn toàn, hash ảnh và bộ review.
- Giữ split gốc và từ chối trùng nhóm nguồn/ảnh giữa train/val/test.
- Train và val phải có cả ba lớp. Đây chỉ là điều kiện kỹ thuật tối thiểu, không bảo đảm dữ liệu đã đủ đa dạng hoặc đủ lớn để train tốt.
- Không sửa ảnh/nhãn gốc tại `local_dataset/det`, không sửa weights đang chạy.

Sau khi có dữ liệu thực sự được duyệt, kiểm tra phân bố/độ đa dạng và bộ validation nhiều quả trước khi tạo lượt fine-tune mới trên Colab. Chưa có model mới đã được huấn luyện từ bộ nhãn này.

## Đo đúng yêu cầu đếm từng quả

Sau khi duyệt ảnh diagnostic hoặc validation, có thể đo model hiện tại trên CPU:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_instance_count.py --reviews .\instance_reviews.json
```

Công cụ chỉ đánh giá ảnh approved trong các vai trò diagnostic/val mặc định, không đánh giá train. Nó báo:

- Count MAE: sai lệch tuyệt đối trung bình về số quả, mục tiêu 0 trên bộ kiểm thử đã chốt.
- Exact count rate: tỷ lệ ảnh có tổng số đúng; báo thêm sai số theo từng loại quả.
- TP/FP/FN và tỷ lệ ảnh không thừa/thiếu instance ở IoU0.50 và0.75, ghép một–một cùng lớp. Box to gom cụm, box trùng và quả bị bỏ sót có thể làm giảm chỉ số này dù tổng số tình cờ đúng.
- Cấu hình, hash model/review/hậu xử lý và phiên bản thư viện để đối chiếu cùng một phép đo. Đây không phải phép đo AP/mAP.

Một bộ kiểm thử hữu hạn đạt tốt không bảo đảm mọi ảnh thực tế đều đếm đúng. Tuyên bố độ chính xác phải gắn với bộ ảnh đã duyệt, phạm vi cảnh và điều kiện đo cụ thể.

## Kiểm tra công cụ

13 test Python đã qua cho xuất nhãn, nguồn dữ liệu, không dùng ảnh chưa duyệt làm nền, loại diagnostic khỏi train, số quả, box trùng/sai, ba lớp, rò rỉ split và matching đếm instance. Đã kiểm tra bằng trình duyệt Edge headless: tải ảnh, lọc vai trò, kéo vẽ, chặn count không khớp, xác nhận, xóa/hoàn tác, loại ảnh và xuất JSON. Bản smoke dùng profile riêng và xóa các nhãn giả của kiểm thử; không duyệt nhãn thật thay người dùng.
