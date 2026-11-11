# Chạy fine-tune Detection từ model C trên Colab

## File đã chuẩn bị trên máy bạn

- Notebook: `BTL_XLA/finetune_C_colab.ipynb`.
- Gói input: `D:\privateProject\xuLyAnh\detection_C_colab_input.zip` (khoảng 296 MB).
- Báo cáo gói input: `D:\privateProject\xuLyAnh\detection_C_colab_input.json`.

Gói ZIP đã có đúng cấu trúc tương đối: model trong `exports/`, dataset trong `local_dataset/det/`, ảnh nền trong `own_neg/`, script trong `work/exp/`, danh sách quarantine trong `work/audit/`. Không cần tự chuyển các thư mục ngoài `BTL_XLA`, và không cần clone Git trên Colab.

Đây là hướng dẫn hiện tại theo yêu cầu chạy cách của bạn bạn. Không dùng notebook cũ `detection_finetune_colab.ipynb` cho lượt này.

## 1. Đã đủ dữ liệu để chạy gì?

Đã đủ để thực hiện một lượt fine-tune Detection theo công thức C, bắt đầu từ **model C hiện tại** trong `exports/det_finetune_best.pt`. Chỉ huấn luyện detector, giữ classifier như cũ.

Phân biệt: lần C gốc fine-tune từ B; lượt này fine-tune tiếp từ C theo hướng dẫn bạn bạn. Không gọi đây là tái hiện chính xác B → C, không mặc định kết quả sẽ giống hoặc tốt hơn C. Đây cũng chưa phải thí nghiệm D1 với ảnh ghép.

Cấu hình học giữ theo `work/exp/det_neg.py`:

| Tham số | Giá trị |
|---|---|
| Model khởi đầu | `exports/det_finetune_best.pt` (C) |
| Epoch tối đa | 25 |
| Batch | 16; có thể giảm 8 hoặc 4 nếu thiếu VRAM |
| Image size | 640 |
| Optimizer / lr0 | SGD / 0.002 |
| Warmup / patience | 1 / 10 |
| Seed | 42 |
| Ảnh tự thu trong `train_ft` | nhân 4 |
| Ảnh nền train | nhân 2 |
| Own_cam | không thêm vào danh sách train C |

Notebook chỉ thay phần vận hành: chọn GPU rõ ràng, workers=2, tách smoke/main, ghi checkpoint lên Drive và đánh giá `last.pt` đúng hướng dẫn thay vì `best.pt` của đoạn cuối script cũ. Notebook dùng nguyên phần chuẩn bị dữ liệu của `det_neg.py` gốc, không sửa file nguồn.

## 2. Upload đúng hai thứ

1. Trên Google Drive của bạn, tạo thư mục `fruit_C`.
2. Trong đó tạo thư mục `input`.
3. Upload `detection_C_colab_input.zip` vào thư mục `input`.

Kết quả:

```text
MyDrive/
  fruit_C/
    input/
      detection_C_colab_input.zip
```

Chờ Drive upload xong trước khi chạy notebook. Không cần giải nén trên Drive, không cần upload từng ảnh, không cần tài khoản/Kaggle token của bạn bạn.

## 3. Mở notebook và chọn GPU

1. Mở https://colab.research.google.com/.
2. Chọn mở notebook, mục upload, rồi chọn `BTL_XLA/finetune_C_colab.ipynb` trên máy bạn.
3. Mở cài đặt runtime, chọn phần cứng **GPU**, chọn T4 nếu được cấp.
4. Chạy ô code ở phần **1. GPU và thư viện**.

Ô này phải in tên GPU. Nếu dừng ở `Chưa có GPU`, kiểm tra lại runtime hoặc thử lại sau khi có quota. Không chọn CPU để cố chạy lượt train này.

Chỉ notebook tính toán trên GPU của Google; máy cá nhân chỉ cần trình duyệt, mạng và dung lượng tải model về. Không cần cài CUDA ở Windows.

Colab miễn phí có hạn mức động, không bảo đảm luôn được cấp T4 hay thời gian chạy cố định. Không dùng thời gian RTX 3060 của lead làm dự đoán chính xác cho Colab.

## 4. Đọc và kiểm tra input

Chạy ô code ở phần **2. Mount Drive và kiểm tra gói input**:

- Cho phép Colab kết nối Google Drive của bạn.
- Notebook chép ZIP về `/content`, giải nén và kiểm tra hash từng file.
- Nếu báo thiếu ZIP, kiểm tra lại đúng `MyDrive/fruit_C/input/detection_C_colab_input.zip`.

Số lượng dự kiến:

```text
train:     2365 ảnh
val:        301 ảnh
test:       373 ảnh
train_ft:   875 ảnh
ảnh nền:    287 train + 71 giữ riêng
quarantine: 50 ảnh
```

Không lấy 875 ảnh train_ft cộng nguyên vào train: script chỉ chọn 165 ảnh có tên bắt đầu `own-`, nhân 4.

## 5. Chạy thử 1 epoch

Ở phần **3. Cấu hình lượt chạy**, để nguyên:

```python
MODE = "smoke"
RUN_VERSION = "v1"
BATCH = 16
RESUME = False
```

Chạy ô đó, rồi chạy ô **4. Chuẩn bị đúng tập train của script gốc**.

Ô chuẩn bị phải cho ra:

```text
2315 ảnh gốc sau loại quarantine
165 ảnh own × 4
287 ảnh nền × 2
Tổng: 3549 lượt ảnh trong danh sách train
71 ảnh nền giữ riêng để đo
```

Đây là số lượt ảnh sau nhân bản, không phải 3549 ảnh độc lập.

Tiếp tục chạy ô **5. Chạy smoke hoặc train chính**. Chờ hoàn tất 1 epoch. Mục tiêu chỉ là phát hiện lỗi dữ liệu, thiếu RAM/VRAM hoặc đường dẫn; không đánh giá chất lượng bằng model smoke.

Model gốc trong `exports/` được giữ nguyên. Checkpoint smoke ở:

```text
MyDrive/fruit_C/runs/exp/det_C_smoke_v1/weights/
```

Nếu smoke lỗi OOM, xem phần xử lý lỗi bên dưới trước khi chạy lại.

## 6. Chạy fine-tune chính

Smoke hoàn tất thì quay lại ô cấu hình phần 3, đổi duy nhất:

```python
MODE = "train"
RUN_VERSION = "v1"
BATCH = 16
RESUME = False
```

Chạy lại các ô **3 → 4 → 5**. Lượt chính khởi tạo từ C gốc, không từ model smoke.

Notebook chạy tối đa 25 epoch; có thể dừng sớm sau 10 epoch không cải thiện theo cơ chế framework. Đọc thời gian từng epoch thực tế để ước lượng tổng thời gian. Không mở thêm lượt train GPU khác trong cùng runtime.

Checkpoint ghi trực tiếp lên Drive ở:

```text
MyDrive/fruit_C/runs/exp/det_C_repeat_v1/
  weights/
    last.pt
    best.pt
    epoch*.pt       # snapshot định kỳ mỗi 5 epoch
  args.yaml
  results.csv
```

Không phải đợi train xong mới chép model về Drive. Tuy nhiên, nếu runtime ngắt đúng lúc đang ghi checkpoint, file đang ghi vẫn có thể chưa hoàn chỉnh; snapshot định kỳ là phương án dự phòng. Không ngắt Drive giữa lúc train.

## 7. So sánh model gốc với last.pt

Sau khi lượt chính kết thúc, chạy phần **6. So baseline C với last.pt**.

Notebook đo lại cả hai model trên cùng dữ liệu và tạo:

```text
MyDrive/fruit_C/runs/exp/det_C_repeat_v1/comparison.csv
MyDrive/fruit_C/runs/exp/det_C_repeat_v1/comparison.json
```

Cần xem đồng thời:

- mAP50 và mAP50–95: tổng thể có giảm nhiều không.
- AP50 từng loại: đặc biệt chuối có kém đi không.
- Số ảnh nền báo sai trên 71 ảnh: giảm hay tăng.

Script `work/eval_det.py` gốc tính mAP ở confidence thấp mặc định và đo thêm ở 0.25. Notebook ghi rõ cả `mAP50` tại conf=0.001 và `mAP50_c25` tại conf=0.25; không trộn hai cột này để so sánh. NMS IoU dùng 0.7; đo ảnh nền tại conf=0.25. Ngưỡng web 0.40 là cấu hình triển khai riêng.

Không tự động nhận model mới chỉ vì đã train xong. Nếu không cải thiện hoặc đánh đổi không phù hợp thì tiếp tục dùng C gốc. Giữ test để kiểm tra cuối, không dùng các lần xem test để dò hàng loạt cấu hình rồi gọi kết quả đó là đánh giá độc lập.

## 8. Tải model về và chạy trên Windows

Chạy phần **7. Tải model candidate**. Notebook xuất và tải file:

```text
MyDrive/fruit_C/outputs/det_C_repeat_v1/det_C_candidate.pt
```

Đặt file tải về vào `BTL_XLA/exports/`, cạnh model cũ. Đừng ghi đè model C ngay.

Khi đã cài môi trường backend theo README, mở PowerShell:

```powershell
cd D:\privateProject\xuLyAnh\BTL_XLA\backend
$env:MODEL_PATH = "D:\privateProject\xuLyAnh\BTL_XLA\exports\det_C_candidate.pt"
$env:HALF = "0"
uvicorn main:app --port 8000
```

Nếu backend cũ đang chạy, dừng rồi khởi động lại trong terminal có các biến môi trường trên. Classifier mặc định vẫn là `exports/cls_finetune_best.pt`.

Mở frontend theo README ở terminal khác. Máy cá nhân chạy nhận diện CPU được; thử ảnh tĩnh trước vì video/webcam có thể chậm. GPU cloud chỉ cần cho phần train.

## 9. Nếu Colab bị ngắt

1. Kết nối lại GPU, mở lại notebook và chạy phần 1–2.
2. Ở phần 3, đặt `MODE="train"`, giữ `RUN_VERSION="v1"`, đặt `RESUME=True`.
3. Giữ nguyên gói ZIP. Có thể giảm `BATCH` nếu cần.
4. Chạy lại phần 3–5. Phần 4 dựng lại các đường dẫn `/content`; phần 5 tiếp tục từ checkpoint trên Drive.

Chỉ tiếp tục được từ epoch đã lưu hoàn chỉnh; phần epoch chưa lưu phải chạy lại. Nếu `last.pt` ghi dở, có thể chọn một file `epoch*.pt` còn tốt bằng biến `RESUME_CHECKPOINT` ở phần 3.

Không dùng resume cho lượt đã hoàn thành. Muốn mở lượt thử mới, đổi `RUN_VERSION` thành `v2`, để `RESUME=False` và chốt mục đích thí nghiệm trước.

## 10. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| Chưa có GPU | Chọn GPU runtime; nếu hết quota thì thử lại sau |
| Không thấy ZIP | Kiểm tra đúng tên và thư mục Drive; chờ upload hoàn tất |
| CUDA out of memory | Restart runtime; giảm batch 16 → 8 → 4; không đổi model sang YOLO11n |
| Run đã tồn tại nhưng chưa có checkpoint | Lượt lỗi trước khi lưu epoch đầu; đổi RUN_VERSION và chạy mới |
| Run đã tồn tại, có checkpoint và train bị ngắt | Giữ tên run, dùng RESUME=True |
| Resume báo khác input | Dùng lại đúng ZIP, model, phiên bản thư viện và run cũ |
| No module named cv2 | Chạy lại ô cài Ultralytics; nếu vẫn thiếu thì cài opencv-python-headless trong Colab |
| Log báo lỗi nhãn/corrupt | Xem chính xác file được báo; không bỏ qua chỉ để train tiếp |

## 11. Giới hạn nhãn đã kiểm tra

Gói này giữ nguyên dữ liệu của người chia sẻ để bám quy trình. Có nhãn rỗng trong train/test; nhiều ảnh ngoài ba loại quả có thể là nền hợp lệ. Tuy nhiên, đã mở xác nhận ít nhất một ảnh chuối train và một ảnh chuối test có nhãn rỗng. Có ba nhãn train dạng polygon; loader Detection của Ultralytics có thể chuyển sang box.

Các vấn đề này không ngăn bắt đầu chạy kỹ thuật, nhưng ảnh hưởng chất lượng học và độ tin cậy của chỉ số. Không khẳng định dataset sạch hoặc mAP phản ánh đầy đủ thực tế. Nếu sửa nhãn, tạo phiên bản dữ liệu mới và clean test riêng; không tự sửa test gốc rồi so trực tiếp với số cũ.

## 12. Tạo lại gói ZIP khi cần

Nếu dữ liệu thay đổi, trên Windows chạy:

```powershell
cd D:\privateProject\xuLyAnh
python BTL_XLA\tools\prepare_c_colab.py --force
```

Lệnh này không train, không cần GPU và không chuyển dữ liệu nguồn. Nó tạo lại ZIP và báo cáo JSON; upload ZIP mới và mở run mới thay vì resume run dùng dữ liệu cũ.

Notebook đã được kiểm tra JSON/cú pháp, công thức chia dữ liệu và đóng gói cục bộ; chưa chạy GPU hoặc xác nhận kết quả học trong phiên chuẩn bị.

Nguồn đối chiếu: [Colab FAQ](https://research.google.com/colaboratory/faq.html), [Ultralytics train/resume](https://docs.ultralytics.com/modes/train/), cùng các script và cấu hình gốc trong repo.
