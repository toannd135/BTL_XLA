# Fine-tune Detection khi máy cá nhân không có GPU

> **Hướng dẫn hiện tại:** người dùng đã bổ sung model/dữ liệu/script và chọn fine-tune tiếp từ C theo hướng dẫn người chia sẻ. Dùng [`chay-finetune-C-tren-colab.md`](chay-finetune-C-tren-colab.md), notebook `finetune_C_colab.ipynb` và gói `detection_C_colab_input.zip` đã chuẩn bị. Nội dung bên dưới là phương án tổng quát cũ để tham khảo; không dùng notebook tổng quát bên dưới thay cho lượt C hiện tại. Phần so C–D1 trong bàn giao là công việc riêng.

Notebook đi kèm: [`../detection_finetune_colab.ipynb`](../detection_finetune_colab.ipynb).

Hướng thực hiện: chuẩn bị và kiểm tra dữ liệu trên Windows, train trên GPU Colab, tải model về chạy CPU. Chỉ train Detection; giữ nguyên Classification tươi/hỏng. Không cần cài CUDA trên máy cá nhân.

## 1. Những gì cần lấy từ lead

Bản checkout hiện tại không có `exports/`, `local_dataset/` và các script `work/exp/` được nhắc trong bàn giao. Các thư mục model/dataset phần lớn bị `.gitignore` loại khỏi Git. Clone repo chưa đủ để tiếp tục đúng thí nghiệm.

Nhờ lead gửi:

1. **Detector đang được chấp nhận**, file `.pt`: ghi chú 08/11/2026 là C, còn D1 đang thử nghiệm. Cần xác nhận D1 đã được đánh giá và chấp nhận chưa. Không tự lấy một `last.pt` chưa rõ nguồn.
2. **Dataset Detection 3 lớp và cách chia tập hiện tại**, gồm train/val/test cùng nhãn. Nếu có train đã loại 50 ảnh nghi nhãn nải/lồng, lấy bản đã làm sạch đó. Xin danh sách ảnh bị loại nếu chỉ nhận dataset gốc.
3. **Kết quả D1 nếu đã có**: tránh chạy lại một thí nghiệm đã hoàn tất.
4. **Classifier `.pt` hiện dùng** nếu cần chạy đủ web trên máy mình. Không cần classifier để train Detection trong notebook mới.

Có thể nhắn: “Anh gửi em detector C hoặc D1 đã chốt, dataset detect 3 class có split cố định và danh sách quarantine với. Máy em không có GPU, em sẽ fine-tune Detection trên Colab, giữ nguyên classifier.”

Nếu chưa lấy được model cũ, `yolo11s.pt` COCO có thể làm khởi tạo cho một thí nghiệm khác, nhưng không phải tiếp tục từ C/D1. Notebook này chủ động yêu cầu model 3 lớp từ lead để tránh nhầm hai trường hợp.

## 2. Chuẩn bị dữ liệu bổ sung: ưu tiên sửa đúng lỗi hiện tại

Theo `CLAUDE.md`, các lỗi cần giải quyết là bỏ sót táo xanh trong cảnh nhiều loại quả, box chuối thừa và nhận nhầm vật khác thành quả. Train lại đúng dữ liệu cũ không bảo đảm sửa các lỗi này.

Mốc khởi đầu đề xuất, không phải yêu cầu cứng:

| Nhóm ảnh | Số lượng mới để bắt đầu | Nội dung |
|---|---:|---|
| Nhiều loại quả | 150–300 | Chuối + táo xanh/đỏ + cam; thay vị trí, khoảng cách, nền |
| Chuối | 100–200 | Từng quả, nhiều quả gần nhau, che khuất một phần |
| Ảnh nền dễ nhầm | 100–200 | Bóng, cốc, tay, ớt chuông, mặt bàn; xác nhận không có ba loại quả |

Chụp nhiều buổi, nhiều cảnh và nhiều quả thực tế. Đừng lấy hàng trăm frame sát nhau từ một video rồi coi là hàng trăm cảnh độc lập. Lưu thông tin buổi chụp/video nguồn để chia tập.

**Không chỉ dùng công tắc tự thu thập của web:** bộ lọc có thể bỏ chính ảnh mà model đang nhận thiếu/nhầm. Lưu thêm ảnh gốc bằng camera/điện thoại rồi gán nhãn thủ công cho các ca khó. Không có GPU vẫn chụp và gán nhãn được.

### Gán nhãn Detection

Dùng công cụ gán bounding box có hỗ trợ xuất YOLO Detection. Vẽ khung cho **từng quả nhìn thấy và phân biệt được**, giữ quy ước nhất quán với nhóm; không vừa có box cả nải vừa có box từng quả cho cùng vùng.

Mapping bắt buộc:

```text
0 banana
1 apple
2 orange
```

Không thêm lớp fresh/rotten vào detector. Mỗi ảnh phải gán đủ các quả thuộc ba lớp, kể cả quả không phải đối tượng chính. Ảnh không có ba loại quả là ảnh nền, dùng file nhãn rỗng. Không gán quả thật là nền để giảm báo sai.

Định dạng nhãn chuẩn YOLO:

```text
class_id x_center y_center width height
```

Bốn tọa độ đều được chuẩn hóa về 0–1 theo kích thước ảnh. Ví dụ:

```text
1 0.500000 0.500000 0.250000 0.400000
```

Đây là box của một quả táo, không phải tọa độ pixel. Phần mềm gán nhãn thường xuất định dạng này tự động.

**Cảnh báo riêng của repo:**

- `own_cam/` là nhãn **6 lớp**: banana_fresh, banana_rotten, apple_fresh, apple_rotten, orange_fresh, orange_rotten. Chỉ với đúng mapping này, đổi sang Detection bằng `new_id = old_id // 2`. Làm trên bản sao, duyệt nhãn sau chuyển đổi. Không đưa thẳng nhãn 6 lớp vào train 3 lớp.
- `own_mixed/` được bàn giao là nhãn nháp còn sai. Chưa sửa và kiểm tra bằng mắt thì chưa dùng train.
- `own_neg_flagged/` có ảnh bị nghi chứa quả thật; không đưa nguyên thư mục vào ảnh nền.
- Nhãn tự sinh cần được kiểm tra, đặc biệt ở tập val/test; không lấy dự đoán model làm đáp án đánh giá model đó.

### Chia train/val/test và gộp dữ liệu cũ

- Giữ nguyên val/test cũ để đối chiếu. Không lấy ảnh test cũ sang train.
- Chia dữ liệu mới **theo buổi chụp/video/cảnh**, ví dụ khoảng 70% train, 15% val, 15% test. Một buổi/video hoặc nhóm cảnh gần giống chỉ thuộc một tập. Không chia ngẫu nhiên từng frame.
- Thêm ảnh mới của từng split vào split tương ứng của dữ liệu cũ. Đặt tên có tiền tố nguồn để không đè file trùng tên.
- Train vẫn phải có dữ liệu cũ đa dạng để model không quên. Lượt đầu không nhân ảnh nền; giữ mức ảnh nền hợp lý, thử có kiểm soát trước khi tăng. Quyết định cũ của nhóm: không nhân hard negative quá 2 lần.
- Dành ảnh nền riêng cho `neg_val` để lựa chọn model, và `neg_test` để kiểm tra cuối. Không trùng với ảnh nền trong train. Nếu đã có 71 ảnh nền giữ riêng của nhóm thì tiếp tục giữ chúng riêng.
- Nên có một bộ ảnh thực tế từ buổi chụp mới hoàn toàn để kiểm tra cuối. Test cũ đã được xem nhiều lần chủ yếu còn giá trị kiểm tra hồi quy.

## 3. Tạo ZIP đúng cấu trúc

Notebook mới yêu cầu cấu trúc sau:

```text
dataset/
  images/
    train/
      old_001.jpg
      session01_001.jpg
      background01.jpg
    val/
    test/
    neg_val/               # tùy chọn, chỉ ảnh nền đã xác nhận
    neg_test/              # tùy chọn, chỉ ảnh nền đã xác nhận
  labels/
    train/
      old_001.txt
      session01_001.txt
      background01.txt     # file rỗng cho ảnh nền
    val/
    test/
```

Nếu lead gửi dạng `det/train/images`, `det/train/labels`, chuyển bằng cách **chép** nội dung sang `dataset/images/train` và `dataset/labels/train`; làm tương tự val/test. Giữ nguyên dữ liệu nguồn. Phải xác nhận mapping 3 lớp trước khi gộp, không chỉ đổi tên thư mục.

Ví dụ đặt thư mục chuẩn bị tại `D:\privateProject\xuLyAnh\detection_cloud`, rồi nén trên PowerShell:

```powershell
Set-Location D:\privateProject\xuLyAnh\detection_cloud
Compress-Archive -LiteralPath .\dataset -DestinationPath .\dataset.zip
```

Nếu archive lớn vượt khả năng `Compress-Archive`, dùng trình nén ZIP khác. Mở ZIP kiểm tra cấp cao nhất là `dataset/`, bên dưới có `images/` và `labels/`; không lồng thêm `detection_cloud/`.

Upload lên Google Drive:

```text
MyDrive/
  fruit_detection/
    input/
      base.pt              # bản sao detector đã được lead chấp nhận
      dataset.zip
```

Lưu riêng tên gốc, nguồn và phiên bản của `base.pt`. Không thay file input giữa một lượt train/resume. Notebook ghi SHA-256 để kiểm tra điều này.

## 4. Mở Colab và chạy notebook

1. Mở https://colab.research.google.com/.
2. Chọn mở/upload notebook và tải `detection_finetune_colab.ipynb` của repo lên.
3. Trong cài đặt runtime, chọn **GPU**, chọn T4 nếu được cấp. Tên menu có thể khác theo ngôn ngữ giao diện.
4. Chạy từng ô từ trên xuống; đừng dùng notebook cũ để “Run all” vì notebook cũ còn train classifier và có các cờ ghi đè.

Các phần trong notebook mới:

| Phần | Việc thực hiện |
|---|---|
| 1 | Xác nhận CUDA GPU và cài `ultralytics==8.4.180` như repo |
| 2 | Mount Drive, chép ZIP về `/content`, giải nén, xác nhận detector đúng 3 lớp |
| 3 | Kiểm tra ảnh đọc được, nhãn đúng 5 cột, class đúng, tọa độ hợp lệ, không trùng ảnh byte giữa các split |
| 4 | Chạy thử 1 epoch, batch 8 |
| 5 | Fine-tune chính và lưu checkpoint vào Drive sau mỗi epoch được lưu |
| 6 | Đo baseline/best/last trên validation và ảnh nền validation |
| 7 | Chọn checkpoint theo validation, rồi đánh giá test |
| 8 | Xuất model mới và CSV so sánh |

Kiểm tra cú pháp không thay được việc chạy trên Colab. Nếu notebook báo lỗi, sửa nguyên nhân trước khi chuyển sang bước tiếp theo.

## 5. Cấu hình lượt đầu và cách tiết kiệm GPU

Giữ kiến trúc detector từ model lead gửi, dự kiến YOLO11s. Không đổi sang model lớn hơn trong lượt đầu.

```python
epochs = 30
patience = 8
imgsz = 640
batch = 8
optimizer = "SGD"
lr0 = 0.001
freeze = 0
amp = True
workers = 2
cache = False
close_mosaic = 5
seed = 42
```

Đây là cấu hình khởi đầu đề xuất, không phải cấu hình đã được chứng minh tốt nhất cho bộ ảnh mới. `lr0` nhỏ để fine-tune nhẹ; chỉ rõ optimizer để tránh `auto` thay đổi ý nghĩa cấu hình học. `patience` giúp dừng khi validation không cải thiện. `amp=True` dùng mixed precision trên GPU.

Để tiết kiệm:

- Kiểm tra nhãn trước khi mở GPU, chỉ bật runtime khi đã có đủ input.
- Chạy thử 1 epoch rồi mới train chính. Ước lượng thời gian bằng log thực tế; không lấy thời gian RTX 3060 của lead làm cam kết cho Colab.
- Đọc ảnh từ `/content`, không train trực tiếp trên hàng nghìn file nhỏ của Drive.
- Không chạy classifier, teacher YOLO11l hoặc thí nghiệm GPU khác song song.
- Không tăng `imgsz`, số epoch và số model thử cùng lúc.
- Nếu hết VRAM: restart runtime, giảm batch 8 → 4 → 2 rồi thử lại. Nếu batch nhỏ vẫn lỗi, kiểm tra có model/tiến trình cũ giữ VRAM không trước khi giảm `imgsz`.
- Nếu vẫn cần giảm chi phí, `imgsz=512` là thí nghiệm riêng và cần đo lại, vì có thể giảm khả năng nhận quả nhỏ. Giữ 640 ở lượt đầu để so sánh dễ hơn.

Colab miễn phí không bảo đảm cấp GPU, loại GPU hoặc thời gian chạy cố định. Nếu hết quota, dừng và thử lại lúc khác; chỉ cân nhắc trả phí khi có deadline. Không có cấu hình notebook nào tự tạo thêm GPU hay bỏ được hạn mức dịch vụ.

## 6. Nếu Colab bị ngắt

Checkpoint backup ở:

```text
MyDrive/fruit_detection/runs/det_ft_cloud_v1/
  input_identity.json
  weights/last.pt
  weights/best.pt
  args.yaml
  results.csv
```

Khôi phục:

1. Kết nối lại GPU, chạy phần 1–3 để cài thư viện và giải nén đúng dữ liệu.
2. Trong ô cấu hình phần 4, giữ `RUN_NAME` cũ và đổi `RESUME = True`, rồi chạy ô đó. Khi True, ô này bỏ qua smoke.
3. Chạy phần 5. Notebook khôi phục checkpoint và gọi `train(resume=True)`.

Chỉ tiếp tục được từ checkpoint đã lưu xong; phần epoch đang chạy khi ngắt sẽ phải làm lại. `resume=True` dành cho lượt bị gián đoạn, không dùng cho lượt đã kết thúc. Muốn thử cấu hình/dataset mới, tạo `RUN_NAME` mới và dùng `RESUME=False`.

## 7. Đánh giá để biết fine-tune có ích không

Đừng chỉ nhìn loss giảm hoặc chỉ mở một ảnh đẹp. Notebook đo lại baseline và model mới trên cùng dữ liệu và cùng cấu hình, xuất CSV.

Cần xem:

- **mAP50, mAP50–95:** độ chính xác phát hiện tổng thể.
- **AP50 từng loại:** tránh cải thiện táo nhưng làm chuối kém đi.
- **Lỗi thực tế:** còn bỏ táo xanh trong ảnh nhiều loại quả không, còn box bao cả nải/trùng không.
- **Tỷ lệ ảnh nền báo sai:** số ảnh nền bị báo có quả chia tổng ảnh nền. Notebook dùng `conf=0.40`, IoU NMS `0.60` tương ứng cấu hình web hiện tại.

`best.pt` được framework chọn theo validation fitness; `last.pt` là checkpoint cuối. Không mặc định last tốt hơn, cũng không mặc định best ít báo sai nền hơn. Chọn giữa chúng **trên validation**, rồi mới chạy test cuối. Nếu test không đạt tiêu chí đã chốt, báo kết quả chưa đạt; không tiếp tục dò tham số dựa vào test rồi coi đó là test độc lập.

Đề xuất chốt với lead trước khi train: mAP50 giảm không quá khoảng 0.01, không giảm đáng kể khả năng phát hiện chuối, giảm lỗi cảnh nhiều quả và không tăng báo sai nền. Khi các chỉ số xung đột, giữ baseline nếu chưa có lý do rõ để đổi.

Notebook đo mAP với `conf=0.001`, khác số bàn giao cũ đo với `0.25`; vì vậy **so baseline được đo lại với candidate trong cùng CSV**, không so trực tiếp số cũ. Ngưỡng thấp để đo mAP không đồng nghĩa đổi ngưỡng web xuống mức đó.

Các mốc cũ chỉ để hiểu bối cảnh: model C được ghi mAP50 0.889, mAP50–95 0.732 ở conf 0.25; nền báo sai 3/71, và 1/71 khi nâng ngưỡng 0.40. Không xem đó là mục tiêu tuyệt đối nếu tập dữ liệu hoặc cách đánh giá đã đổi.

## 8. Tải model về Windows và thử bằng CPU

Phần 8 xuất:

```text
MyDrive/fruit_detection/outputs/det_ft_cloud_v1/
  det_candidate.pt
  metadata.json
  validation_comparison.csv
  final_test_comparison.csv
```

Đặt `det_candidate.pt` vào `BTL_XLA/exports/`. Giữ model cũ để đối chiếu. Đặt classifier hiện tại vào đó nếu máy chưa có.

Trong terminal backend mới:

```powershell
Set-Location D:\privateProject\xuLyAnh\BTL_XLA\backend
$env:MODEL_PATH = "D:\privateProject\xuLyAnh\BTL_XLA\exports\det_candidate.pt"
$env:CLS_MODEL_PATH = "D:\privateProject\xuLyAnh\BTL_XLA\exports\cls_finetune_best.pt"
$env:HALF = "0"
uvicorn main:app --port 8000
```

Lệnh trên giả định đã cài môi trường backend theo README; biến môi trường chỉ có hiệu lực trong terminal đó. Nếu backend đang chạy, dừng rồi khởi động lại bằng cấu hình mới. Frontend mở bằng terminal khác theo README (`npm install` lần đầu, rồi `npm run dev`).

Máy không có CUDA GPU sẽ chạy model trên CPU; webcam có thể không mượt. Kiểm tra ảnh tĩnh trước. Tốc độ CPU và độ chính xác là hai tiêu chí riêng. Không cần xuất TensorRT để chạy CPU. Việc cân nhắc ONNX/OpenVINO nên làm sau khi model đạt yêu cầu, vì backend hiện tải model qua `YOLO` và còn có classifier.

## 9. Gửi gì cho lead sau khi hoàn tất

Gửi model candidate, tên/hash model gốc, phiên bản dataset và cách chia, cấu hình train, phiên bản thư viện, CSV baseline/mới, ảnh minh họa lỗi trước/sau và kết luận nhận hay giữ baseline. Nêu rõ classifier giữ nguyên. Nếu chưa có bộ test thực tế độc lập, ghi rõ giới hạn này.

## Nguồn chính thức đã đối chiếu

- [Google Colab FAQ — GPU và hạn mức thay đổi](https://research.google.com/colaboratory/faq.html)
- [Ultralytics — định dạng dataset Detection](https://docs.ultralytics.com/datasets/detect/)
- [Ultralytics — train, cấu hình và resume](https://docs.ultralytics.com/modes/train/)
- [Ultralytics — callbacks lưu model](https://docs.ultralytics.com/usage/callbacks/)
- [Ultralytics — validation và ý nghĩa ngưỡng confidence](https://docs.ultralytics.com/modes/val/)
- [Cấu hình đúng phiên bản 8.4.180 của repo](https://github.com/ultralytics/ultralytics/blob/v8.4.180/ultralytics/cfg/default.yaml)
