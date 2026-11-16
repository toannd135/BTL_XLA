# Cải thiện độ chính xác detector — 16/11/2026

## Trạng thái

Code ứng dụng vẫn ở nguyên main. Gói này nằm trong out/, không sửa code, trọng số hay nhãn gốc. Có 15 ảnh train và 8 ảnh validation lấy từ lượt duyệt đã chuẩn bị. Không chứa diagnostic/test. Đây là mẫu để sửa và kiểm tra quy trình, không phải toàn bộ dữ liệu huấn luyện mới.

`obb_drafts_UNREVIEWED.zip` chứa ảnh và khung xoay do detector hiện tại dự đoán ở conf 0.25, IoU 0.7, imgsz640. Tất cả nhãn đều chưa được duyệt; có thể thiếu quả, nhầm loại hoặc gộp nhiều quả. File nhãn rỗng không có nghĩa là ảnh đã được xác nhận làm nền.

`draft_provenance.json` giữ nguồn ảnh, split, hash, số box nháp, scores và trạng thái pending. Không đưa gói nháp này vào train trực tiếp.

## 1. Duyệt OBB một lần trong CVAT

1. Mở https://app.cvat.ai, đăng nhập và tạo một project mới, ví dụ fruit_obb_clean_v1. Có thể để nhãn trống khi tạo để nhập từ dataset.
2. Vào project mới → Actions → Import dataset.
3. Chọn định dạng **Ultralytics YOLO Oriented Bounding Boxes** (tên có thể kèm số phiên bản). Không chọn YOLO Detection thường hoặc Segmentation.
4. Tải file obb_drafts_UNREVIEWED.zip trong thư mục này. CVAT hỗ trợ nhập dataset có ảnh và nhãn, tách các subset thành task tương ứng. Giữ train/val như gói đã chuẩn bị.
5. Mở từng task/job. Kiểm tra mọi quả mục tiêu; sửa khung có sẵn, thêm quả bị thiếu, xóa box trùng và sửa loại. Lưu kết quả trong CVAT.
6. Xuất dataset của project với cùng định dạng OBB. **Tắt Save images / Include images**, vì tài khoản hiện tại bị giới hạn xuất project kèm ảnh. 23 ảnh gốc vẫn có trong draft_dataset/images và được kiểm tra hash; có thể ghép nhãn xuất về với ảnh theo tên file và split. Lưu ZIP nhãn về thư mục này dưới tên **obb_reviewed_v1.zip**.
7. Báo lại đường dẫn zip. Cần kiểm tra nhãn/split/nguồn ảnh trước khi dựng dữ liệu train kết hợp.

Nhãn và thứ tự hiện tại: 0=banana, 1=apple, 2=orange. Đối chiếu names trong data.yaml sau khi xuất, không tự giả định ID không đổi.

Quy tắc duyệt:

- Mỗi quả nhìn thấy và xác định được có một khung riêng. Một khung không đại diện cho cả nải hoặc đĩa.
- Khung xoay có thể chồng sang quả bên cạnh vì chuối cong; kiểm tra theo từng cuống/thân/đầu quả.
- Nho, lê và vật ngoài ba loại không được gán nhầm thành táo/cam/chuối.
- Ảnh có cả nho và táo: vẫn gán đủ táo. Chỉ xác nhận ảnh nền nếu không có quả mục tiêu.
- Ảnh quá che khuất khiến không xác định đủ từng quả: ghi tên ảnh vào danh sách loại. Không biến ảnh đó thành nền bằng cách xóa hết nhãn.

Nguồn định dạng và thao tác:
- https://docs.cvat.ai/docs/manual/advanced/import-datasets/
- https://docs.cvat.ai/docs/manual/advanced/formats/format-yolo-ultralytics/
- https://docs.ultralytics.com/datasets/obb/

## 2. Chuẩn bị tập train thực tế

Sau khi duyệt, kết hợp ảnh mới với phần dữ liệu cũ đã rà sạch; giữ đủ ba loại. 15 ảnh train trong lượt đầu là mẫu rà lỗi, chưa đủ làm toàn bộ tập fine-tune. Ưu tiên bổ sung theo lỗi đo được: chuối nằm sát nhau, táo xanh, nhiều loại cùng ảnh, nho/lê cạnh quả mục tiêu, ánh sáng/nền thực tế.

Giữ nhóm nguồn/phiên quay và ảnh gần giống trong cùng split. Không lấy ảnh diagnostic/test vào train. Dữ liệu nền bổ sung cho train không lấy từ 71 ảnh nền đã giữ riêng để kiểm tra. Không ghi đè local_dataset/det/val, test hoặc work/obb/data.

Dựng một thư mục mới, ví dụ fruit_clean_obb_v1, với images/train, images/val, labels/train, labels/val và data.yaml. Nhãn OBB: class + 8 tọa độ bốn góc chuẩn hóa 0–1. Nhãn HBB từ công cụ JSON cũ không phải nhãn OBB đã duyệt; đổi một HBB thành bốn góc không tạo được khung xoay sát quả.

Kiểm tra trước train: ảnh/nhãn khớp tên, đủ ba lớp trong train và validation, không có box gộp, nhãn trùng, tọa độ lỗi, ảnh thiếu nhãn vô tình bị coi là nền, hoặc trùng nguồn giữa các split. Kiểm tra lại ảnh vẽ nhãn sau khi chuyển định dạng.

## 3. Fine-tune trên Colab sau khi dữ liệu đạt yêu cầu

Chọn T4 GPU; lưu kết quả trên Drive của bạn. Khởi tạo từ exports/det_obb_s.pt hiện tại. Lượt đầu chỉ fine-tune detector; giữ nguyên classifier tươi/hỏng để tách tác động.

Ví dụ cấu hình thử, chỉ dùng khi bộ OBB sạch đã dựng xong và đã giải nén đúng đường dẫn. Các đường dẫn dưới đây là vị trí cần chuẩn bị trong runtime Colab, không phải tài nguyên đã tồn tại sẵn trên Drive của bạn:

```python
from pathlib import Path
import torch
from ultralytics import YOLO

DATA = Path('/content/fruit_clean_obb_v1/data.yaml')
INIT = Path('/content/drive/MyDrive/fruit_accuracy_v1/det_obb_baseline.pt')
PROJECT = '/content/drive/MyDrive/fruit_accuracy_v1/runs'
assert torch.cuda.is_available(), 'Chọn runtime GPU trước khi train'
assert DATA.is_file() and INIT.is_file(), 'Kiểm tra dữ liệu sạch và model khởi tạo'

model = YOLO(str(INIT))
model.train(
    data=str(DATA), epochs=30, imgsz=640, batch=8, device=0,
    optimizer='AdamW', lr0=0.0005, seed=42, patience=10,
    close_mosaic=5, project=PROJECT, name='obb_clean_v1',
    exist_ok=False,
)
```

Đây là cấu hình bắt đầu thí nghiệm, không phải cam kết tăng độ chính xác. Không dùng lại tên obb_colab. Script train_obb.py hiện tại cố định work/obb/data/obb.yaml và tự resume nếu có last.pt, nên chạy nguyên script đó có thể tiếp tục train dữ liệu/lượt cũ.

Resume chỉ dành cho lượt thực sự bị ngắt và còn checkpoint có optimizer. Model đã kết thúc và được strip optimizer không dùng như checkpoint resume. Thí nghiệm mới cần tên run mới.

## 4. So sánh trước khi thay model

Trên cùng validation sạch, đánh giá model gốc và candidate với cùng imgsz, conf, IoU. Theo dõi AP/P/R từng loại, ghép một-một đúng loại với từng quả, sai số đếm từng loại, tỷ lệ đếm đúng, box trùng/gộp và ảnh nền báo sai.

Đo riêng detector, sau đó đo pipeline app với CLIP/teacher cùng cấu hình cho cả hai model. Không dùng phép đo detector đơn để kết luận chất lượng pipeline app. tools/evaluate_instance_count.py trên main hiện gọi HBB detect ở960; chưa dùng nguyên script đó để đánh giá model OBB.

Chọn cấu hình/checkpoint bằng validation; test giữ riêng để đánh giá sau khi chốt. Ba ảnh diagnostic đã duyệt dùng rà lỗi, không phải test cuối độc lập. Không đặt mục tiêu cứng rằng một ảnh phải ra khoảng20 rồi chỉnh ngưỡng cho riêng ảnh đó.

## 5. Chạy thử candidate sau khi đạt tiêu chí

Lưu candidate riêng, ví dụ exports/det_obb_clean_v1.pt. Giữ exports/det_obb_s.pt để quay lại. Khởi động backend với MODEL_PATH trỏ candidate; các cấu hình còn lại giữ như bản đối chứng khi so sánh. Chỉ thay mặc định khi candidate cải thiện các lỗi mục tiêu mà không làm suy giảm đáng kể các loại còn lại hoặc tăng báo sai vật khác.

## Phần việc hiện tại

Bạn: duyệt 23 ảnh nháp trong CVAT và xuất obb_reviewed_v1.zip. Có thể chia nhau hai task train/val.

Phần tiếp theo sau khi có zip: kiểm tra nhãn/split, chọn dữ liệu cũ sạch để kết hợp, bổ sung ca khó còn thiếu và chuẩn bị gói train Colab mới. Chưa có bộ train hoàn chỉnh đã duyệt hoặc trọng số mới từ gói này.
