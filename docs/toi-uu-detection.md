# Tối ưu Detection: thay đổi đã làm và cách sử dụng

**Kết quả Colab ngày 2026-11-11:** lượt `rotation_v1` chưa đạt. Giữ baseline `det_C_candidate.pt`; chưa dùng best/last của lượt xoay cho app. Test: baseline mAP50 88,74%, best 86,67%; ảnh nền báo sai tăng từ 1/71 lên 5/71. Xem [bảng val/test và quyết định](handoff/ket-qua-rotation-v1.md).

**Yêu cầu đếm riêng từng quả chưa đạt:** bộ lọc box chỉ xử lý một số khung trùng/bao cụm. Bước hiện tại là [duyệt lại nhãn từng quả và đo lỗi đếm/instance](duyet-nhan-tung-qua.md). Đã chuẩn bị công cụ offline tại `out/instance_review_v1/review.html`; chưa có dataset mới đã duyệt hay model mới đạt yêu cầu này.

## 1. Dùng ngay với model hiện tại

Tab **Ảnh** mặc định dùng **Kiểm tra kỹ**, đã bỏ nút lựa chọn Nhanh. API không truyền mode cũng mặc định careful. Sau khi cập nhật backend, bấm **Nhận dạng lại ảnh này** để lấy kết quả từ pipeline mới. Camera/video vẫn gửi mode fast để giữ tốc độ khung hình. CSV ghi chế độ thực sự đã dùng.

Khung hiển thị bằng số `#1`, `#2`… và màu trùng với số trong bảng kết quả, thay cho nhãn lớn che ảnh. Bấm một dòng trong bảng (hoặc Enter/Space khi dòng được chọn bằng bàn phím) để làm nổi khung tương ứng và làm mờ khung khác. Bấm lại dòng đó hoặc **Hiện tất cả khung** để bỏ chọn. Tình trạng tươi/hỏng nằm trong bảng.

Nếu khung sai, chọn dòng rồi bấm **Loại khung #… khỏi kết quả**. Khung đó được bỏ khỏi ảnh, bảng, tổng số và thống kê/CSV; không sửa weights hoặc nhãn dataset. Có nút **Khôi phục các khung đã loại**. Việc loại khung bám theo đối tượng gốc nên thay ngưỡng lọc không làm loại nhầm sang quả khác. CSV thêm cột `khung_loai_thu_cong`; nhận dạng lại ảnh sẽ xóa các lựa chọn loại của kết quả trước. Thao tác này không thêm được quả mà detector bỏ sót.

- Kiểm tra kỹ: lần lượt xử lý ảnh gốc và ảnh xoay 90° ở kích thước 960, đưa box về tọa độ gốc rồi dùng NMS để gộp box trùng. Giữ ngưỡng confidence 0.40 và IoU 0.60 mặc định; không cộng điểm từ các hướng.
- Lọc box bao cụm khi có hai box nhỏ cùng lớp: diện tích từng box bằng 25–85% box lớn, ít nhất 80% nằm trong box lớn, độ tin cậy ít nhất bằng một nửa box lớn, IoU giữa hai box nhỏ <=0.30 và hợp phần nằm trong box lớn phủ >=85% diện tích box lớn. Một box lồng đơn lẻ không đủ để loại box lớn.
- Nếu còn cặp bao/lồng đáng nghi, xử lý thêm ảnh xoay ngược 90°. Chỉ nhận box từ hướng bổ sung nếu nó hỗ trợ thay một box bao cụm hiện có; không thêm các phát hiện mới ngoài vùng đó. Ảnh rỗng hoặc chỉ có một box không chạy hướng thứ ba. Đó là quy tắc hình học, chưa chứng minh số box bằng số quả.
- Video/webcam tiếp tục dùng một lượt ảnh640. Kiểm tra kỹ chậm hơn; vẫn có thể bỏ sót, nhận sai hoặc gom cụm khi chưa có hai box nhỏ đủ điều kiện thay thế.
- Mặc định giới hạn PyTorch và OpenCV ở 2 luồng CPU, các lượt nhận diện chạy tuần tự. Có thể đặt `CPU_THREADS=1` nếu muốn giảm tải thêm. Giới hạn luồng không bảo đảm nhiệt độ cụ thể; tốc độ tùy máy.
- Classifier không thay đổi; crop được phân loại thành từng nhóm tối đa 8 để giới hạn bộ nhớ.
- Model `.pt` hiện tại không bị thay hoặc ghi đè. App vẫn dùng MODEL_PATH bạn đặt khi khởi động.

Khởi động lại backend trong PowerShell (dừng tiến trình cũ bằng Ctrl+C trước):

```powershell
cd D:\privateProject\xuLyAnh\BTL_XLA
$env:MODEL_PATH = "$PWD\exports\det_C_candidate.pt"
$env:HALF = "0"
$env:CPU_THREADS = "2"
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Frontend đang chạy bằng Vite sẽ nhận source mới; tải lại trang. Nếu chưa chạy:

```powershell
cd D:\privateProject\xuLyAnh\BTL_XLA\frontend
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

## 2. Những gì đã kiểm tra trên CPU

### Bản lọc box bao cụm (2026-11-11)

Đã chạy 16 test cho tọa độ ảnh không vuông, NMS, box bao cụm, hai box trùng, một box lồng, khác lớp, tín hiệu nhỏ/yếu, vùng phủ không đủ và giới hạn đóng góp của hướng bổ sung. Kiểm tra API thật: mode mặc định careful, mode fast của video, mode không hợp lệ, bbox trong ảnh và đầu ra classifier. Frontend build thành công; thao tác loại khung/đổi ngưỡng/khôi phục/CSV được kiểm tra ở cấp hàm, chưa kiểm thử thao tác trình duyệt tự động.

Trên `bananas.webp`, box cũ khoảng `[468,19,909,786]` bao vùng giữa đã được bỏ và thay bằng khung nhỏ từ hướng bổ sung cùng khung nhỏ đã có. Tổng vẫn 5 box ở conf0.40; vẫn còn box gom vùng bên trái, chưa thể coi là đếm đủ từng quả. Trên 6 ảnh mẫu còn lại, số box giữ nguyên so với chế độ kỹ hai hướng: ảnh đơn 1, cốc 0, ảnh ba loại quả 3, ảnh nải 8, ảnh nhiều chuối 10, test.webp 9. Đây là kiểm tra hồi quy, không phải số quả thật đã được gán nhãn.

71 ảnh nền ở conf0.40: trước và sau lọc đều báo sai 3/71 ảnh. Thời gian CPU đo trên 7 ảnh mẫu khoảng 0,64–1,42 giây riêng detector trong phiên kiểm tra này; ảnh cần hướng thứ ba có thêm chi phí. Máy và trạng thái chạy ảnh hưởng thời gian. Báo cáo/preview: `out/box_merge_review/checks.json`, `out/box_merge_review/*_after.jpg`.

### Kết quả trước khi thêm lọc box bao cụm

Dùng `det_C_candidate.pt`, ngưỡng 40%, cùng ảnh đầu vào:

| Ảnh | Nhanh | Kiểm tra kỹ |
|---|---:|---:|
| bananas.webp (chuối xếp dọc) | 0 box | 5 box |
| manyBanana.webp (chuối chồng nhau) | 8 box | 10 box |
| singleBanana.jpg | 1 box | 1 box |
| cup.jpg | 0 box | 0 box |
| bananaappleorange.jpg | 3 box | 3 box |

Đây là số box, chưa phải đánh giá đếm đúng trên bộ ảnh gán nhãn sạch. Ảnh nhiều chuối vẫn có box gom cụm hoặc trùng. Không khẳng định đã giải quyết đếm mọi quả che khuất.

Trên 71 ảnh nền giữ riêng, cùng ngưỡng 0.40:

- Nhanh: 0/71 ảnh báo có quả.
- Kiểm tra kỹ: 3/71 ảnh báo có quả.

Các số này khác phép đo conf0.25 trong bàn giao. Chế độ kiểm tra kỹ có thêm phát hiện nhưng tăng nhận nhầm nền. Đây là số đo trước khi đổi mặc định theo yêu cầu người dùng; bản hiện tại dùng kiểm tra kỹ cho tab Ảnh. Thời gian đo ở bản trước từng lượt API trên một số ảnh khoảng 0.12 giây ở chế độ nhanh và 0.36–0.53 giây ở chế độ kỹ; không phải cam kết tốc độ trên mọi máy/ảnh.

Báo cáo chi tiết: `out/optimization_preview/comparison.json`, `background_check.json`, `api_checks.json`. Ảnh minh họa nằm cùng thư mục.

Ở bản trước đã kiểm tra phép đổi tọa độ ảnh xoay trên ảnh không vuông, NMS giữa các hướng, API mặc định fast và hai chế độ, tên lớp, box trong giới hạn ảnh, classifier, chế độ không hợp lệ và ảnh không đọc được. Đây là ghi nhận lịch sử; bản hiện tại mặc định careful.

## 3. Lượt train mới trên Colab: tăng khả năng chịu hướng ảnh

Đã chuẩn bị:

- Notebook: `finetune_rotation_colab.ipynb`.
- ZIP: `out/rotation_v1/detector_rotation_colab_input.zip` (khoảng 347 MB).
- Manifest: `out/rotation_v1/data/rotation_manifest.json`.
- Script tạo dữ liệu: `tools/prepare_rotation_data.py`.
- Script train/evaluate: `tools/train_rotation.py` (có bản sao trong ZIP).

Dữ liệu được dựng trên bản sao:

1. Loại 50 ảnh train trong danh sách quarantine của nhóm.
2. Loại thêm 3 biến thể thuộc nhóm `b_f011` có chuối nhưng nhãn rỗng; không tự bịa box.
3. Chuyển 3 dòng polygon train thành box bao polygon, không sửa nhãn val/test gốc.
4. Thêm 519 ảnh xoay 90/180/270 từ 173 ảnh train có nhãn. Tọa độ box được biến đổi tương ứng.
5. Giữ replay 165 ảnh own ×4 và 287 ảnh nền ×2. Tổng 4.065 lượt ảnh train, gồm cả ảnh lặp.
6. Giữ 301 ảnh val và 373 ảnh test gốc. Thêm riêng 301 bản xoay lấy từ nguồn val để validation theo dõi cả hướng mới. Nguồn val không đưa vào train.
7. Giữ 71 ảnh nền riêng để kiểm tra hồi quy. Các ảnh người dùng vừa thử không đưa vào train.

**Không dùng ảnh ghép tự sinh trong lượt này.** Đã tạo 240 ảnh nháp để xem thử, nhưng preview còn lỗi cắt nền và đối tượng/nhãn đáng nghi. Chúng bị loại khỏi train list, script chủ động từ chối `--use-synthetic` với gói này. Không chỉnh cờ để ép train ảnh chưa đạt. Cần một bộ ảnh ghép được kiểm tra hoặc ảnh thật gán nhãn sạch cho thí nghiệm tiếp theo.

Các nhãn train khác chưa được duyệt thủ công toàn bộ. Val/test gốc còn giới hạn đã biết, gồm ảnh chuối có nhãn rỗng. Đây là thí nghiệm tăng tính ổn định với hướng ảnh, không phải bộ dữ liệu đã được chứng minh sạch.

## 4. Các bước chạy notebook mới

1. Trên Drive tạo `MyDrive/fruit_rotation/input/`, upload `detector_rotation_colab_input.zip` vào đó.
2. Upload và mở `finetune_rotation_colab.ipynb` trên Colab; chọn GPU runtime.
3. Chạy phần 1 (cài thư viện) và phần 2 (mount Drive, giải nén, kiểm tra hash).
4. Phần 3 giữ `ACTION="smoke"`, `RUN_NAME="rotation_v1"`, `BATCH=8`, `RESUME=False`; chạy phần 3 rồi 4 để thử 1 epoch.
5. Smoke xong, đổi `ACTION="train"`, chạy lại phần 3–4. Lượt chính luôn bắt đầu từ candidate cũ, không từ model smoke.
6. Train xong, đổi `ACTION="val"`, chạy lại phần 3–4 để so baseline/best/last trên val gốc + val xoay. Chọn `CHECKPOINT` dựa trên validation trước khi xem test.
7. Đổi `ACTION="test"` để kiểm tra baseline với checkpoint đã chọn trên test gốc. Không dùng test để dò lại tham số.
8. Chỉ khi cần lưu candidate riêng để xem xét: đổi `ACTION="export"`, chạy lại 3–4. Không tự thay model app. Với kết quả `rotation_v1` ngày 2026-11-11, giữ baseline; không cần export để triển khai lượt này.

Model gốc là `det_C_candidate.pt` đã tải về, được chép trong ZIP thành `exports/base.pt`. Cấu hình đã chuẩn bị: 25 epoch tối đa, SGD lr0=0.001, batch8, ảnh640, degrees15, patience8, close_mosaic5. Kết quả GPU người dùng gửi ngày 2026-11-11 được ghi ở [bàn giao kết quả](handoff/ket-qua-rotation-v1.md); cấu hình này chưa cải thiện được chất lượng tổng thể.

Checkpoint/log lưu trực tiếp tại `MyDrive/fruit_rotation/runs/rotation_v1/`. Nếu bị ngắt và checkpoint còn trạng thái resume hợp lệ như mô tả bên dưới: mở lại GPU, chạy 1–2, đặt `ACTION="train"`, `RESUME=True`, cùng RUN_NAME và ZIP; chạy 3–4. Nếu chưa có checkpoint đầu thì dùng RUN_NAME mới. Nếu OOM giảm batch xuống4. Không chạy train trên CPU máy cá nhân.

### Khi ô chạy báo lỗi

- `NameError: ACTION is not defined`: chưa chạy phần 3 trong phiên hiện tại. Chạy phần 3 trước phần 4. Nếu vừa reset/đổi runtime, chạy lại phần 1–2 trước để cài thư viện, mount Drive và giải nén.
- `CalledProcessError ... exit status 1`: đây là thông báo script con đã lỗi, chưa phải nguyên nhân. Xem traceback của `train_rotation.py` phía trên. Ô phần 4 mới gộp stdout/stderr, hiển thị trực tiếp, lưu log tại `MyDrive/fruit_rotation/runs/_logs/` và in lại 80 dòng cuối khi lỗi. Có thể chép riêng ô phần 4 mới vào notebook đang mở; không cần upload lại ZIP dữ liệu.
- Chỉ đặt `RESUME=True` khi checkpoint của lượt bị ngắt còn `epoch >= 0`, có `optimizer` và cấu hình `train_args.epochs=25`. Có file `last.pt` chưa đủ: checkpoint đã hoàn tất thường có epoch=-1, optimizer=None. Ultralytics có thể chuyển sang train mới với mặc định 100 epoch khi resume file này. Notebook mới kiểm tra và chặn trước khi gọi script, kể cả khi dùng ZIP cũ. Nếu lượt đã kết thúc, dùng `ACTION="val"`, `RESUME=False`. Nếu script báo `Run exists` nhưng không có checkpoint thì dùng RUN_NAME mới, giữ thư mục cũ để xem log. Không xóa lượt cũ hoặc đổi batch/resume chỉ dựa vào dòng `CalledProcessError`.

## 5. Tiêu chí nhận model

Chưa thay model app tự động. Cần so với baseline trên cùng cấu hình, kiểm tra AP từng loại, bỏ sót và khung trùng ở cảnh nhiều quả; giữ tỷ lệ nhận nhầm nền trong mức nhóm chấp nhận. Nếu model mới không cải thiện rõ đúng lỗi hướng ảnh hoặc làm cảnh đơn giản kém đi, giữ model hiện tại.

Để đánh giá đáng tin hơn, bước dữ liệu tiếp theo vẫn là bổ sung ảnh thật nhiều quả có nhãn và nguồn cảnh riêng cho validation/test. Thêm một lượt train trên dữ liệu xoay chưa thay được bước đó.
