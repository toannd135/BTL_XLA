# Bước tiếp theo: train thử detector trên Colab

Không yêu cầu sửa thêm nhãn trong lượt này. Giữ nguyên code main và model ứng dụng.

## Hai file cần dùng

- `fruit_accuracy_trial_20261116.zip`: dữ liệu và baseline tự chứa, khoảng407MiB.
- `finetune_accuracy_trial_colab.ipynb`: notebook chạy trên GPU Colab.

## Thao tác

1. Mở Google Drive của bạn, vào **My Drive** và upload ZIP ngay thư mục gốc. Không giải nén ZIP trên Drive, không cần dữ liệu từ Drive bạn khác.
2. Mở https://colab.research.google.com → **File → Upload notebook**, chọn notebook ở thư mục này.
3. Chọn **Runtime → Change runtime type → T4 GPU**. Notebook dừng nếu không có GPU để tránh train CPU chậm.
4. Chạy lần lượt các ô từ trên xuống. Ô mount Drive: cho phép truy cập Drive của bạn.
5. Giữ `ACTION='smoke'`, `RESUME=False`, chạy thử1epoch. Log chi tiết lưu trên Drive, màn hình chỉ hiện tiến độ định kỳ để giảm lag.
6. Smoke xong đổi `ACTION='train'`, `RESUME=False`, chạy lại hai ô **Cấu hình lượt chạy** và **Thực thi**. Giữ `EPOCHS=20`, `BATCH=8`; gặp CUDA OOM hạ BATCH xuống4.
7. Train xong đổi `ACTION='val'`, `RESUME=False`, chạy lại hai ô đó để so baseline/candidate trên cùng dữ liệu validation.
8. Gửi nội dung `MyDrive/fruit_accuracy_trial_20261116/runs/obb_patch_v1/comparison_val.json` để đánh giá. Chưa thay model app.

Nếu train bị ngắt, chạy lại setup và dùng cùng NAME, ACTION=train, RESUME=True. Chỉ resume checkpoint chưa kết thúc và còn optimizer. Run đã xong cần ACTION=val; thí nghiệm khác dùng NAME mới. Không chạy lại notebook train OBB cũ vì nó có thể dùng lại dữ liệu/lượt cũ.

## Gói thực sự dùng dữ liệu nào?

- Giữ14ảnh train bổ sung từ bản bạn gửi; lặp4lần có chủ đích để chúng không bị chìm trong dữ liệu cũ.
- Cách ly ảnh train gán sai chuối thành cam, ảnh val nhiều quả gán thiếu và ảnh val táo có box nền/trùng.
- Loại hai ảnh val chưa có nhãn đúng theo yêu cầu của bạn.
- Dùng4ảnh val chuối đã duyệt để rà count; chúng không được đưa vào train.
- Kết hợp bộ OBB cũ: tổng5340lượt ảnh train,4594ảnh train khác nhau;297ảnh val gồm cả ba loại.
- Nhãn OBB vượt biên đã được chuyển sang ảnh bản sao có thêm viền, giữ hình học theo pixel. Original ZIP, val/test gốc và model app không bị ghi đè.

## Giới hạn cần đọc khi xem kết quả

Đây là thí nghiệm bổ sung nhãn, không phải bộ dữ liệu hoàn toàn sạch. Val cũ còn nhãn nải;4ảnh val được rà lại chỉ có chuối và là dữ liệu phát triển. Thống kê mAP/count chỉ hỗ trợ so tương đối. Notebook chỉ đo detector, chưa đo cả CLIP/teacher/classifier. Không tự nhận model mới hoặc hứa accuracy tăng sau20epoch. Không có ảnh test/diagnostic nào được đưa vào train.

`input_manifest.json` giữ danh sách ảnh, nguồn, hash, các ảnh cách ly và các giới hạn. `package_summary.json` giữ hash ZIP để notebook kiểm tra đúng bản upload.
