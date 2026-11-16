# Kết quả kiểm tra obb_reviewed_v1.zip — 16/11/2026

Đã đọc được ZIP; đúng 3 nhãn banana/apple/orange và đúng danh sách 15 train +8 val. ZIP không kèm ảnh, ảnh gốc đã ghép theo tên/hash.

Theo xác nhận của bạn, hai ảnh val `7fc208b0975b8a85053b0559` và `85caa7b72160f48981c54944` bị loại khỏi bản sao mới. Không tạo nhãn rỗng để biến chúng thành nền. Còn 21 ảnh (15 train,6 val),84 box do bạn gửi.

## Ba chỗ cần sửa trong CVAT

Số frame dưới đây là số trong ô điều hướng phía trên màn hình CVAT, tính từ0; khớp thứ tự các ảnh trong train.txt/val.txt của ZIP. Tên file là căn cứ chính nếu thứ tự hiển thị khác.

| Task | Frame | Tên ảnh | Cần sửa |
|---|---:|---|---|
| train | 10 | 95369e3dc95bb75bb39d7499.jpg | Box1 đang gán orange cho một quả chuối. Đổi về banana; kiểm tra lại ba khung đang đại diện đúng từng quả. |
| val | 4 | 81d5f64b7d06b9ac1ccea6f0.jpg | Ảnh cụm nhiều quả màu cam hiện chỉ có6box nháp. Cần gán đủ mọi quả mục tiêu xác định được, hoặc báo loại ảnh/thay bằng ảnh dễ duyệt hơn. Không giữ6box này làm nhãn đầy đủ. Nếu loại ảnh này thì tập val hiện tại sẽ không còn nhãn cam và cần bổ sung ảnh cam khác. |
| val | 7 | fd1fb0e99efd248fdd05be40.jpg | Có khung trên lá/cành/nền và các khung chồng lên những phần của cùng quả. Xóa khung không phải quả, sửa mỗi quả thành một khung và kiểm tra đủ quả nhìn thấy. |

Ảnh đối chiếu: `can_sua_3_anh.png`; các trang tổng quan là audit_page_1.png đến audit_page_4.png. Đây là kiểm tra bằng mắt, chưa phải chứng nhận rằng mọi nhãn còn lại đều hoàn hảo.

Sửa các vị trí trên, Save trong CVAT, rồi xuất lại project ở định dạng **Ultralytics YOLO Oriented Bounding Boxes**, **tắt Save images**. Lưu thành `out/obb_review_20261116/obb_reviewed_v2.zip`, giữ bản v1 để đối chiếu. Có thể giữ hai ảnh đã loại trong project CVAT; bộ dựng dữ liệu sẽ tiếp tục loại đúng hai ID đã được xác nhận.

## Lỗi góc vượt biên đã xử lý trên bản sao

36box trên17ảnh có góc ngoài khoảng0–1. Đã tạo bản sao `geometry_fixed_f5cd4923_NOT_READY` với thêm viền đen ngoài ảnh, rồi dịch/chuyển hệ tọa độ tất cả box theo kích thước ảnh mới. Không cắt góc, không thu nhỏ hay đổi góc khung bạn vẽ. Hình học theo pixel trước/sau giữ trong sai số0.001px.

ZIP v1 và ảnh/nhãn gốc không bị sửa. Không tự đổi class, tự thêm/xóa box hoặc tự coi nhãn thiếu là nền. Bản sao không có data.yaml để tránh dùng nhầm khi các lỗi nội dung còn tồn tại. Khi có v2 phải kiểm tra và xử lý biên lại trên v2; không dùng nhãn sửa sau này với ảnh đã thêm viền nếu chưa chuyển hệ tọa độ tương ứng.

## Trạng thái sử dụng

Chưa train và chưa tạo gói train cuối. Bộ21ảnh vẫn là mẫu rà nhãn; train hiện chỉ có1box táo. Sau khi sửa nhãn cần kết hợp dữ liệu cũ sạch và thêm các ca còn thiếu để tránh lệch lớp. Giữ nguyên code main, model app và val/test gốc.
