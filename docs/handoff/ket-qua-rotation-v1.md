# Kết quả rotation_v1 — 2026-11-11

## Quyết định

Giữ baseline `exports/det_C_candidate.pt` làm model được khuyến nghị cho app. Không đưa best/last của `rotation_v1` vào app: validation không đạt, test xác nhận giảm ở cả ba loại quả và tăng nhận nhầm nền. Lượt thí nghiệm đã có đủ kết quả val/test để ghi nhận; không cần train lại hoặc export để triển khai lượt này.

Baseline là model đầu vào gói ZIP (`exports/base.pt`), không phải một model mới chưa train. Script đã chuẩn bị cấu hình train tối đa 25 epoch, thêm dữ liệu xoay 90/180/270 cho cả chuối, táo và cam. Các bảng dưới lấy từ JSON Colab do người dùng gửi trong cuộc trao đổi; không phải phép đánh giá lại trên máy local.

## Validation

Val gồm 301 ảnh val gốc và 301 bản xoay từ nguồn val. Các bản xoay không phải bộ cảnh độc lập. Tất cả số AP/mAP dưới đây là phần trăm.

| Model | mAP50 | mAP50–95 | AP50 chuối | AP50 táo | AP50 cam | Ảnh nền báo sai / 71 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 90,30 | 75,03 | 86,15 | 92,87 | 91,87 | 1 |
| best | 87,51 | 70,44 | 87,02 | 89,88 | 85,62 | 5 |
| last | 81,52 | 64,35 | 80,28 | 87,35 | 76,93 | 2 |

Best tăng AP50 chuối 0,87 điểm phần trăm, nhưng giảm táo 2,99 và cam 6,25 điểm; nền báo sai tăng. Best được chọn trong hai checkpoint mới để đo test cho báo cáo. Baseline vẫn được ưu tiên từ validation; không dùng test để dò chọn thêm checkpoint.

## Test

Test gốc 373 ảnh, cùng cấu hình đánh giá cho baseline và best trong lượt này. Chỉ số mAP/AP đo với conf=0.001; phép kiểm tra 71 ảnh nền dùng conf=0.40. Script dùng ảnh640, IoU0.60, agnostic NMS. Không so trực tiếp với các bảng cũ có cấu hình đánh giá khác.

| Model | mAP50 | mAP50–95 | AP50 chuối | AP50 táo | AP50 cam | Ảnh nền báo sai / 71 |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 88,74 | 72,66 | 83,37 | 87,35 | 95,50 | 1 |
| best | 86,67 | 66,92 | 81,69 | 86,06 | 92,26 | 5 |

So với baseline, best giảm mAP50 2,07 điểm phần trăm, mAP50–95 5,73 điểm; AP50 chuối/táo/cam giảm lần lượt 1,68 / 1,29 / 3,24 điểm. Không có bằng chứng từ các số này để nhận model xoay vào app hoặc khẳng định đã giải quyết đếm quả chồng nhau.

## Lưu kết quả và hướng tiếp theo

Giữ `comparison_val.json`, `comparison_test.json`, `args.yaml`, `results.csv`, `input_identity.json` và weights tại `MyDrive/fruit_rotation/runs/rotation_v1/` để truy nguyên lượt train. Chưa có log train đầy đủ trên máy local nên chưa kết luận nguyên nhân giảm là overfit hay thành phần dữ liệu/cấu hình nào.

Bước dữ liệu tiếp theo: duyệt và sửa nhãn từng quả trên ảnh nhiều quả/chuối; bổ sung ảnh thật nhiều loại quả, tách nguồn cảnh cho train và validation. Đánh giá riêng bỏ sót, khung trùng, box gom nhiều quả và từng loại quả trước khi mở lượt train khác. Giữ test gốc để đối chiếu; không sửa nhãn test tại chỗ.

## Sự cố resume cần nhớ

Checkpoint trên Drive đã được kiểm tra: epoch=-1, train_args.epochs=25, optimizer=None. Đây là checkpoint đã được hoàn tất/lưu gọn, không dùng để resume. Giá trị 25 là cấu hình tối đa, không chứng minh lượt đã chạy đủ 25 epoch.

Sau khi thử resume đã xuất hiện log 4/100. Mã Ultralytics local có nhánh chuyển sang train mới khi checkpoint thiếu trạng thái resume; chưa có toàn bộ log từ Colab để xác nhận đầy đủ diễn biến lượt 100. Notebook và script local đã bổ sung kiểm tra checkpoint còn optimizer, epoch hợp lệ và tổng epoch đúng trước khi resume. Notebook có kiểm tra này ngay cả khi dùng ZIP cũ. Không chỉ dựa vào việc tồn tại `last.pt` để bật RESUME.
