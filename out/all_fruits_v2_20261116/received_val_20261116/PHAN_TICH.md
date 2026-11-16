# Phân tích kết quả validation v2

Đã đối chiếu JSON người dùng gửi với cả 3 JSON trong ZIP và tính lại operating-point metrics
từ 297 bản ghi mỗi model; kết quả khớp. SHA manifest trong ZIP input khớp báo cáo Colab.
Không chạy train hoặc test. Không sửa các model app.

## Cùng giao thức trong báo cáo: imgsz1280, conf đếm0.4, NMS0.7

| Chỉ số | Baseline | Trial v1 | Trial v2 |
|---|---:|---:|---:|
| mAP50 | 0.6321 | 0.5387 | **0.7909** |
| mAP50-95 | 0.3014 | 0.2523 | **0.5385** |
| Macro F1 | 0.4833 | 0.4554 | **0.7540** |
| MAE đếm cả ba loại (thấp hơn tốt hơn) | 1.313 | 1.788 | **0.556** |
| Đếm đúng cả ba loại trên ảnh | 33.00% | 23.23% | **63.30%** |

V2 tốt hơn hai bản cũ trong cấu hình1280 này. Chưa thể kết luận tốt nhất trong mọi cấu hình,
vì báo cáo trước baseline/v1 ở640 đạt mAP50-95 khoảng0.742/0.750.
Hai báo cáo khác kích thước và chi tiết giao thức, cần kiểm tra lại cùng cách đo.

## Từng loại ở conf0.4 (baseline -> v2)

| Loại | F1 | Precision | Recall | MAE đếm trên tất cả297ảnh |
|---|---:|---:|---:|---:|
| Chuối | .354 -> .507 | .451 -> .431 | .291 -> .614 | .330 -> .387 |
| Táo | .539 -> .909 | .377 -> .931 | .944 -> .888 | .576 -> .064 |
| Cam | .557 -> .847 | .426 -> .784 | .808 -> .919 | .407 -> .104 |

Táo/cam giảm box thừa rõ; táo đánh đổi một phần recall. Chuối tăng phát hiện/ghép đúng box
nhưng precision và MAE đếm chưa cải thiện. Không nói mọi chỉ số của ba loại đều tăng.

## Nhận xét từ ảnh

- `val/baseline_previews/scene_0125.png`: một quả cam mốc nhưng baseline vẽ7box;
  v2 còn2box trên cùng ảnh. Có cải thiện nhưng vẫn thừa1box.
- `val/trial_v2_previews/scene_0004.png`: hai quả táo vẫn có3box; box thứ3 là lỗi model thật.
- `val/trial_v2_previews/scene_0241.png`: nhãn cũ bao cả nải bằng1box nhưng ảnh có nhiều quả;
  dự đoán5box không thể đánh giá là thừa4quả chỉ dựa nhãn này.
  Source là rf_v7__b_f054_png.rf.8aa6a...; bản augment khác b_f054 đã được người dùng duyệt4quả.
  Nhãn đánh giá trong cùng nhóm ảnh gốc còn không thống nhất về từng quả/nải.

4 ảnh val đã duyệt: GT1/4/4/1; baseline1280 dự đoán2/1/1/2;
v1 dự đoán2/3/4/2; v2 dự đoán2/6/3/1. Không dùng chúng như test độc lập.

## Bước tiếp theo

Không train thêm và chưa chạy test373.
Chạy `kiem_tra_kich_thuoc_all3_colab.ipynb` để đo640/1024 trên nguyên297val cho cả3model,
cùng batch/ngưỡng và cùng cách tính. Kết quả1280 dùng lại; không cần upload input/model lần nữa.
Notebook chỉ kiểm tra validation và có chặn chạy toàn bộ dữ liệu trên CPU laptop.

Sau khi chọn model/kích thước bằng val, cần kiểm tra ảnh thực có nhãn từng quả thống nhất,
bao gồm táo/cam/chuối và vật khác. Legacy test vẫn chỉ là regression, không phải benchmark độc lập.
CLIP/teacher/classifier chưa được đo trong các bảng này. Chưa có bằng chứng lỗi nho đã giải quyết.
