# Tối ưu đường biên instance segmentation

Ngày: 17/11/2026.

## Luồng xử lý

1. YOLO-seg trả mask ở độ phân giải ảnh gốc (`retina_masks=True`).
2. Giữ contour gốc; giản lược tối đa 0,75 pixel, từ chối giản lược nếu diện tích thay đổi trên 2%.
3. OBB/HBB đề xuất vị trí cho **cả chuối, táo và cam**. Gộp các đề xuất trùng,
   bỏ box cụm khi có đủ box con riêng biệt. Với chuối đã có nhiều đề xuất OBB,
   giữ ưu tiên OBB để tránh box HBB bao cả nải làm hỏng việc tách từng quả.
4. SAM2 nhận một điểm trong mỗi quả, các điểm âm dọc thân quả bên cạnh và box tương ứng.
   Các prompt của một cụm được xử lý chung trong một lần predict.
   Ở vùng nhiều mask tranh chấp, SAM2 chọn theo logit của ảnh trước khi nhị phân hóa.
5. Chỉ thay mask cụm khi tất cả mask con hợp lệ: đủ diện tích, chứa điểm của mình,
   loại điểm của quả khác, chồng lấp không quá 10% diện tích quả nhỏ hơn,
   và phủ được ít nhất 85% mask gốc. Điểm chất lượng SAM tối thiểu 0,55.
   Đây là các ngưỡng bảo vệ ban đầu, chưa phải ngưỡng tối ưu đã đo trên dữ liệu đã duyệt.
6. Nếu SAM không tách đạt, thử YOLO-seg trên crop từng đề xuất trong ngân sách cho phép.
   Chỉ thay cụm khi toàn bộ mask con hợp lệ; nếu vẫn không đạt thì giữ mask gốc và báo duyệt.
   Không chia pixel theo khoảng cách tới trục box.
7. Với đề xuất ngoài các mask hiện có, nhìn lại crop có đệm 30% bằng YOLO-seg.
   Chỉ thêm quả khi model xác nhận đúng loại, confidence ít nhất 0,5 và có mask phù hợp.
   Kiểm tra biên crop để tránh cắt một quả thành nhiều instance; chặn mask trùng.
   Các đề xuất quá nhỏ so với những quả cùng loại được giữ lại để tránh đếm nho thành táo.
   Muốn bổ sung một loại chưa có trong kết quả gốc phải có đề xuất HBB đủ mạnh và mask xác minh.
   Đây là các bộ lọc ban đầu; không thay thế dữ liệu đã duyệt hoặc kiểm chứng loại bằng CLIP.
8. Frontend vẽ trực tiếp polygon, bo nét nối; không uốn spline làm biên phình ra khỏi mask.

Không tái dựng phần quả bị che. Phiên bản này dùng một polygon cho mỗi instance;
mask SAM có nhiều mảnh đáng kể được giữ lại để duyệt thay vì tự nối các mảnh.

## Cấu hình

Backend tự ưu tiên `yolo11s-seg.pt` ở thư mục dự án nếu có, nếu không dùng OBB cũ.
`MODEL_PATH` luôn được ưu tiên hơn cơ chế chọn mặc định. Checkpoint pretrained chính thức
có 80 lớp COCO; backend chỉ lấy banana/apple/orange và giữ đúng ID 46/47/49.
Không coi checkpoint này là model đã fine-tune bằng mask trái cây của dự án.

| Biến môi trường | Mặc định | Tác dụng |
|---|---|---|
| `SEG_IMGSZ` | `640` | Kích thước đầu vào YOLO-seg |
| `SEG_RETINA_MASKS` | `1` | Trả mask ở độ phân giải ảnh gốc |
| `SEG_SIMPLIFY_EPS` | `0.75` | Sai lệch contour tối đa do giản lược, theo pixel ảnh |
| `SAM_REFINE` | `1` | Dùng SAM cho cụm chuối/táo/cam khi có trọng số tại máy |
| `SAM_MODEL_PATH` | Tự tìm tiny rồi base | Chọn checkpoint SAM2 khác |
| `OBB_ENSEMBLE_PATHS` | Ba checkpoint OBB trong `exports/` | Danh sách checkpoint OBB phân tách bằng dấu phẩy |
| `INSTANCE_CROP_CHECKS` | `12` | Tối đa số lượt kiểm tra crop mỗi ảnh; cho phép 0–24 |
| `ENHANCE_IMAGE` | `0` | Bật lại phóng ảnh nhỏ/làm nét/CLAHE khi có phép đo xác nhận lợi ích |

Khi chuối bị gộp mà chưa tách được, API trả `quality.decision =
do_not_auto_count` và `quality.automatic_count_allowed = false`. Đây là
fail-closed: hệ thống không tự trả một số đếm có vẻ hợp lệ nhưng sai bản chất.
Táo/cam không bị khóa bởi cảnh báo chuối nếu không có nhóm chuối pending.

Tự tìm SAM tại `work/weights/sam2.1_t.pt`, sau đó `work/weights/sam2.1_b.pt`.
Không tự tải SAM trong lúc khởi động backend. Với máy chưa có trọng số,
tải từ release chính thức Ultralytics hoặc khởi tạo `SAM` một lần để tải trước.

```powershell
# Chạy từ thư mục BTL_XLA, sau khi đã cài môi trường Python.
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolo11s-seg.pt')"
.\.venv\Scripts\python.exe -c "from pathlib import Path; from ultralytics import SAM; p=Path('work/weights'); p.mkdir(parents=True, exist_ok=True); SAM(str(p/'sam2.1_t.pt'))"
```

Khởi động lại backend sau khi đổi mã hoặc trọng số. `/api/health` hiển thị
`det_model`, `seg_retina_masks`, `sam_refine`, `sam_model` để kiểm tra cấu hình thật.
`SAM_REFINE=0` tắt SAM khi cần giảm bộ nhớ; vẫn có bước xác minh crop và báo duyệt khi chưa tách được.
SAM chỉ suy luận khi có một cụm phù hợp. Ảnh chỉ có táo/cam cũng được xét đề xuất OBB/HBB.
`/api/health` có `instance_refine_classes` và `instance_crop_checks` để xác nhận bản backend mới.
Model teacher chỉ được tải ở luồng OBB, vì luồng segmentation không sử dụng nó.

Ảnh tải lên được giữ nguyên kích thước và màu trước khi model tự resize đầu vào.
Không áp dụng phóng ảnh nhỏ, làm nét và CLAHE cho mọi ảnh nữa: kiểm tra API trên ảnh thật
cho thấy các bước này làm thay đổi kết quả so với suy luận trên ảnh gốc.

## Chồng lấp và cảnh báo

“Chồng lấp mask” đo phần mask trùng với hợp của các mask khác. 0% không có nghĩa
là quả không bị che. Các trường `occlusion_ratio` và `is_occluded` được giữ tương thích
với kết quả cũ, nhưng cũng chỉ biểu thị chồng lấp.

Hai mask hợp lệ được so theo IoU mask, không theo IoU box bao. Chỉ số được tính lại
sau bước SAM và sau khi CLIP bỏ instance, để không tham chiếu quả đã bị loại.
Mask thiếu không được giả thành hình chữ nhật hợp lệ; giao diện dùng box và báo duyệt.
Tính chỉ số trên vùng bao từng mask để giảm cấp phát bộ nhớ ở ảnh lớn.
Khi các detector lệch số quả ở luồng segmentation, backend giữ kết quả bảo thủ và ghi
diagnostics thay vì cộng mù các detection. Mask tách được chưa chứng minh đã tìm đủ mọi quả.
`quality.instance_refinement` ghi riêng số cụm đã tách, cụm chưa tách và quả bổ sung
cho từng loại. Cảnh báo táo/cam không còn dùng thông báo riêng của chuối.

## Kiểm tra và đánh giá

```powershell
cd backend
..\.venv\Scripts\python.exe -m unittest test_detection test_segmentation
```

```powershell
cd frontend
npm.cmd test
npm.cmd run build
```

Benchmark đếm đã sửa ánh xạ nhãn COCO theo `model.names`, dùng cùng bộ kiểm tra chất lượng
với API, đánh giá toàn bộ split theo mặc định. Khi giới hạn ảnh, lấy mẫu ngẫu nhiên có seed,
không lấy các tên đầu tiên vốn chủ yếu là táo. Thiếu file nhãn được báo lỗi.

```powershell
# Chạy từ thư mục BTL_XLA. Bộ này chỉ đo detector, không bao gồm SAM/CLIP/classifier.
.\.venv\Scripts\python.exe tools/benchmark_count_seg.py --model yolo11s-seg.pt --data-split local_dataset/det/val --max-images 12 --output work/audit/seg_boundary_smoke.json
```

Unit test kiểm tra giữ cuống mảnh/chỗ lõm, không tạo mask giả, các mask tách biệt nhưng
box chồng nhau, tách táo/cam, dùng HBB khi thiếu OBB, giữ đúng ID lớp COCO,
chặn mask trùng/thiếu/chất lượng thấp, bỏ box cụm có nhiều box con, xác minh crop,
không đếm thêm proposal chưa có mask, không cắt quả tại biên crop, phục hồi khi crop lỗi,
giữ đề xuất chuối OBB và cập nhật chỉ số sau CLIP.
Frontend test kiểm tra contour được dùng nguyên vẹn khi tô và vẽ viền.

Công cụ xuất ảnh kiểm tra, chạy từ `BTL_XLA`:

```powershell
.\.venv\Scripts\python.exe tools/preview_segmentation.py local_dataset/det/val/images/mbk__banana_20.jpg local_dataset/det/val/images/mbk__banana_21.jpg local_dataset/det/val/images/mbk__mixed_16.jpg
```

Ảnh và JSON nằm trong `out/seg_boundary_review/`. Thêm `--no-sam` để xem mask gốc
và các vùng bị đánh dấu cần tách. Công cụ này không áp dụng CLIP, classifier hoặc enhancement;
không gọi đó là kết quả toàn bộ API. SAM không tách được đáng tin cậy mọi nải chuối:
trong ba ảnh thử có một vùng nải vẫn giữ mask gốc để duyệt.

Kiểm tra trước khi mở rộng: 42 test backend, 2 test canvas, build frontend và
3 request ảnh qua `TestClient`. Sau khi mở rộng cho ba loại, 59 test backend và
11 request ảnh người dùng qua API đều thành công.
Trọng số YOLO11s-seg và SAM2.1 Tiny
đã tải từ release chính thức `ultralytics/assets` v8.4.0. Report thử API tại
`out/seg_boundary_review/api_smoke.json`; report này dùng CPU và tắt CLIP,
nhưng có detector, helper OBB, SAM, detector kiểm chứng và classifier tươi/hỏng.
Ảnh nải `mbk__banana_20` vẫn cần duyệt; `mbk__banana_21` có mask tách nhưng vẫn cần
đối chiếu số quả. Đây là kiểm tra vận hành, không chứng minh đã đếm đủ hoặc biên đạt nhãn chuẩn.

## Kiểm tra bộ ảnh thực tế của người dùng

Lượt mở rộng đã đọc 11 ảnh trong thư mục người dùng cung cấp, không sửa ảnh gốc.
Ảnh táo dày có cả lỗi gộp và lỗi bỏ sót ở rìa: `apple.webp` có 13 vùng ở luồng cũ,
luồng mới bổ sung 8 vùng đã được YOLO-seg kiểm tra lại, trả 21 vùng táo.
**21 là số dự đoán, chưa phải số đúng đã được người gán nhãn xác nhận.**

Các ca kiểm tra chống đếm thừa gồm `cup.jpg` (vẫn 0), `singleBanana.jpg` (vẫn 1)
và `test.webp` (không thêm táo từ các đề xuất OBB trên chùm nho).
Nải `bunchOfBanana.jpg` vẫn chưa tách đạt: giữ mask gốc, cần duyệt.
Không tuyên bố đã giải quyết toàn bộ cảnh quả chồng nhau.

```powershell
.\.venv\Scripts\python.exe tools/preview_segmentation.py --image-dir "$env:USERPROFILE\Pictures\XuLyAnh" --output out/fruit_instances_user/preview
```

Ảnh trước/sau và report API ở `out/fruit_instances_user/`:
`baseline.json`, `api_after.json`, `comparison.jpg`. Kiểm tra trên CPU, `HALF=0`,
CLIP tắt theo cấu hình chạy hiện tại. Thêm suy luận crop làm tăng thời gian xử lý;
ngân sách `INSTANCE_CROP_CHECKS` giới hạn chi phí này. Chưa đo trên GPU.

Các kiểm tra này xác minh hành vi mã. Cần ảnh gốc và mask người duyệt để kết luận
độ chính xác biên trên ảnh người dùng gửi. Cần đo tốc độ trên GPU đích;
SAM và xác minh crop thêm chi phí suy luận ở cảnh đông quả. Chưa train lại trên pseudo-label chưa duyệt.

## Luồng chống đếm thừa sau refinement

Sau khi SAM hoặc YOLO-seg crop thay mask cụm, backend chạy dedup ở mức mask,
không dùng IoU của box. Hai mask cùng loại chỉ bị coi là trùng khi IoU mask
ít nhất 0,60 hoặc mask nhỏ bị mask lớn bao phủ ít nhất 85%; các mask sát nhau
nhưng có vùng nhìn thấy riêng vẫn được giữ lại. Thứ tự kết quả gốc được bảo toàn.

Mask SAM cũng phải khớp proposal tương ứng: ít nhất 35% diện tích mask SAM và
15% diện tích proposal phải giao nhau. Nếu không đạt, toàn bộ cụm bị giữ nguyên
và đánh dấu cần duyệt, thay vì chấp nhận tách một quả thành nhiều instance.

Benchmark chạy toàn bộ luồng segmentation/refinement (không chạy classifier/CLIP):

```powershell
.\.venv\Scripts\python.exe tools\benchmark_instance_pipeline.py `
  --model yolo11s-seg.pt `
  --obb-model exports/det_obb_s.pt `
  --sam-model work/weights/sam2.1_t.pt `
  --data-split local_dataset/det/val `
  --output work/audit/instance_pipeline_results.json
```

Kết quả benchmark này mới được dùng để chọn `NMS_IOU`, `SEG_IMGSZ` và các ngưỡng
refinement; benchmark detector-only không đủ để kết luận pipeline sản phẩm.

## Luồng ensemble cho chuối nhiều quả

Với `mbk__banana_20`, `mbk__banana_24` và `bunchOfBanana.jpg`, cả YOLO-seg và
OBB hiện chỉ sinh một mask/box bao cả nải. Khi không có proposal từng quả,
SAM không thể tự suy ra số quả đáng tin cậy; không đổi NMS hay tăng `imgsz`
để ép tách tự động trong production. Backend hiện chạy ensemble ba OBB
(`det_obb_s`, `det_obb_map95_s1`, `det_obb_map95_s2`), gộp proposal theo hình
học rồi mới đưa vào bước refinement. Không cộng các box trùng của từng model.

Ensemble đã được kiểm tra trên bốn ảnh chuối đại diện và cho lần lượt 1, 1, 1
và 2 instance. Vì các model vẫn đồng thuận ở group box trên các nải khó,
đây là giới hạn của checkpoint hiện có chứ không phải lỗi NMS. Khi ensemble
không có proposal từng quả, API giữ mask gốc và trả diagnostics
`banana_split_pending`; không tự bịa số quả hoặc nhân bản detection.

Muốn thay checkpoint mà không sửa mã, đặt `OBB_ENSEMBLE_PATHS` bằng danh sách
đường dẫn phân tách bằng dấu phẩy rồi khởi động lại backend. Cần benchmark lại
trên ảnh thực tế trước khi dùng checkpoint mới làm mặc định.

Phần công cụ review polygon/SAM-assisted vẫn được giữ trong repository như công
cụ phát triển dữ liệu, nhưng không còn là flow sản phẩm bắt buộc.

### Train model segmentation riêng cho chuối

Model riêng chỉ được train từ polygon chuối đã xác nhận. Không dùng box cả nải
làm nhãn instance. Có thể tạo dữ liệu biến thể xoay/lật để tăng độ bền với nải
chéo; script không sửa dataset nguồn:

```powershell
.\.venv\Scripts\python.exe tools\augment_banana_seg.py `
  --source out\banana_seg_reviewed `
  --out work\banana_seg_augmented --copies 3

.\.venv\Scripts\python.exe tools\train_banana_seg.py `
  --data work\banana_seg_augmented\data.yaml `
  --epochs 80 --imgsz 768 --name banana_seg_v1
```

`train_banana_seg.py` chỉ tạo candidate trong `runs/banana_seg/`, không tự thay
`MODEL_PATH`. Cần so sánh candidate với model hiện tại trên tập test tách riêng,
đặc biệt các ảnh nải nhiều quả, trước khi nối candidate vào production.

<!-- Legacy review commands retained below for dataset-development use only. -->

```powershell
.\.venv\Scripts\python.exe tools\instance_review.py prepare `
  --source-root . --out out\banana_review_base

.\.venv\Scripts\python.exe tools\prepare_banana_review.py `
  --source out\banana_review_base `
  --diagnostics "$env:USERPROFILE\Pictures\XuLyAnh" `
  --out out\banana_instance_review_v1 --limit 80
```

Mở `out/banana_instance_review_v1/review.html`, chọn **Vẽ polygon từng quả**,
bấm theo đường biên nhìn thấy rồi double-click để kết thúc. Polygon được lưu
trực tiếp cho segmentation; box bao chỉ được giữ thêm để tương thích OBB.
Ưu tiên duyệt ảnh có nhiều quả và các ảnh đang bị báo `bunch`/`nested`.
Sau khi duyệt xong có thể xuất tập chuối riêng bằng:

```powershell
.\.venv\Scripts\python.exe tools\instance_review.py build `
  --bundle out\banana_instance_review_v1 `
  --reviews instance_reviews.json `
  --out out\banana_instance_clean_v1 --allow-subset
```

Tập này là đầu vào để fine-tune segmentation/OBB. Không đưa ảnh chưa duyệt
hoặc polygon bao cả nải vào train.

### Gán nhãn bán tự động bằng SAM

Backend có endpoint `/api/segment_prompt` để tạo polygon từ các điểm người
dùng chọn. Khởi động backend với `SAM_REFINE=1`, mở bundle review rồi:

1. Chọn `SAM: điểm trong quả` và click vào phần thân của một quả.
2. Nếu SAM lấy dính quả bên cạnh, chọn `SAM: điểm phủ định` và click vào quả
   bên cạnh (có thể thêm nhiều điểm).
3. Bấm `Tạo mask bằng SAM`, kiểm tra polygon sinh ra, xóa và thử lại nếu sai.
4. Chỉ approve sau khi số mask bằng số quả nhìn thấy.

Điểm xanh là điểm dương, điểm đỏ là điểm phủ định. Cách này vẫn cần một lần
click cho mỗi quả, nhưng không cần tự vẽ toàn bộ đường biên; polygon SAM được
lưu vào JSON và được xuất thành nhãn segmentation khi build dataset.
