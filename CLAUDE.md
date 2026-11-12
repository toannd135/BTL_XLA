# Nhận dạng trái cây và độ tươi (BTL_XLA / fruit)

Hai tầng YOLO11: detector (banana, apple, orange) rồi classifier (fresh, rotten) trên từng crop. Web: FastAPI + React/Vite.
Máy: RTX 3060 Laptop 6GB, Ultralytics 8.4.180. Máy hay nóng (CPU ~97–100°C khi train): kê cao máy, không chạy GPU song song lúc train (6GB dễ OOM).

## Chạy app
- Backend: `cd backend && uvicorn main:app --reload --port 8000` (biến: `CONF` 0.40, `NMS_IOU` 0.6, `HALF` 1, `MODEL_PATH`, `CLS_MODEL_PATH`, `COLLECT_DIR`, `CORS_ORIGINS`)
- Frontend: `cd frontend && npm run dev` (cổng 5173). Không có tên model trong frontend; chỉ gọi API.
- Khởi động app/git: đưa lệnh cho người dùng tự gõ `! <lệnh>`.

## Trạng thái model (cập nhật 2026-11-08)
- **2026-11-12: CLIP kiểm tra loại quả** (`backend/clip_check.py`, open_clip **ViT-L-14 laion2b_s32b_b82k** mặc định, ~9 ms/crop, 1.6 GB VRAM; `CLIP_ARCH=ViT-B-32` cho máy yếu; zero-shot, không train; model lưu `work/weights/hf/`). L-14 so với B-32 (test): ảnh vật khác báo sai 69 vs 110/1054, đếm đúng 77.5% vs 78.8%, own_mixed táo gọi là cam/chuối 9/11 vs 14/20 (chưa CLIP). Chỉnh prompt (`work/fruitcls/tune_prompts.py`, `prompt_sets.py`) không cải thiện trên test, giữ bộ gốc. **Cứu box gần ngưỡng** (2026-11-12): detector chạy conf `RESCUE_CONF` 0.25, box điểm < 0.4 chỉ giữ khi CLIP chắc là quả >= `CLIP_T_RESCUE` 0.9 (val: đếm đúng 79.7% -> 81.7%; test 77.5% -> 79.9%, chuối sót 32 -> 17, nền 0 -> 1/71). **Hai lần nhìn** (`ClipChecker.probs_boxes`): box bị coi là other thì nhìn lại phần giữa (thu 10%), dùng kết quả đó nếu chắc là quả >= `CLIP_T_INNER` 0.9 (sửa cam bị dâu che; bản không siết làm ảnh vật khác báo sai 69 -> 126). Thu nhỏ crop cho mọi box: mất chuối chéo, không dùng. Test cuối: vật khác báo sai 79/1054. `CLIP_T_SWAP` hạ 0.9 -> 0.8 (val như nhau). Box cứu có `rescued: true` trong API; frontend (`labels.js passConf`) tính nó như đạt ngưỡng 0.4, nếu không thanh ngưỡng (min 0.4) sẽ ẩn. Số dưới đây là B-32: Sau detector: p(other)>=0.8 thì bỏ box, loại quả khác detector với p>=0.9 thì đổi nhãn (`detection.refine_types`). Biến `CLIP_CHECK` (1), `CLIP_T_OTHER`, `CLIP_T_SWAP`. Đo `work/fruitcls/eval_gate.py <det> clip test`: test đếm đúng 75.6% -> 78.8%, chuối gọi là cam 13 -> 1, nền 3 -> 0/71, 1054 ảnh vật khác (`own_other/`, Wikimedia, chỉ để đo) báo sai 318 -> 110; đánh đổi: xoá nhầm táo 4, cam 1. ~5 ms/crop. Không sửa được bỏ sót quả hay box sai hình (cần detector tốt hơn).
- **Cài gói Python: luôn `--no-deps`** (open_clip mới kéo torch/CUDA 13, /home gần đầy, gỡ cu13 xoá luôn thư viện cu12). Môi trường chuẩn: torch 2.5.1+cu121, triton 3.1.0, sympy 1.13.1.
- **2026-11-12: backend mặc định dùng `exports/det_obb_s.pt`** (YOLO11s-OBB train Colab 60 epoch, người dùng chọn). So với C: táo xanh stock 2/3 (C 0/3), nền sai 3/71 (C 1), MAE đếm test táo 0.25/cam 0.16/chuối 0.71 (C 0.18/0.14/0.44; chuối test có nhãn nải nên chỉ so tương đối). Kết quả: `work/obb/eval/obb_colab.json`. Nho vẫn chưa có trong ảnh nền.
- 2026-11-11: backend mặc định dùng `exports/det_C_candidate.pt`, `QUALITY_IMGSZ` 640 (960 đếm chuối thừa ~2.4x). Đếm chuối: `work/eval_banana_count.py`, xem `docs/handoff/dem-chuoi.md`.
- `exports/det_finetune_best.pt` = **thí nghiệm C** (`runs/exp/det_C_neg/weights/last.pt`, epoch 11/25; fine-tune từ B, lr0 0.002, thêm 287 ảnh nền own_neg x2, bỏ own_cam). Bản B: `exports/det_B_clean_backup.pt/.onnx`; bản A: `det_A_baseline_backup.*`. Chọn last.pt vì best.pt (epoch 1) do val không có ảnh nền.
- `exports/cls_finetune_best.pt` = classifier yolo11n-cls (acc 0.999 public, 27/28 own). Chưa thấy lý do train lại.
- Backend dùng FP16 (`quantize=16`) và NMS 0.6, chưa chạy thử với bản C (cần khởi động lại backend).

| Test 373 ảnh (conf 0.25, NMS 0.7) | A cũ | B | C (đang dùng) |
|---|---|---|---|
| mAP50 / mAP50-95 | 0.903 / 0.758 | 0.910 / 0.759 | 0.889 / 0.732 |
| AP50 chuối / táo / cam | 0.802 / 0.927 / 0.979 | 0.836 / 0.935 / 0.960 | 0.805 / 0.905 / 0.958 |
| Ảnh nền báo sai (71 ảnh giữ riêng) | - | 34 (48%) | 3 (4%) |
Chi tiết: `work/audit/eval_A_baseline.json`, `eval_B_clean.json`, `fp_*.csv` (mỗi box thừa kèm `error_type`).

- **Ngưỡng tin cậy chốt 0.40** (model C, quét 0.15-0.6): F1 tổng 0.82, P/R 0.80/0.84, nền báo sai 1/71; chuối F1 0.73, táo 0.89, cam 0.90. Backend `CONF=0.4`, frontend thanh trượt mặc định 0.4 (min 0.4). Đo so sánh thí nghiệm vẫn dùng conf 0.25 nếu cần đối chiếu số cũ.

## Phát hiện quan trọng
- Box chuối thừa trên test (~104) chủ yếu là **nhãn lẫn lộn nải/từng quả** (42 box từng quả trong nhãn nải, 27 trùng lặp, 23 lệch, chỉ ~11 nền thật). Quy ước đã chốt: **gán từng quả**.
- Rà nhãn: `work/audit_det_labels.py` -> `work/audit/det_label_audit.csv` (bunch/nested/missing/empty-label). Không có rò rỉ nguồn Roboflow giữa train/val/test; ảnh gần giống (dHash) ~1%.
- **Nhận nhầm vật khác thành quả là thật và nặng**: trên 358 ảnh nền sạch (`own_neg/`), A báo quả ở 146 ảnh, B ở 153 ảnh (~42%). Cao nhất: ớt chuông, bóng, cốc, mặt người, bàn tay, mặt bàn. Tập test hiện không phản ánh điều này.
- Ảnh trong `own_cam/` do chính model gán nhãn (chỉ giữ khi detector + YOLO11l đồng ý, classifier >= 0.8) nên thiên lệch ca dễ, gần như toàn `apple_rotten`. Không cải thiện được lỗi nhận sai nếu chỉ train thêm chúng.

- **Cảnh nhiều loại quả chung khung gần như không có trong dữ liệu** (train: 21/2365 ảnh có >=2 loại, test: 2/373). Ảnh stock chuối+táo xanh+cam (`work/diag/green-apples-...`): cả A/B/C, mọi imgsz/TTA đều KHÔNG ra box táo nào (conf>=0.10), táo xanh bị đẩy thành "orange 0.14-0.18". Không phải lỗi backend/ngưỡng/NMS/mapping. Công cụ: `python work/diag_image.py <ảnh>`.

- `own_mixed/` (2026-11-08): 95 ảnh nhiều loại quả tải từ Wikimedia (giấy phép CC/PD, nguồn ở `own_mixed/sources.csv`; script `work/fetch_mixed.py`), 43 ảnh YOLO11l thấy >=2 loại (`work/label_mixed.py`). **Nhãn nháp KHÔNG đủ sạch để train** (duyệt mắt: nhầm lê/mận thành táo, box bao cả chùm, khay quả cắt lát bị gán nhãn nhiều box). Chưa dùng vào train; cần người duyệt/sửa nhãn hoặc dùng ảnh ghép tổng hợp.

## Dữ liệu
- `local_dataset/det/{train,val,test,train_ft}`, `local_dataset/cls/...` (nguồn chính, đừng sửa val/test).
- `own_cam/` (33 ảnh thu thập webcam, nhãn 6 class), `own_neg/` (358 ảnh nền sạch từ Wikimedia Commons, đã lọc bằng YOLO11l), `own_neg_flagged/` (126 ảnh YOLO11l thấy quả thật, KHÔNG dùng làm nền).
- `work/audit/quarantine_train.txt`: 50 ảnh train nghi nhãn nải/lồng, loại khỏi train ở B (không xoá file).

## Kế hoạch tiếp theo
0. **D1 đã đo 2026-11-12 -> KHÔNG nhận.** D1 best (`runs/exp/det_D_synth/weights/best.pt`): mAP50 0.892 (C_candidate 0.895), AP táo 0.879 (C 0.911), nền báo sai 9/71 ở conf 0.4 (C 0/71); nhưng ra đủ 3 táo xanh trên ảnh stock. Ảnh ghép dạy thêm táo xanh nhưng tăng báo nhầm vật khác. Nho chưa có trong ảnh nền (app nhận nho thành táo 91-93%). `runs/obb/obb_s1` dừng ở epoch 17/60, không dùng cho app. Mô tả D1 cũ: = fine-tune từ C + 700 ảnh ghép nhiều loại quả (`work/exp/det_synth.py`, `runs/exp/det_D_synth`). Mai: so C vs D1 theo `docs/handoff/thi-nghiem-C-D1.md` rồi quyết định nhận hay không.
1. ~~**Thí nghiệm C**~~ XONG 2026-11-08 (xem bảng trên; đánh đổi: mAP50 -0.02, chuối -0.03, nền sai 48%->4%). Có thể thử C2 với neg_rep=1 để giữ mAP chuối. Mô tả gốc: bắt đầu từ cấu hình B, thêm ảnh nền `own_neg/` (giữ 20% theo khối làm bộ kiểm tra nền độc lập, không train), không nhân bản quá 2x. Script có sẵn: `work/exp/det_neg.py` (đã gộp own_cam + nền; cần thêm loại 50 ảnh quarantine để đúng cấu hình B; chỉnh `neg_rep` <= 2). Đo: tỷ lệ báo quả sai trên ảnh nền giữ riêng (hiện ~42%), mAP/recall ba class, banana P/R. Dùng `work/eval_det.py <weights> <tag>`.
2. **Clean test bổ sung (cần người)**: ~30 ảnh test bị cắm cờ ở `work/audit/det_label_audit.csv` (split=test): người xác minh/sửa nhãn thành bản sao riêng, KHÔNG thay test gốc.
3. **Thí nghiệm D**: thêm ảnh webcam chuối/táo/cam đã kiểm tra nhãn vào `own_cam/` (cần người cung cấp; ưu tiên chuối). Giữ bộ test webcam độc lập.
4. Sau khi dữ liệu ổn: thử imgsz 512/640/768 (batch 4/4/1 trên 6GB); audit crop classifier (`CROP_PAD` 0.10) và đánh giá fresh/rotten theo từng loại quả; TensorRT FP16 sau cùng.
5. Mỗi thí nghiệm: cùng val/test, cùng seed 42, cùng conf 0.25 và IoU khi so; lưu `eval_<tag>.json`. Mỗi lần train yolo11s 80 epoch ~35 phút.

## Lưu ý kỹ thuật
- Chạy `eval_det.py`/predict GPU khi đang train có thể OOM; resume: `YOLO(last.pt).train(resume=True)`.
- `pkill -f "<mẫu>"` trong Bash tự giết shell: dùng `pgrep -f "[m]ẫu" | xargs kill`.
- Notebook `fruit_yolo_retrain_20261018.ipynb`: chạy lại ô 5.1 sẽ ghi đè `exports/det_finetune_best.*` bằng yolo11n; dùng scripts trong `work/exp/` thay vì notebook cho detector.
- `own_neg/` chưa có trong `.gitignore` (nên thêm). Bàn giao cũ: `docs/handoff/`.
