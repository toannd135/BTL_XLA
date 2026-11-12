# Bàn giao 2026-11-12: CLIP kiểm tra loại quả (det_obb_s + CLIP)

## Mục tiêu
Giảm nhận nhầm (vật khác thành quả: dâu, nho; táo <-> cam) không cần train. Người dùng chọn detector `exports/det_obb_s.pt` (OBB Colab 60 epoch).

## Đã làm (chưa commit)
- `backend/clip_check.py`: open_clip ViT-L-14 laion2b_s32b_b82k (model ở `work/weights/hf/`, 2.2 GB gồm cả B-32). `probs_boxes`: crop nới 10%; nếu p(other)>=0.8 thì nhìn lại phần giữa (thu 10%), dùng nếu chắc là quả >= 0.9.
- `backend/detection.py::refine_types`: bỏ box other>=0.8, đổi nhãn khi CLIP>=0.8 (`CLIP_T_SWAP`), cứu box điểm 0.25-0.4 khi CLIP chắc là quả >=0.9. Test `backend/test_detection.py` 18/18.
- `backend/main.py`: `CLIP_CHECK=1`, detector chạy conf 0.25, trả thêm `rescued`. `/api/health` có `clip_check`.
- Frontend: `labels.js passConf` (box rescued tính như 0.4, nếu không bị thanh ngưỡng min 0.4 ẩn), tem nhãn gắn đỉnh polygon (`draw.js`, đã commit ff76bcf0). Build OK.
- Đo: `work/fruitcls/eval_gate.py <det> clip <val|test> t_other,t_swap[,t_rescue]` (env DET_CONF, TWO_VIEW); `tune_prompts.py` (CLIP_ARCH, CLIP_PAD, SETS). Số liệu chi tiết trong CLAUDE.md.
- `own_other/raw`: 1055 ảnh Wikimedia vật khác (chỉ để đo; chưa lọc ảnh có quả thật). `.gitignore` đã thêm own_other/, work/weights/hf/, work/fruitcls/data/.
- Môi trường: sự cố pip (kéo CUDA 13) đã khôi phục: torch 2.5.1+cu121, triton 3.1.0, sympy 1.13.1, cu12 libs. open_clip_torch 2.30.0 + timm cài `--no-deps`. /home còn ~2 GB (cache pip 7.8 GB, chưa xoá - cần người dùng đồng ý).

## Kết quả test (det_obb_s, 373 ảnh)
Đếm đúng 75.6% (không CLIP) -> 79.9%; vật khác báo sai 318 -> 79/1054; nền 3 -> 1/71; chuối sót 32 -> 17.

## Chưa sửa được (cần train detector / dữ liệu thật)
- Bỏ sót táo trong cảnh nhiều quả (own_mixed táo miss ~200/269), thau quả dày, quả cắt lát (nửa cam CLIP 0.75 < 0.9).
- Box chéo trùm nhiều quả chuối.
- Táo vàng đôi khi bị CLIP coi là other.

## Đang chờ
- Người dùng khởi động lại backend (`env -u MODEL_PATH uvicorn main:app --port 8000`) và tải lại frontend để kiểm tra `Desktop/images.jpeg`, `Desktop/ea7372f0a115214b7804.jpg`.
- Quyết định commit; quyết định `pip cache purge`.
- Hướng tiếp: sửa nhãn `own_mixed/` (95 ảnh) rồi fine-tune detector để bớt bỏ sót.
