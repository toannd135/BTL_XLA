# Nhận dạng trái cây và độ tươi

Web demo cho đồ án: YOLO11 detect quả (chuối, táo, cam), cắt từng quả rồi phân loại tươi/hỏng.

## Cài đặt

```bash
git clone <url-repo> && cd fruit
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt opencv-python-headless
```

Repo đã kèm 2 model trong `exports/` (`det_finetune_best.pt` detect, `cls_finetune_best.pt` phân loại tươi/hỏng). Có GPU NVIDIA thì cài PyTorch bản CUDA trước; không có GPU vẫn chạy được trên CPU (đặt `HALF=0` khi chạy backend).

## Chạy detection bằng dòng lệnh

```bash
python run_detection.py <anh.jpg | thu_muc_anh>        # thêm --conf 0.4 để đổi ngưỡng
```

In danh sách quả (loại, tươi/hỏng, độ tin cậy) và lưu ảnh vẽ box vào `out/`. Ngưỡng mặc định 0.4; hạ xuống 0.25-0.35 nếu bỏ sót chuối.

## Train lại detector

Dữ liệu train (~270MB) không nằm trên git. Nhận file `fruit_train_data.zip` từ người chia sẻ (Drive/USB), giải nén vào gốc repo:

```bash
unzip fruit_train_data.zip       # tạo local_dataset/det, own_neg, own_cam, work/audit
python train_detection.py        # fine-tune từ exports/det_finetune_best.pt, 25 epoch, batch 16
```

Tuỳ chọn: `python train_detection.py <model.pt> <tên_run> <epochs> <batch> <neg_rep (≤2)>`.

- Cần GPU NVIDIA. VRAM 6GB mà báo hết bộ nhớ thì hạ batch xuống 8; không chạy việc khác trên GPU lúc train.
- Kết quả ở `runs/exp/<tên_run>/weights/{best,last}.pt`. Cuối script tự in so sánh bản đang dùng và bản mới (số ảnh nền báo sai, mAP50, mAP50-95 trên test). Model cũ không bị ghi đè.
- Dùng `last.pt`, không dùng `best.pt` (thường rơi vào epoch 1 vì tập val không có ảnh nền). Muốn áp dụng bản mới: copy `last.pt` đè `exports/det_finetune_best.pt`, rồi khởi động lại backend.
- Bị ngắt giữa chừng: `python -c "from ultralytics import YOLO; YOLO('runs/exp/<tên_run>/weights/last.pt').train(resume=True)"`.

## Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Mặc định dùng `exports/det_finetune_best.pt` và `exports/cls_finetune_best.pt`.
Đổi model bằng biến môi trường `MODEL_PATH` (detect) và `CLS_MODEL_PATH` (classify).

- `GET /api/health`
- `POST /api/predict` (form-data, trường `file`)

## Frontend (React + Ant Design)

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Mở http://localhost:5173

Gồm 2 tab:
- **Ảnh**: thả một hoặc nhiều ảnh (tối đa 50, mỗi ảnh ≤ 10MB), hệ thống tự nhận dạng lần lượt. Xem từng ảnh lớn (chuyển ảnh bằng phím ←/→), dải ảnh thu nhỏ có khung, tổng hợp tươi/hỏng, tải ảnh kết quả, xuất CSV.
- **Camera và video**: file video (phát, tua, đổi tốc độ, đổi/bỏ video) hoặc webcam; khung hình được gửi lên server liên tục, hiện số khung/giây, lưu khung hình. Máy có nhiều camera thì chọn trong danh sách; danh sách tự cập nhật khi cắm/rút thiết bị.

Dùng camera điện thoại: cắm cáp USB, trên điện thoại chọn chế độ USB là "Webcam" (Android 14 trở lên) hoặc dùng app webcam như DroidCam, Iriun. Điện thoại hiện thành một camera trong danh sách.

Ngưỡng độ tin cậy chỉnh chung cho cả 2 tab (mặc định 50%). Quả nào có độ tin cậy tươi/hỏng dưới 60% được ghi là "chưa chắc".

Server chỉ nhận trang web mở từ `http://localhost:5173` hoặc `http://127.0.0.1:5173`. Mở từ địa chỉ khác thì đặt biến môi trường `CORS_ORIGINS` (phân cách bằng dấu phẩy) khi chạy backend và sửa `VITE_API_URL` trong `frontend/.env`. Webcam chỉ dùng được khi trang mở bằng localhost hoặc https.
