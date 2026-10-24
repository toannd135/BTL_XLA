# Nhận dạng trái cây và độ tươi

Web demo cho đồ án: YOLO11 detect quả (chuối, táo, cam), cắt từng quả rồi phân loại tươi/hỏng.

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
