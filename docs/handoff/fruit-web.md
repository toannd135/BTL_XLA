# Bàn giao: web nhận dạng trái cây (React + antd + FastAPI)

## Trạng thái
- Backend `backend/main.py`: detect (`exports/det_finetune_best.pt`) -> crop (pad 10%) -> classify fresh/rotten (`exports/cls_finetune_best.pt`), `agnostic_nms=True`, warm-up, `threading.Lock` quanh predict. Kiểm tra đầu vào: rỗng 400, >10MB 413 (chỉ đọc tối đa 10MB+1), >40M pixel 413 (đọc kích thước bằng Pillow trước khi decode), decode lỗi 400. CORS lấy từ env `CORS_ORIGINS` (mặc định localhost + 127.0.0.1 :5173).
- Frontend 2 tab: "Ảnh" (`ImagesView.jsx`: 1 hoặc nhiều ảnh, hàng đợi tự nhận dạng, khung xem + ←/→, dải ảnh thu nhỏ, CSV chống công thức Excel) và "Camera và video" (`LiveView.jsx`: video có controls + canvas đè, tốc độ, tua; webcam có chọn camera, tự cập nhật khi cắm/rút thiết bị, huỷ lần mở đang chờ quyền khi đổi nguồn/rời tab/bấm lặp, báo khi webcam bị ngắt hoặc bị từ chối quyền).
- `message` dùng `App.useApp()` (main.jsx bọc `<AntApp>`), hết cảnh báo antd.
- `vite.config.js` bật `watch.usePolling`: ổ NTFS ở /mnt làm Vite lỡ sự kiện sửa file.

## Đã kiểm chứng 2026-10-18 (Playwright + Chrome, script ở scratchpad phiên 9c7d53f5)
- phone_e2e 10/10 (Chrome giả 2 camera + giả cắm/rút điện thoại), webcam_e2e 21/21 x3 lần (camera giả phát ảnh táo `cam.y4m`: nhận dạng ~29 khung/s, các ca đua), images_e2e 20/20, e2e (video + EXIF) 14/14, không lỗi/cảnh báo JS. `npm run build` OK. curl 11 ca đầu vào lạ OK.

## Cách chạy
- `cd backend && uvicorn main:app --port 8000` ; `cd frontend && npm run dev` (cần `.env` từ `.env.example`).
- Playwright: `~/.npm/_npx/9833c18b2d85bc59/node_modules/playwright`, `executablePath: /usr/bin/google-chrome`; webcam giả: `--use-fake-ui-for-media-stream --use-fake-device-for-media-stream[=device-count=2] --use-file-for-fake-video-capture=x.y4m` (file y4m làm device-count bị bỏ qua).

## Lưu ý
- Antd Dragger lặng lẽ bỏ file không khớp `accept` khi kéo thả -> ô ảnh nhận cả `video/*` rồi chuyển tab.
- OpenCV 5.0 trong máy KHÔNG decode AV1 -> xử lý video phía trình duyệt.
- `own/videos /apple/v14044g50000d7ll7unog65ukjqanr40.mp4` bị ghi đè thành JPEG lúc 2026-10-18 09:13, chưa khôi phục.

## Đang chờ
- Người dùng thử với webcam/điện thoại thật (Android 14+ chế độ USB "Webcam" hoặc DroidCam/Iriun).
- Chưa commit git (toàn bộ repo untracked).
