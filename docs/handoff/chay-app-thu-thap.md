# Bàn giao: chạy app + thu thập ảnh webcam

## Mục tiêu
Chạy web demo nhận dạng trái cây (backend FastAPI + frontend Vite) và kiểm tra việc thu thập ảnh webcam vào `own_cam/`.

## Nhánh/trạng thái git
- Không xác định được: lệnh git không trả output ở `BTL_XLA/` (có thể không phải repo đang dùng). Chưa commit gì trong phiên này.
- Không có thay đổi mã nguồn nào trong phiên này.

## Đã xong
- Kiểm tra cấu trúc, README, môi trường backend (uvicorn, fastapi, ultralytics có sẵn), model trong `exports/`.
- Backend (`uvicorn main:app --port 8000`) và frontend (`vite`, cổng 5173) đang chạy do người dùng khởi động.
- Xác nhận phiên thu thập mới `cam_20261108-200211` đã lưu 1 ảnh lúc 20:02:14; tổng 15 ảnh.

## Quyết định đã chốt
- Khởi động app: đưa lệnh cho người dùng tự gõ bằng `! <lệnh>`, không chạy nền bằng agent.
- Thu thập qua `POST /api/collect` (frontend gửi khung hình), lưu vào `own_cam/`.

## Tệp quan trọng
- `fruit/README.md:5` hướng dẫn chạy backend/frontend.
- `fruit/backend/main.py:23-24` đường dẫn model; `:37` CORS; `:154` endpoint `/api/collect`.
- `fruit/backend/collect.py:21` thư mục xuất `own_cam`; `:1-10` quy tắc sinh nhãn.
- `fruit/frontend/.env` `VITE_API_URL=http://localhost:8000`.

## Việc còn lại
- Người dùng xác nhận trang web đang bật thu thập; nếu không có ảnh mới thì kiểm tra log backend và nút thu thập trên web.
- Nếu muốn, đưa `own_cam/` vào `own/` theo notebook ô 1.1c (chưa làm).

## Đang chờ
- Người dùng: xác nhận trạng thái nút thu thập trên web và dán lỗi nếu có.

## Lệnh kiểm tra
- `curl -s http://localhost:8000/api/health`
- `curl -s http://localhost:8000/api/collect/stats`
- `ls -lt fruit/own_cam/images | head`
- `ps -eo pid,etime,args | grep -E "uvicorn|vite"`
