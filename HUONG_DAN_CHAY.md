# Chạy detection (chuối / táo / cam + tươi / hỏng)

Repo đã kèm 2 model trong `exports/` (`det_finetune_best.pt`, `cls_finetune_best.pt`). Không cần tải dữ liệu.

```bash
git clone <url-repo> && cd fruit
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt opencv-python-headless
python run_detection.py <anh.jpg | thu_muc_anh>       # ảnh kết quả nằm trong out/
```

Tuỳ chọn: `--conf 0.4` (mặc định, hạ 0.25–0.35 nếu bỏ sót chuối).

## Chạy web
```bash
cd backend && uvicorn main:app --port 8000      # terminal 1
cd frontend && npm install && cp .env.example .env && npm run dev   # terminal 2 → http://localhost:5173
```
Có GPU NVIDIA thì cài torch bản CUDA trước; không có GPU vẫn chạy trên CPU (đặt `HALF=0` cho backend).
