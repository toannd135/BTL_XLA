# Member clone/pull về và chạy

## Tài nguyên trong repo

Repo được bố trí để không phụ thuộc các thư mục nằm cạnh `BTL_XLA` trên máy bàn giao:

- Code backend/frontend, `tools/`, notebook, `docs/`, file cấu hình mẫu và khóa phiên bản frontend `package-lock.json`.
- `exports/det_C_candidate.pt`, `exports/det_finetune_best.pt`, `exports/cls_finetune_best.pt`: ba model thực, không phải đường dẫn tới file máy khác. Bản được khuyến nghị cho app là `det_C_candidate.pt`; rotation_v1 không được nhận.
- `local_dataset/det/`: ảnh và nhãn detection, gồm train/val/test/train_ft và các phần dữ liệu đang có.
- `own_cam/`, `own_neg/` và các thư mục nguồn ảnh đã có trong repo.
- `work/**/*.py`, metadata CSV/JSON audit và `work/audit/quarantine_train.txt`. Script tự sinh lại danh sách train/YAML theo đường dẫn trên máy mới.

`.gitignore` loại `.venv`, `node_modules`, cache, `.env`, Kaggle credential, `out`, `runs`, file ZIP và các file trung gian sinh ra trong `work`. Không chặn `.pt`, `.onnx` hoặc dataset nữa. Các thư mục nguồn ngoài repo trên máy bàn giao vẫn được giữ; repo có bản sao riêng.

Phạm vi dữ liệu đang có là detection. Máy bàn giao không có `local_dataset/cls`, nên repo không kèm trọn bộ dữ liệu để train lại classifier; model classifier đã có để chạy suy luận. Các bàn giao thí nghiệm lịch sử có thể nhắc tới backup weights/ảnh ghép không còn trên máy này. Không tuyên bố tái lập mọi thí nghiệm cũ từ repo hiện tại.

## Windows: cài và kiểm tra

Sau khi clone/pull, mở PowerShell ở thư mục repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt opencv-python-headless
.\.venv\Scripts\python.exe tools\check_repo_resources.py
```

Lệnh kiểm tra chỉ đọc file, không train và không cần GPU. Nếu thiếu model/dataset, kiểm tra người push đã commit cả các file đó chưa; chỉ sửa `.gitignore` không tự đưa file vào commit. Nếu model là Git LFS pointer thì cần `git lfs pull` trước khi chạy app. Cấu hình bàn giao hiện tại dùng file model thông thường, chưa chuyển lịch sử sang LFS.

## Chạy backend trên CPU

Tại thư mục repo:

```powershell
$env:MODEL_PATH = "$PWD\exports\det_C_candidate.pt"
$env:HALF = "0"
$env:CPU_THREADS = "2"
.\.venv\Scripts\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Kiểm tra `http://127.0.0.1:8000/api/health`: model detector phải là `det_C_candidate.pt`. App vẫn có giới hạn nhận diện nhiều quả/chồng nhau như mô tả trong tài liệu tối ưu; đủ tài nguyên chạy không có nghĩa model đã đếm đúng mọi ảnh.

## Chạy frontend

Mở PowerShell thứ hai trong repo:

```powershell
cd frontend
npm.cmd ci
npm.cmd run dev -- --host 127.0.0.1 --port 5173 --strictPort
```

Mở `http://127.0.0.1:5173`. API mặc định trỏ tới localhost:8000; nếu cần đổi địa chỉ, chép `frontend/.env.example` thành `frontend/.env` rồi sửa `VITE_API_URL`. File `.env` riêng không commit.

## Công cụ dữ liệu

Các công cụ chuẩn bị dữ liệu ưu tiên `local_dataset` bên trong repo; vẫn hỗ trợ cấu trúc nguồn nằm ngoài repo qua tham số đường dẫn hoặc fallback cũ.

Tạo lại bộ duyệt nhãn (không GPU):

```powershell
.\.venv\Scripts\python.exe tools\instance_review.py prepare
Invoke-Item .\out\instance_review_v1\review.html
```

Nếu thư mục đầu ra đã tồn tại, chọn phiên bản mới bằng `--out`; không ghi đè bản đã duyệt. Xem [hướng dẫn nhãn từng quả](duyet-nhan-tung-qua.md). Các ảnh diagnostic trong Pictures của máy bàn giao không tự đưa vào repo; công cụ vẫn chạy với dataset trong repo.

Muốn tạo gói Colab detection C từ nguồn đã có:

```powershell
.\.venv\Scripts\python.exe tools\prepare_c_colab.py
```

Không cần train lại chỉ để chạy app. Chưa nên chạy tiếp recipe rotation_v1 để kỳ vọng cải thiện vì lượt này đã không đạt val/test.

## Người bàn giao: commit và push

Chạy ở repo sau khi đã kiểm tra các tài nguyên:

```powershell
git status --short
git add .
git diff --cached --stat
git commit -m "Include runnable app, models, detection data and handoff docs"
git push -u origin HEAD
```

Các lệnh này do người bàn giao thực hiện; chưa được chạy thay trong quá trình sửa ignore. Không cần force-push hoặc xóa lịch sử. File ZIP sinh ra bị ignore để tránh đưa thêm bản sao dữ liệu nặng vào Git. Với tài nguyên mới lớn hơn giới hạn file thông thường của dịch vụ Git, cần chọn Git LFS hoặc phân phối riêng thay vì bỏ ignore ZIP rồi push trực tiếp.
