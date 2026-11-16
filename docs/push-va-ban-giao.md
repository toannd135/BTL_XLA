# Push tài nguyên và hai kết quả detector

`.gitignore` đã mở code, model, dataset, notebook, tài liệu, báo cáo, ZIP, `out/`, `runs/`, dữ liệu thô
và các tài nguyên trong `work/`. Giữ bỏ qua môi trường cài đặt, cache, credential và bản sao khôi phục Git.
Không xóa hay thay model/dataset gốc khi mở các quy tắc này.

Ba ZIP hiện vượt100MiB được quản lý bằng Git LFS qua `.gitattributes`; các model hiện tại dưới100MiB
vẫn là file Git thông thường. Không viết lại lịch sử hoặc chuyển các model cũ sang LFS trong tác vụ này.
GitHub chặn file thường vượt100MiB; xem [giới hạn file](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Git LFS có hạn mức lưu trữ/băng thông của tài khoản; xem [Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-git-large-file-storage).

## Tình trạng hai lượt mới

Xem [báo cáo V1/V2](../reports/detection_trials/README.md) và `exports/trial_models.json`.
Đã có đủ trọng số V1 và V2 (`exports/det_obb_trial_v1.pt`, `exports/det_obb_trial_v2.pt`), SHA256 khớp báo cáo.
Kết quả JSON, ảnh validation, PR/F1 và confusion matrix của V2 đã có trong repo làm việc.
Không tự thay model mặc định của backend.

## Người push chạy tại thư mục BTL_XLA

Git LFS đã được khởi tạo local trên máy bàn giao; `core.longpaths=true` chỉ được đặt cho repo này
để Git đọc được những đường dẫn dataset dài trên Windows.

```powershell
git lfs install --local
git status --short
git add .
git diff --cached --stat
git lfs status
git commit -m "Include latest detector results, resources and LFS inputs"
git push origin main
```

Lệnh commit/push do bạn thực hiện. Không dùng force-push và không cần viết lại lịch sử.
`git add --dry-run --all` đã được kiểm tra; đây là đọc thử, không stage toàn bộ nội dung thay bạn.
Đừng bỏ `.gitattributes` khỏi commit, nếu không các ZIP lớn không được đóng gói đúng bằng LFS.

## Thành viên khác

Cần cài Git LFS trước khi tải ZIP từ repo. Sau clone/pull:

```powershell
git lfs install --local
git lfs pull
```

Môi trường Python/node_modules tự cài theo README. CLIP cache được tải lần đầu, không commit cache theo máy.
Dataset classifier gốc chưa có trên máy bàn giao; bỏ quy tắc ignore không tạo ra dữ liệu đang thiếu.

Repo lịch sử vẫn có9file dưới `own/videos ` (tên thư mục có dấu cách cuối). Những đường dẫn đó được
sparse-checkout bỏ qua trên máy Windows hiện tại; tác vụ này không xóa hoặc đổi tên chúng.
GitHub vẫn nhận commit, nhưng clone/checkout đầy đủ trên Windows còn cần xử lý đường dẫn này hoặc
loại thư mục đó bằng sparse checkout. `core.longpaths` giải quyết độ dài đường dẫn, không giải quyết dấu cách cuối.
