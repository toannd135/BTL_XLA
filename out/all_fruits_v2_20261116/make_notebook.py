import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
package = json.loads((HERE / "package_summary.json").read_text(encoding="utf-8"))
old = json.loads((HERE.parent / "accuracy_trial_20261116/finetune_accuracy_trial_colab.ipynb").read_text(encoding="utf-8"))
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": dedent(source).strip() + "\n", "outputs": [], "execution_count": None})


md('''
    # Tối ưu chung chuối / táo / cam — all3 v2

    Đây là lượt thử dữ liệu cân bằng + train 1024, chưa phải model đã được chứng minh tốt hơn.
    Giữ detector YOLO11s-OBB và classifier hiện tại. Không thay app tự động.

    **Chỉ cần upload `fruit_all3_v2_20261116.zip` thẳng vào My Drive của bạn; không giải nén.**
    Gói chứa dữ liệu cả 3 loại, baseline, trial v1 và runner; không cần Drive của bạn khác/GitHub/token.

    Chọn Runtime → Change runtime type → **T4 GPU**. Chạy lần lượt từ trên xuống.
    Lần đầu smoke → train → val. Chỉ chạy test sau khi đã xem val và chốt CHECKPOINT.
''')
code('''
    import torch
    assert torch.cuda.is_available(), 'Chọn T4 GPU trong Runtime → Change runtime type. Không chạy lượt này bằng CPU.'
    print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
''')
md('''
    ## 1. Mount Drive và kiểm tra gói dữ liệu

    Khi runtime mới: chạy lại setup từ đầu; checkpoint nằm trên Drive nên không bị ghi đè.
''')
code('''
    from google.colab import drive
    drive.mount('/content/drive')
    from pathlib import Path
    import hashlib, json, shutil, zipfile

    ZIP_DRIVE = Path('/content/drive/MyDrive/fruit_all3_v2_20261116.zip')
    ZIP_LOCAL = Path('/content/fruit_all3_v2_20261116.zip')
    ROOT = Path('/content/fruit_all3_v2_20261116')
    PROJECT = Path('/content/drive/MyDrive/fruit_all3_v2_20261116/runs')
    EXPECTED_SHA256 = '__SHA__'

    def sha_file(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                value.update(chunk)
        return value.hexdigest()

    assert ZIP_DRIVE.is_file(), 'Upload đúng tên ZIP vào thư mục gốc My Drive của tài khoản vừa mount.'
    if not ZIP_LOCAL.exists():
        print('Copy ZIP từ Drive...')
        shutil.copyfile(ZIP_DRIVE, ZIP_LOCAL)
    assert sha_file(ZIP_LOCAL) == EXPECTED_SHA256, 'ZIP sai phiên bản hoặc chưa upload đủ.'
    ROOT.mkdir(exist_ok=True, parents=True)
    with zipfile.ZipFile(ZIP_LOCAL) as archive:
        assert all((ROOT / item.filename).resolve().is_relative_to(ROOT.resolve()) for item in archive.infolist())
        archive.extractall(ROOT)
    manifest = json.loads((ROOT / 'manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        assert sha_file(ROOT / record['image']) == record['image_sha256'], record['image']
        assert sha_file(ROOT / record['label']) == record['label_sha256'], record['label']
    for record in manifest['models'].values():
        assert sha_file(ROOT / record['path']) == record['sha256']
    print(json.dumps(manifest['summary'], ensure_ascii=False, indent=2))
    print('Kết quả lưu trên Drive:', PROJECT)
'''.replace('__SHA__', package['sha256']))
md('''
    ## 2. Thư viện

    Giữ Torch/CUDA có sẵn; pip dùng --no-deps. Không nâng cấp toàn bộ môi trường.
''')
deps = next(c['source'] for c in old['cells'] if c['cell_type'] == 'code' and 'import importlib.metadata as metadata' in ''.join(c['source']))
code(''.join(deps))
md('''
    ## 3. Cấu hình

    Lần đầu để ACTION='smoke': 1 epoch ở đúng kích thước 1024 để kiểm tra cả dữ liệu và bộ nhớ.
    Smoke xong đổi ACTION='train', chạy lại ô 3 và 4. Train tối đa 30 epoch, early stopping 8.
    Train xong đổi ACTION='val', chạy lại ô 3 và 4.

    AUTO_RESUME=True: tiếp tục từ last.pt khi train ngắt; lượt hoàn tất sẽ được bỏ qua.
    Giữ NAME, EPOCHS, IMGSZ khi tiếp tục. Nếu lỗi CUDA out of memory, giảm BATCH xuống 1.
    Nếu chưa kịp có checkpoint và thư mục run đã tồn tại, dùng NAME mới để giữ nguyên log cũ.

    Cả 3 model được đo cùng imgsz1280, cùng conf/iou. Số AP này không so trực tiếp với báo cáo 640 cũ.
''')
code('''
    ACTION = 'smoke'          # smoke → train → val; test chỉ sau khi đã chọn model bằng val
    NAME = 'all3_balanced_v2'
    EPOCHS = 30
    BATCH = 2
    IMGSZ = 1024
    EVAL_IMGSZ = 1280
    AUTO_RESUME = True
    CHECKPOINT = 'trial_v2'   # test: baseline / trial_v1 / trial_v2; chọn sau khi xem val
    print(dict(ACTION=ACTION, NAME=NAME, EPOCHS=EPOCHS, BATCH=BATCH, IMGSZ=IMGSZ,
               EVAL_IMGSZ=EVAL_IMGSZ, AUTO_RESUME=AUTO_RESUME, CHECKPOINT=CHECKPOINT))
''')
md('''
    ## 4. Thực thi

    Chỉ in tiến độ định kỳ và kết quả từng epoch; log đầy đủ lưu Drive để tránh lag trình duyệt.
    Đừng chạy nhiều lượt song song. Khi lỗi, gửi 60 dòng cuối hiện phía dưới.
''')
code('''
    required = ['ROOT', 'PROJECT', 'ACTION', 'NAME', 'EPOCHS', 'BATCH', 'IMGSZ', 'EVAL_IMGSZ', 'AUTO_RESUME', 'CHECKPOINT']
    missing = [key for key in required if key not in globals()]
    if missing:
        raise RuntimeError('Chạy setup và ô 3 trước. Thiếu: ' + ', '.join(missing))
    import subprocess, sys, signal, time
    from collections import deque
    from datetime import datetime, timezone
    from IPython.display import display, Markdown
    command = [sys.executable, '-u', str(ROOT / 'run_all3.py'), ACTION,
               '--root', str(ROOT), '--project', str(PROJECT), '--name', NAME,
               '--epochs', str(EPOCHS), '--batch', str(BATCH), '--imgsz', str(IMGSZ),
               '--eval-imgsz', str(EVAL_IMGSZ), '--checkpoint', CHECKPOINT]
    if not AUTO_RESUME:
        command.append('--no-auto-resume')
    logs = PROJECT.parent / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path = logs / f'{NAME}_{ACTION}_{stamp}.log'
    effective_name = NAME + ('_smoke' if ACTION == 'smoke' else '')
    progress = PROJECT / 'progress' / (effective_name + '.json')
    started, next_status, seen_epoch = time.time(), 0, None
    print('Log:', log_path, flush=True)
    with log_path.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, text=True)
        try:
            while process.poll() is None:
                time.sleep(10)
                try:
                    status = json.loads(progress.read_text(encoding='utf-8'))
                    if status['time'] >= started and status['epoch'] != seen_epoch:
                        seen_epoch = status['epoch']
                        print('Epoch', status['epoch'], '/', status['epochs'], status['metrics'], flush=True)
                except (OSError, ValueError, KeyError):
                    pass
                elapsed = int(time.time() - started)
                if elapsed >= next_status:
                    print(f'{ACTION}: đã chạy {elapsed}s; chi tiết trong log trên Drive', flush=True)
                    next_status = elapsed + 30
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=20)
            print('Đã yêu cầu dừng. Lần sau giữ NAME, AUTO_RESUME=True để tiếp tục từ checkpoint đã lưu.')
            raise
        return_code = process.wait()
    with log_path.open(encoding='utf-8', errors='replace') as stream:
        print(''.join(deque(stream, maxlen=60)))
    if return_code:
        raise RuntimeError(f'Script dừng mã {return_code}. Đọc lỗi gốc phía trên. Log: {log_path}')
    if ACTION == 'smoke':
        print("Smoke xong: đổi ACTION='train', chạy lại ô 3 và 4.")
    elif ACTION == 'train':
        print("Train xong: đổi ACTION='val', chạy lại ô 3 và 4.")
    else:
        report_path = PROJECT / NAME / f'comparison_{ACTION}.json'
        display(Markdown(report_path.with_suffix('.md').read_text(encoding='utf-8')))
        print('Gửi file này để phân tích:', report_path)
        print('Các ảnh đối chiếu nằm trong:', PROJECT / NAME / 'evaluation' / ACTION)
''')
md('''
    ## Sau validation

    Gửi `MyDrive/fruit_all3_v2_20261116/runs/all3_balanced_v2/comparison_val.json`.
    So từng loại: AP50, Precision/Recall/F1 và MAE đếm; xem ảnh trong evaluation/val/*_previews.
    Chọn model dựa trên val trước khi ACTION='test'. Test đo model đã chọn + baseline và 71 ảnh nền cũ;
    đây là kiểm tra hồi quy, chưa phải bộ test thực tế độc lập. Không tinh chỉnh theo kết quả test.

    Candidate nằm ở weights/best.pt. Chưa thay exports/det_obb_s.pt.
    Bộ mới vẫn thừa kế nhãn cũ, cần thêm ảnh thực nhiều loại và ảnh nho đã kiểm tra để đánh giá triển khai.
    Báo cáo chỉ gồm detector, chưa gồm CLIP/teacher/classifier của app.
''')
for i, cell in enumerate(cells):
    cell['id'] = f'all3-cell-{i}'
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
path = HERE / 'finetune_all3_v2_colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print(path)
