import json
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
old = json.loads((HERE / "finetune_all3_v2_colab.ipynb").read_text(encoding="utf-8"))
cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"})


def code(source):
    cells.append({"cell_type": "code", "metadata": {}, "source": dedent(source).strip() + "\n", "execution_count": None, "outputs": []})


md('''
    # Kiểm tra kích thước cho cả 3 loại — không train lại

    Dùng nguyên ZIP `fruit_all3_v2_20261116.zip` đã có trên My Drive và kết quả train hiện tại.
    Không cần upload ZIP/model lần nữa. Chọn T4 GPU; đảm bảo lượt train trước đã dừng.

    Notebook chỉ đo validation297 ảnh ở 640 và 1024, so baseline/trial_v1/trial_v2.
    Kết quả1280 dùng lại từ báo cáo đã gửi. Không chạy test, không thay model app.
''')
setup = next(c for c in old["cells"] if c["cell_type"] == "code" and "ZIP_DRIVE =" in "".join(c["source"]))
gpu = next(c for c in old["cells"] if c["cell_type"] == "code" and "torch.cuda.is_available()" in "".join(c["source"]))
deps = next(c for c in old["cells"] if c["cell_type"] == "code" and "import importlib.metadata as metadata" in "".join(c["source"]))
code("".join(gpu["source"]))
md("## 1. Kết nối Drive và kiểm tra dữ liệu cũ")
code("".join(setup["source"]))
md("## 2. Thư viện (giữ nguyên Torch/CUDA)")
code("".join(deps["source"]))
md('''
    ## 3. Cấu hình

    Giữ đúng NAME của lượt train v2. BATCH=2 như báo cáo vừa gửi.
    Không có ACTION=train trong notebook này.
''')
code('''
    NAME = 'all3_balanced_v2'
    BATCH = 2
    assert (PROJECT / NAME / 'weights/best.pt').is_file(), 'Không tìm thấy best.pt; kiểm tra NAME và tài khoản Drive.'
    assert (PROJECT / NAME / 'comparison_val.json').is_file(), 'Cần giữ comparison_val.json của lượt vừa chạy.'
    print('Model đã train:', PROJECT / NAME / 'weights/best.pt')
''')
md('''
    ## 4. Đo 640/1024

    Chạy một lượt; log lưu Drive. Nếu Colab ngắt, chạy lại setup và ô này:
    các model/kích thước đã đo xong sẽ được dùng lại, không đo lại toàn bộ.
''')
runner = (HERE / 'run_resolution_check.py').read_text(encoding='utf-8')
code("RUNNER_SOURCE = " + repr(runner) + "\n(ROOT / 'run_resolution_check.py').write_text(RUNNER_SOURCE, encoding='utf-8')\n")
code('''
    import subprocess, sys, signal, time
    from collections import deque
    from datetime import datetime, timezone
    from IPython.display import display, Markdown
    output_dir = PROJECT / NAME / f'resolution_check_d0abaa79_batch{BATCH}'
    progress = output_dir / 'progress.json'
    logs = PROJECT.parent / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path = logs / f'{NAME}_resolution_check_{stamp}.log'
    command = [sys.executable, '-u', str(ROOT / 'run_resolution_check.py'),
               '--root', str(ROOT), '--project', str(PROJECT), '--name', NAME, '--batch', str(BATCH)]
    print('Log:', log_path, flush=True)
    started = time.time()
    with log_path.open('w', encoding='utf-8') as output:
        process = subprocess.Popen(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, text=True)
        try:
            while process.poll() is None:
                time.sleep(20)
                stage = 'Đang khởi tạo'
                try:
                    stage = json.loads(progress.read_text(encoding='utf-8'))['stage']
                except (OSError, ValueError, KeyError):
                    pass
                print(int(time.time() - started), 's |', stage, flush=True)
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=20)
            raise
        return_code = process.wait()
    with log_path.open(encoding='utf-8', errors='replace') as stream:
        print(''.join(deque(stream, maxlen=60)))
    if return_code:
        raise RuntimeError(f'Lỗi mã {return_code}; đọc lỗi gốc ngay phía trên. Log: {log_path}')
    report = output_dir / 'resolution_comparison.json'
    display(Markdown(report.with_suffix('.md').read_text(encoding='utf-8')))
    print('Gửi file này:', report)
''')
md('''
    Gửi `resolution_comparison.json` để chọn model + kích thước dựa trên AP từng loại, F1 và sai số đếm.
    Kết quả vẫn dùng validation cũ có nhãn nải; cần kiểm tra ảnh thực và không xem đây là test độc lập.
''')
for i, cell in enumerate(cells):
    cell['id'] = f'resolution-cell-{i}'
notebook = {'cells': cells, 'metadata': old['metadata'], 'nbformat': 4, 'nbformat_minor': 5}
path = HERE / 'kiem_tra_kich_thuoc_all3_colab.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding='utf-8')
print(path)
