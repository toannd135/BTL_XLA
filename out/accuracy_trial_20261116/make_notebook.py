import json
from pathlib import Path
import textwrap

HERE=Path(__file__).resolve().parent
summary=json.loads((HERE/"package_summary.json").read_text(encoding="utf-8"))
cells=[]


def markdown(text):
    cells.append({"cell_type":"markdown","metadata":{},"source":textwrap.dedent(text).strip()+"\n"})


def code(text):
    cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],
                  "source":textwrap.dedent(text).strip()+"\n"})


markdown("""
    # Thử cải thiện detector OBB — 16/11/2026

    Chọn **Runtime → Change runtime type → T4 GPU**. Notebook chỉ train trên GPU.
    Upload `fruit_accuracy_trial_20261116.zip` vào **My Drive của bạn, ngay thư mục gốc**.
    Gói tự chứa ảnh, nhãn, baseline và runner; không cần clone repo, lấy dữ liệu từ Drive bạn khác hay thêm token GitHub.

    Dữ liệu: 14 ảnh train bổ sung (lặp4lần có chủ đích) + bộ OBB cũ;4ảnh val chuối đã duyệt kết hợp val cũ.
    Hai ảnh chưa có nhãn đã loại theo yêu cầu. Ba ảnh còn lỗi nội dung được cách ly, không tự sửa/gán nhãn đúng.
    Val cũ còn nhãn nải nên số liệu chỉ để so tương đối;4ảnh chuối là kiểm tra phát triển, không phải test cuối.
    Đây là thí nghiệm; không tự thay model app và không cam kết accuracy tăng.
    """
)
code("""
    import torch
    assert torch.cuda.is_available(), 'Chọn Runtime → Change runtime type → T4 GPU. Không train lượt này trên CPU.'
    print('GPU:', torch.cuda.get_device_name(0))
    print('Torch:', torch.__version__)
    """
)
markdown("""
    ## 1. Mount Drive và giải nén dữ liệu

    Chọn cho phép truy cập Drive của chính bạn. Dữ liệu được copy vào `/content` để đọc ảnh nhanh;
    checkpoint và log sẽ lưu trên Drive. Chạy lại ô này khi runtime mới hoặc bị mất kết nối.
    """
)
code("""
    from google.colab import drive
    drive.mount('/content/drive')
    import hashlib, json, shutil, zipfile
    from pathlib import Path

    ZIP_DRIVE = Path('/content/drive/MyDrive/fruit_accuracy_trial_20261116.zip')
    ZIP_LOCAL = Path('/content/fruit_accuracy_trial_20261116.zip')
    ROOT = Path('/content/fruit_accuracy_trial_20261116')
    PROJECT = Path('/content/drive/MyDrive/fruit_accuracy_trial_20261116/runs')
    EXPECTED_SHA256 = '__ZIP_SHA256__'

    def sha256_file(path):
        value = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4*1024*1024), b''):
                value.update(chunk)
        return value.hexdigest()

    assert ZIP_DRIVE.is_file(), 'Upload ZIP vào My Drive, tên đúng fruit_accuracy_trial_20261116.zip'
    if not ZIP_LOCAL.is_file():
        print('Copy ZIP từ Drive sang runtime...')
        shutil.copyfile(ZIP_DRIVE, ZIP_LOCAL)
    assert sha256_file(ZIP_LOCAL) == EXPECTED_SHA256, 'ZIP chưa tải đủ hoặc sai phiên bản; kiểm tra file upload'
    ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_LOCAL) as archive:
        assert all((ROOT / entry.filename).resolve().is_relative_to(ROOT.resolve()) for entry in archive.infolist())
        archive.extractall(ROOT)
    manifest = json.loads((ROOT / 'input_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        assert sha256_file(ROOT / record['image']) == record['image_sha256'], record['image']
        assert sha256_file(ROOT / record['label']) == record['label_sha256'], record['label']
    assert sha256_file(ROOT / 'baseline.pt') == manifest['baseline_model_sha256']
    print('Train entries:', manifest['train_entries'], '| unique:', manifest['train_unique'])
    print('Val:', manifest['val_images'], '| reviewed banana val:', manifest['reviewed_banana_val_images'])
    print('Đã kiểm tra hash ảnh, nhãn và baseline. Kết quả sẽ lưu ở:', PROJECT)
    """.replace("__ZIP_SHA256__",summary["sha256"])
)
markdown("""
    ## 2. Thư viện

    Giữ torch/torchvision có sẵn của Colab. Các lệnh pip đều dùng `--no-deps` để tránh thay Torch/CUDA.
    """
)
code("""
    import importlib.metadata as metadata
    import subprocess, sys
    from packaging.requirements import Requirement

    torch_before = {name: metadata.version(name) for name in ('torch', 'torchvision')}
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
                    'ultralytics==8.4.180', 'ultralytics-thop', 'polars', 'nvidia-ml-py'], check=True)
    # Polars separates its native runtime into an explicit matching-version wheel.
    runtime_requirements = []
    for requirement in metadata.requires('polars') or []:
        parsed = Requirement(requirement)
        if parsed.name.startswith('polars-runtime-') and (parsed.marker is None or parsed.marker.evaluate({'extra': ''})):
            runtime_requirements.append(parsed.name + str(parsed.specifier))
    if runtime_requirements:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps', *runtime_requirements], check=True)
    assert torch_before == {name: metadata.version(name) for name in torch_before}, 'Torch/torchvision changed unexpectedly'
    import ultralytics, polars
    print('Ultralytics:', ultralytics.__version__, '| Polars:', polars.__version__)
    """
)
markdown("""
    ## 3. Cấu hình lượt chạy

    Lần đầu để `ACTION='smoke'`: chạy1epoch, lưu riêng dưới tên có hậu tố `_smoke`.
    Smoke xong đổi `ACTION='train'`, `RESUME=False` rồi chạy lại ô cấu hình và ô thực thi.
    Train tối đa20epoch, có early stopping. Nếu OOM, hạ BATCH từ8xuống4.

    Nếu lượt train bị ngắt: chạy lại setup, đặt cùng NAME, `ACTION='train'`, `RESUME=True`.
    Chỉ resume checkpoint chưa kết thúc và còn optimizer. Run đã train xong không resume nữa.
    Train xong đặt `ACTION='val'`, `RESUME=False` để so baseline/candidate trên cùng validation.
    """
)
code("""
    ACTION = 'smoke'          # smoke → train → val
    NAME = 'obb_patch_v1'     # Thí nghiệm khác dùng tên mới; giữ nguyên tên khi resume
    EPOCHS = 20
    BATCH = 8
    RESUME = False
    print(dict(ACTION=ACTION, NAME=NAME, EPOCHS=EPOCHS, BATCH=BATCH, RESUME=RESUME))
    """
)
markdown("""
    ## 4. Thực thi và theo dõi

    Log chi tiết lưu trên Drive; notebook chỉ in trạng thái định kỳ/epoch để tránh lag vì hàng nghìn dòng progress bar.
    Nếu lỗi,60dòng cuối của log hiện ra. Không bấm chạy nhiều lượt song song.
    """
)
code("""
    required = ['ROOT', 'PROJECT', 'ACTION', 'NAME', 'EPOCHS', 'BATCH', 'RESUME']
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError('Chạy các ô setup và cấu hình trước. Thiếu: ' + ', '.join(missing))
    import signal, time
    from collections import deque
    from datetime import datetime, timezone
    import subprocess, sys
    assert ACTION in {'smoke', 'train', 'val'}
    assert not RESUME or ACTION == 'train', 'Đặt RESUME=False cho smoke/val'
    assert isinstance(NAME, str) and NAME and '/' not in NAME and '\\\\' not in NAME
    command = [sys.executable, str(ROOT / 'run_trial.py'), ACTION,
               '--root', str(ROOT), '--project', str(PROJECT), '--name', NAME,
               '--epochs', str(EPOCHS), '--batch', str(BATCH)]
    if RESUME:
        command.append('--resume')
    log_dir = PROJECT.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'{NAME}_{ACTION}_{stamp}.log'
    effective_name = NAME + ('_smoke' if ACTION == 'smoke' else '')
    progress = PROJECT / 'progress' / (effective_name + '.json')
    started = time.time()
    seen_epoch = None
    next_status = 0
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
            print('Đã yêu cầu dừng. Resume chỉ từ checkpoint đã ghi xong trên Drive.')
            raise
        return_code = process.wait()
    tail = ''.join(deque(log_path.open(encoding='utf-8', errors='replace'), maxlen=60))
    print(tail)
    if return_code:
        raise RuntimeError(f'Script dừng mã {return_code}. Lỗi gốc ở60dòng cuối phía trên; log: {log_path}')
    if ACTION == 'smoke':
        print('Smoke xong. Đổi ACTION=train, RESUME=False; chạy lại ô3và4.')
    elif ACTION == 'train':
        print('Train xong. Đổi ACTION=val, RESUME=False; chạy lại ô3và4.')
    else:
        report_path = PROJECT / NAME / 'comparison_val.json'
        report = json.loads(report_path.read_text(encoding='utf-8'))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print('Gửi comparison_val.json để chốt bước tiếp. Chưa thay model app tự động.')
    """
)
markdown("""
    ## Sau cùng

    File so sánh: `MyDrive/fruit_accuracy_trial_20261116/runs/obb_patch_v1/comparison_val.json`.
    Candidate: cùng thư mục run, `weights/best.pt`. Giữ baseline gốc.
    Chỉ cân nhắc chạy app với candidate sau khi xem AP từng loại, đếm chuối và ảnh thực tế/vật khác.
    Số liệu ở đây không phải kết quả pipeline có CLIP/teacher và không phải test cuối độc lập.
    """
)
for i,cell in enumerate(cells):
    cell["id"]=f"cell-{i}"
notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                                  "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
destination=HERE/"finetune_accuracy_trial_colab.ipynb"
destination.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding="utf-8")
print(destination)
