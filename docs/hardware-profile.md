# Hardware profile

Audit date: 2026-11-18. No GPU configuration was selected based on assumed VRAM.

## Observed environment

| Item | Observed value |
|---|---|
| Python | 3.11.8 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA build | 12.1 |
| `torch.cuda.is_available()` | `false` |
| `nvidia-smi` | Fails: cannot communicate with NVIDIA driver |
| GPU name / compute capability | Not observable |
| Dedicated VRAM total/free | Not observable; do not infer it from the CUDA build |
| OpenCV | 5.0.0 |
| Ultralytics | 8.4.180 |

This execution environment cannot validate the user's separate 2–3 GB target
machine. No measured GPU memory figures are available and no CUDA compatibility
claim is made.

## Checkpoint sizes (disk only; not VRAM use)

- `yolo11n-seg.pt`: approximately 6.2 MB.
- `yolo11s-seg.pt`: approximately 20.7 MB.
- custom segmentation candidate `best.pt`: approximately 20 MB.
- `det_obb_s.pt`: approximately 19.8 MB.
- `cls_finetune_best.pt`: approximately 3.2 MB.

Checkpoint file size does not predict peak training or inference VRAM. Activations,
input resolution, batch, optimizer state, workspace and other resident models
must be included in actual measurements.

## Target-machine measurement checklist

On the target computer, capture the following before configuration selection:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,memory.free,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA device')"
```

Record idle VRAM and free VRAM immediately before each experiment. Measure peak
allocated and reserved memory from PyTorch and sample process usage with
`nvidia-smi`. Save the exact command, checkpoint hash and software versions with
the result. Do not extrapolate 320 px success to 416/512 px without a separate
test.
