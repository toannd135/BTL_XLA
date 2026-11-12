"""Chỉnh danh sách prompt CLIP: tính đặc trưng ảnh một lần, thử nhiều bộ prompt (chỉ mã hoá chữ).
- Quả: crop box nhãn gốc local_dataset/det/<split> (nới 10%).
- Vật khác: crop do detector (det_obb_s, conf 0.3) báo trên own_other/raw; tách nửa theo hash tên file (tune = chẵn, report = lẻ).
In: tỉ lệ quả bị bỏ (p_other >= t_other), quả bị đổi sai loại (p >= t_swap), vật khác bị loại đúng.
Dùng: python work/fruitcls/tune_prompts.py [split=val] [half=tune|report]"""
import hashlib, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend")); sys.path.insert(0, str(ROOT / "work/fruitcls"))
import clip_check  # noqa: E402
from detection import detect_obb  # noqa: E402
from prompt_sets import SETS  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else "val"
half = sys.argv[2] if len(sys.argv) > 2 else "tune"
import os
ARCH, PRE = os.getenv("CLIP_ARCH", "ViT-B-32"), os.getenv("CLIP_PRE", "laion2b_s34b_b79k")
ck = clip_check.ClipChecker(ARCH, PRE)
tok = clip_check.open_clip.get_tokenizer(ARCH)


def feats(crops):
    out = []
    with torch.no_grad():
        for i in range(0, len(crops), 64):
            x = torch.stack([ck.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops[i:i + 64]])
            f = ck.model.encode_image(x.to(ck.device, dtype=ck.text.dtype)); out.append(f / f.norm(dim=-1, keepdim=True))
    return torch.cat(out)


PAD = float(os.getenv('CLIP_PAD', '0.10'))


def pad_crop(img, b, pad=None):
    pad = PAD if pad is None else pad
    h, w = img.shape[:2]; x1, y1, x2, y2 = b; px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(w, x2 + px)), int(min(h, y2 + py))
    return img[y1:y2, x1:x2] if x2 - x1 > 8 and y2 - y1 > 8 else None


fruit, labs = [], []
d = ROOT / "local_dataset/det" / split
for lp in sorted((d / "labels").glob("*.txt")):
    img = cv2.imread(str(next((d / "images").glob(lp.stem + ".*")))); h, w = img.shape[:2]
    for ln in lp.read_text().split("\n"):
        p = ln.split()
        if len(p) == 5:
            c, x, y, bw, bh = int(p[0]), *map(float, p[1:])
            cr = pad_crop(img, [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
            if cr is not None:
                fruit.append(cr); labs.append(c)
det = YOLO(str(ROOT / "exports/det_obb_s.pt"))
other = []
for p in sorted((ROOT / "own_other/raw").glob("*.jpg")):
    if (int(hashlib.md5(p.name.encode()).hexdigest(), 16) % 2 == 0) != (half == "tune"):
        continue
    img = cv2.imread(str(p))
    if img is None:
        continue
    for b in detect_obb(det, img, conf=0.3, quantize=16)[1]:
        cr = pad_crop(img, b)
        if cr is not None:
            other.append(cr)
Ff, Fo, labs = feats(fruit), feats(other), np.array(labs)
print(f"split={split} half={half}: {len(labs)} crop quả, {len(other)} crop vật khác (detector báo)")


def group_probs(F, fruit_p, other_p, template):
    prompts, groups = [], []
    for g, n in enumerate(("banana", "apple", "orange")):
        prompts += fruit_p[n]; groups += [g] * len(fruit_p[n])
    prompts += other_p; groups += [3] * len(other_p)
    with torch.no_grad():
        t = ck.model.encode_text(tok([template.format(x) for x in prompts]).to(ck.device)); t = t / t.norm(dim=-1, keepdim=True)
    p = (100.0 * F @ t.T).float().softmax(-1)
    return torch.zeros(p.shape[0], 4, device=p.device).index_add_(1, torch.tensor(groups, device=p.device), p).cpu().numpy()


for name, (fp, op, tpl) in SETS.items():
    if name not in os.getenv('SETS', 'v0').split(','):
        continue
    Pf, Po = group_probs(Ff, fp, op, tpl), group_probs(Fo, fp, op, tpl)
    for t_o, t_s in ((0.8, 0.9), (0.8, 0.8), (0.9, 0.8)):
        drop = Pf[:, 3] >= t_o
        top = Pf[:, :3].argmax(1)
        swap = (~drop) & (top != labs) & (Pf[np.arange(len(labs)), top] >= t_s)
        rej = Po[:, 3] >= t_o
        lost = {n: int((drop | swap)[labs == k].sum()) for k, n in enumerate(("banana", "apple", "orange"))}
        print(f"{name:10s} t={t_o},{t_s}  quả mất/sai {int((drop | swap).sum())}/{len(labs)} {lost}  vật khác bị loại {rej.mean():.1%}")
