"""Kiểm tra loại quả trên từng crop bằng CLIP (zero-shot, không train).
Trả xác suất 4 nhóm theo thứ tự NAMES: banana, apple, orange, other (other = cộng dồn mọi vật khác).
Dùng cùng detection.refine_types: bỏ box "other", đổi nhãn khi CLIP chắc chắn khác detector."""
import os
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "work/weights/hf"))   # model CLIP lưu trong dự án (/home gần đầy)
import open_clip  # noqa: E402

NAMES = {0: "banana", 1: "apple", 2: "orange", 3: "other"}
FRUIT_PROMPTS = {
    "banana": ["a banana", "a bunch of bananas", "a rotten banana", "a green banana"],
    "apple": ["an apple", "a red apple", "a green apple", "a yellow apple", "a rotten apple"],
    "orange": ["an orange", "a mandarin orange", "a tangerine", "a rotten orange", "a sliced orange"],
}
# Không cần liệt kê hết mọi vật: crop không giống quả nào sẽ rơi vào các mục chung ("an object", ...).
OTHER_PROMPTS = [
    "a strawberry", "grapes", "a mango", "a pear", "a plum", "a lemon", "a lime", "a peach", "a nectarine", "an apricot",
    "a kiwi fruit", "a pomegranate", "cherries", "a tomato", "a persimmon", "a bell pepper", "a papaya", "a pineapple",
    "a passion fruit", "a dragon fruit", "a lychee", "a fig", "berries", "a watermelon", "a melon", "a coconut", "an avocado",
    "a potato", "an onion", "a vegetable", "a ball", "a cup", "a bottle", "a human face", "a hand", "a table",
    "a leaf", "an empty background", "an object", "a toy", "food",
]
TEMPLATE = "a photo of {}."


class ClipChecker:
    # Mặc định ViT-L-14 (test: loại 85% crop vật khác, B-32 77%, cùng số quả mất 17/349; ~9 ms/crop, 1.6 GB VRAM).
    # Máy yếu: CLIP_ARCH=ViT-B-32 CLIP_PRETRAINED=laion2b_s34b_b79k (~2 ms/crop, 0.6 GB).
    def __init__(self, arch=os.getenv("CLIP_ARCH", "ViT-L-14"), pretrained=os.getenv("CLIP_PRETRAINED", "laion2b_s32b_b82k"), device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained, device=self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model.half()
        tok = open_clip.get_tokenizer(arch)
        prompts, groups = [], []
        for g, name in enumerate(("banana", "apple", "orange")):
            prompts += FRUIT_PROMPTS[name]; groups += [g] * len(FRUIT_PROMPTS[name])
        prompts += OTHER_PROMPTS; groups += [3] * len(OTHER_PROMPTS)
        self.groups = torch.tensor(groups, device=self.device)
        with torch.no_grad():
            t = self.model.encode_text(tok([TEMPLATE.format(p) for p in prompts]).to(self.device))
            self.text = t / t.norm(dim=-1, keepdim=True)
        self.names = NAMES

    def probs_boxes(self, img, boxes, pad=0.10, inner=-0.10, t_other=0.8, t_inner=float(os.getenv("CLIP_T_INNER", "0.9"))):
        """Xác suất cho từng box xyxy. Crop nới `pad`; box nào bị coi là "other" (>= t_other) thì nhìn lại phần giữa
        (thu `inner` mỗi cạnh); nếu phần giữa chắc là quả (>= t_inner) thì dùng kết quả đó: quả bị vật khác che/dính cạnh (cam cạnh dâu) không bị xoá,
        còn vật khác thật vẫn "other" ở cả hai lần nhìn. Không thu nhỏ cho mọi box vì chuối nằm chéo chỉ còn nền ở giữa."""
        def crop(b, p):
            h, w = img.shape[:2]; x1, y1, x2, y2 = b; px, py = (x2 - x1) * p, (y2 - y1) * p
            x1, y1 = int(max(0, min(w - 1, x1 - px))), int(max(0, min(h - 1, y1 - py)))
            return img[y1:int(min(h, max(y1 + 1, y2 + py))), x1:int(min(w, max(x1 + 1, x2 + px)))]
        P = self.probs([crop(b, pad) for b in boxes])
        redo = [k for k in range(len(P)) if P[k, 3] >= t_other]
        if redo:
            Q = self.probs([crop(boxes[k], inner) for k in redo])
            for k, q in zip(redo, Q):
                if q[:3].max() >= t_inner:   # chỉ tin lần nhìn giữa khi nó chắc là quả
                    P[k] = q
        return P

    @torch.no_grad()
    def probs(self, crops_bgr, batch=32):
        """crops_bgr: list ảnh BGR (numpy). Trả mảng Nx4 xác suất theo NAMES."""
        from PIL import Image
        out = []
        for i in range(0, len(crops_bgr), batch):
            x = torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))) for c in crops_bgr[i:i + batch]])
            x = x.to(self.device, dtype=self.text.dtype)
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).float().softmax(dim=-1)            # N x số prompt
            g = torch.zeros(p.shape[0], 4, device=p.device).index_add_(1, self.groups, p)
            out.append(g.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 4), np.float32)
