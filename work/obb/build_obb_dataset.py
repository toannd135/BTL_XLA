"""Đổi dữ liệu thật sang YOLO-OBB (bản sao trong work/obb/data/real, không sửa nhãn gốc).
- Chuối: khung xoay = minAreaRect mặt nạ SAM2 nếu mặt nạ có dáng một quả (aspect >= 2.0); ngược lại là box nải/lỗi.
  Train: bỏ cả ảnh có box chuối dạng nải (để không dạy model box nải; ảnh ghép thay phần này). Val/test: giữ box thẳng.
- Táo/cam: giữ box thẳng gốc (4 góc, góc 0).
- Thành phần train như C: ảnh công khai (đã bỏ quarantine) + own x4 + ảnh nền own_neg x2; cộng ảnh ghép work/obb/data/synth.
Dùng: python work/obb/build_obb_dataset.py"""
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBB = ROOT / "work" / "obb"
DATA = OBB / "data"


def corners(x1, y1, x2, y2):
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def main():
    inst = defaultdict(list)
    for line in open(OBB / "instances.jsonl"):
        r = json.loads(line)
        inst[(r["split"], r["image"])].append(r)
    c_list = (ROOT / "work/exp/det_C_neg_train.txt").read_text().split()
    stats = Counter()
    out_lists = defaultdict(list)
    done = {}
    jobs = [("train", p) for p in c_list] + [(sp, str(p)) for sp in ("val", "test")
                                            for p in sorted((ROOT / "local_dataset/det" / sp / "images").iterdir())]
    for sp, p in jobs:
        key = (sp, p)
        if key in done:
            if done[key]:
                out_lists[sp].append(done[key])
            continue
        src = Path(p)
        rows, bunchy = [], False
        for r in sorted(inst.get(key, []), key=lambda t: t["i"]):
            w, h = r["w"], r["h"]
            x1, y1, x2, y2 = r["box"]
            if r["cls"] == 0 and "obb" in r and r["aspect"] >= 2.0 and r["solidity"] >= 0.5:
                pts = [v for xy in r["obb"] for v in (min(max(xy[0] / w, 0), 1), min(max(xy[1] / h, 0), 1))]
                stats[f"{sp}_banana_obb"] += 1
            else:
                if r["cls"] == 0:
                    bunchy = True
                    stats[f"{sp}_banana_axis"] += 1
                pts = [min(max(v, 0), 1) for v in corners(x1 / w, y1 / h, x2 / w, y2 / h)]
            rows.append(f"{r['cls']} " + " ".join(f"{v:.5f}" for v in pts))
        if sp == "train" and bunchy:
            stats["train_dropped_bunchy"] += 1
            done[key] = None
            continue
        sub = DATA / "real" / sp
        (sub / "images").mkdir(parents=True, exist_ok=True)
        (sub / "labels").mkdir(parents=True, exist_ok=True)
        name = src.name if "/neg/" not in p else f"neg_{src.name}"
        dst = sub / "images" / name
        if not dst.exists():
            shutil.copy2(src, dst)
        (sub / "labels" / f"{dst.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        done[key] = str(dst)
        out_lists[sp].append(str(dst))
        stats[f"{sp}_images"] += 1
    syn = sorted(str(p) for p in (DATA / "synth" / "images").glob("*.jpg"))

    def rel(ps):   # "./..." tương đối theo DATA: Ultralytics tự nối với thư mục chứa tệp .txt, chạy được trên Colab
        return "\n".join("./" + Path(p).relative_to(DATA).as_posix() for p in ps) + "\n"

    (DATA / "train.txt").write_text(rel(out_lists["train"] + syn))
    (DATA / "val.txt").write_text(rel(out_lists["val"]))
    (DATA / "test.txt").write_text(rel(out_lists["test"]))
    names = "names:\n  0: banana\n  1: apple\n  2: orange\n"
    # không ghi "path": Ultralytics lấy thư mục chứa obb.yaml
    (DATA / "obb.yaml").write_text(f"train: train.txt\nval: val.txt\ntest: test.txt\nnc: 3\n{names}")
    stats["train_entries_real"] = len(out_lists["train"])
    stats["synth"] = len(syn)
    print(json.dumps(stats, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
