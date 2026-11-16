"""Build a portable experimental OBB input; never modify app or source labels."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import zipfile

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/"work/obb/data"
REVIEW=ROOT/"out/obb_review_20261116"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(stem):
    return re.sub(r"(_(?:jpg|jpeg|png|bmp|webp))?\.rf\.[a-z0-9]+$","",stem,flags=re.I)


def validate(text):
    counts=Counter()
    for line in text.splitlines():
        if not line.strip():
            continue
        v=list(map(float,line.split()))
        if len(v)!=9 or v[0] not in (0,1,2) or not all(math.isfinite(x) and 0<=x<=1 for x in v[1:]):
            raise ValueError("Invalid OBB row")
        counts[int(v[0])]+=1
    return counts


def main():
    destination=HERE/"fruit_accuracy_trial_20261116.zip"
    if destination.exists():
        raise ValueError("Input ZIP already exists; preserve the version")
    audit=json.loads((REVIEW/"review_audit/report.json").read_text(encoding="utf-8"))
    drafts=json.loads((REVIEW/"draft_provenance.json").read_text(encoding="utf-8"))
    originals={r["id"]:r for r in drafts["items"]}
    geometry=Path(audit["output"])
    excluded={r["id"] for r in audit["user_confirmed_exclusions"]}
    quarantined={r["id"] for r in audit["pending_semantic_corrections"]}
    selected=[r for r in audit["items"] if r["id"] not in excluded|quarantined]
    excluded_names={Path(originals[i]["source_image"]).name for i in excluded|quarantined}
    replacement_names={Path(r["source"]).name for r in selected}
    train_lines=[]
    val_lines=[]
    reviewed_val=[]
    source_groups={"train":set(),"val":set()}
    hashes={"train":set(),"val":set()}
    records=[]
    invalid=[]
    skipped=[]
    entries=[]
    for split in ("train","val"):
        paths=[(BASE/p.strip()).resolve() for p in (BASE/(split+".txt")).read_text().splitlines() if p.strip()]
        unique={}
        for path in paths:
            if path.name in excluded_names or path.name in replacement_names:
                skipped.append({"split":split,"source":path.relative_to(ROOT).as_posix(),"reason":"quarantined/excluded or replaced by submitted review"})
                continue
            if path not in unique:
                label=path.parent.parent/"labels"/(path.stem+".txt")
                if not path.is_file() or not label.is_file():
                    raise FileNotFoundError(f"Missing base resource: {path}")
                text=label.read_text(encoding="utf-8-sig")
                try:
                    counts=validate(text)
                except ValueError as exc:
                    invalid.append({"split":split,"source":path.relative_to(ROOT).as_posix(),"reason":str(exc)})
                    unique[path]=None
                    continue
                alias="base_"+hashlib.sha256(path.relative_to(BASE).as_posix().encode()).hexdigest()[:24]
                relative=f"images/{split}/{alias}{path.suffix.lower()}"
                image_hash=digest(path)
                source_groups[split].add(group(path.stem))
                hashes[split].add(image_hash)
                record={"split":split,"image":relative,"label":f"labels/{split}/{alias}.txt",
                        "source":path.relative_to(ROOT).as_posix(),"image_sha256":image_hash,
                        "label_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "original_label_sha256":digest(label),"origin":"legacy_obb","class_counts":dict(counts)}
                records.append(record)
                entries.append((path,record["image"],text,record["label"]))
                unique[path]=relative
            if unique[path] is not None:
                (train_lines if split=="train" else val_lines).append("./"+unique[path])
    for item in selected:
        split=item["role"]
        original=originals[item["id"]]
        path=geometry/item["image"]
        label=geometry/item["label"]
        text=label.read_text(encoding="utf-8")
        counts=validate(text)
        relative=f"images/{split}/reviewed_{item['id']}.png"
        target_label=f"labels/{split}/reviewed_{item['id']}.txt"
        source_groups[split].add(original["source_group"])
        hashes[split].add(original["display_image_sha256"])
        record={"split":split,"image":relative,"label":target_label,"source":item["source"],
                "image_sha256":digest(path),"label_sha256":hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "original_label_sha256":digest(label),
                "origin":"submitted_review_geometry_padded","class_counts":dict(counts),
                "source_group":original["source_group"],"padding_ltrb":item["padding_ltrb"]}
        records.append(record)
        entries.append((path,relative,text,target_label))
        if split=="train":
            train_lines.extend(["./"+relative]*4)
        else:
            val_lines.append("./"+relative)
            reviewed_val.append("./"+relative)
    if source_groups["train"] & source_groups["val"] or hashes["train"] & hashes["val"]:
        raise ValueError("Source group or exact image leakage across derived train/validation")
    classes={split:Counter() for split in ("train","val")}
    for record in records:
        classes[record["split"]].update({int(k):v for k,v in record["class_counts"].items()})
    if any(set(classes[split])!={0,1,2} for split in classes):
        raise ValueError("Derived train and legacy validation must retain all three classes")
    manifest={"schema":1,"date":"2026-11-16","experimental":True,
              "baseline_model_sha256":digest(ROOT/"exports/det_obb_s.pt"),
              "submitted_zip_sha256":digest(REVIEW/"obb_reviewed_v1.zip"),
              "user_excluded_ids":sorted(excluded),"quarantined_review_ids":sorted(quarantined),
              "added_review_images":dict(Counter(r["role"] for r in selected)),
              "review_train_repetitions":4,"train_entries":len(train_lines),"train_unique":len(set(train_lines)),
              "val_images":len(val_lines),"reviewed_banana_val_images":len(reviewed_val),
              "class_boxes_unique":classes,"invalid_legacy_resources":invalid,"skipped_base_resources":skipped,
              "limitations":["Legacy validation still includes historical bunch labels; aggregate AP/count metrics are relative checks.",
                              "Reviewed supplemental validation contains bananas only; it is not an independent three-class test.",
                              "No diagnostic or test images used for training. No original val/test files changed.",
                              "Known semantic annotation errors are quarantined, not silently accepted or auto-corrected.",
                              "New review data is small; training does not guarantee improvement and must not replace the app automatically."],
              "files":records}
    names="names:\n  0: banana\n  1: apple\n  2: orange\n"
    package_files={
        "data.yaml":"train: train.txt\nval: val.txt\n"+names,
        "reviewed_banana_val.yaml":"train: train.txt\nval: reviewed_banana_val.txt\n"+names,
        "train.txt":"\n".join(train_lines)+"\n",
        "val.txt":"\n".join(val_lines)+"\n",
        "reviewed_banana_val.txt":"\n".join(reviewed_val)+"\n",
        "input_manifest.json":json.dumps(manifest,ensure_ascii=False,indent=2),
    }
    with zipfile.ZipFile(destination,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=1) as zipped:
        for i,(path,relative,text,label) in enumerate(entries,1):
            zipped.write(path,relative)
            zipped.writestr(label,text)
            if i%1000==0:
                print(f"Packed {i}/{len(entries)} unique images",flush=True)
        for name,text in package_files.items():
            zipped.writestr(name,text)
        zipped.write(ROOT/"exports/det_obb_s.pt","baseline.pt")
        zipped.write(HERE/"run_trial.py","run_trial.py")
    (HERE/"input_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    summary={k:manifest[k] for k in ("added_review_images","train_entries","train_unique","val_images","reviewed_banana_val_images","class_boxes_unique")}
    summary.update(zip=str(destination),bytes=destination.stat().st_size,sha256=digest(destination))
    (HERE/"package_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    main()
