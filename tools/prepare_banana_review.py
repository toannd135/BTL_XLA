"""Prepare a focused manual review bundle for individual banana annotations."""

import argparse
import json
import shutil
from pathlib import Path


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "out/instance_review_v1")
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--out", type=Path, default=repo / "out/banana_instance_review_v1")
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.out.resolve()
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    candidates = []
    for item in manifest["items"]:
        banana_boxes = [box for box in item.get("boxes", []) if int(box[0]) == 0]
        text = f"{item['source_image']} {' '.join(item.get('warnings', []))}".lower()
        crowded = len(banana_boxes) >= 2 or any(
            word in text for word in ("bunch", "nested", "banana", "chuoi")
        )
        if banana_boxes and crowded:
            candidates.append(item)

    candidates.sort(key=lambda item: (
        0 if any(word in " ".join(item.get("warnings", [])).lower()
                 for word in ("bunch", "nested")) else 1,
        0 if len(item.get("boxes", [])) >= 2 else 1,
        item["source_image"],
    ))
    selected = candidates[:max(1, args.limit)]

    diagnostic_items = []
    if args.diagnostics:
        for image in sorted(args.diagnostics.resolve().iterdir()):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            diagnostic_items.append({
                "id": f"diagnostic-{image.stem}",
                "role": "diagnostic",
                "source_image": f"diagnostic/{image.name}",
                "image": f"images/diagnostic_{image.name}",
                "boxes": [],
                "warnings": ["diagnostic-only", "label-every-visible-banana"],
            })

    output.mkdir(parents=True)
    (output / "images").mkdir()
    for item in selected:
        source_image = source / item["image"]
        target = output / item["image"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target)
    for item in diagnostic_items:
        shutil.copy2(args.diagnostics / Path(item["source_image"]).name,
                     output / item["image"])

    focused = dict(manifest)
    focused["items"] = selected + diagnostic_items
    focused["counts"] = {"selected_banana": len(selected), "diagnostic": len(diagnostic_items)}
    focused["policy"] = (
        "Manual banana review: annotate one box per visible banana, never one box "
        "around a bunch; approve only after checking every visible fruit."
    )
    (output / "manifest.json").write_text(
        json.dumps(focused, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "manifest.js").write_text(
        "window.INSTANCE_DATA = " + json.dumps(focused, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    shutil.copy2(repo / "tools/instance_review.html", output / "review.html")
    print(json.dumps({
        "review": str(output / "review.html"),
        "selected_banana": len(selected),
        "diagnostic": len(diagnostic_items),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
