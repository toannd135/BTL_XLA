"""Create a focused review bundle for crowded, flagged fruit images."""
import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("out/instance_review_v2"))
    parser.add_argument("--out", type=Path, default=Path("out/instance_crowded_review_v1"))
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.out.resolve()
    if output.exists():
        raise ValueError(f"Output exists: {output}")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    items = manifest["items"]
    crowded = [
        item for item in items
        if item["role"] == "diagnostic" or len(item["boxes"]) >= 2 or item["warnings"]
    ]
    crowded.sort(key=lambda item: (
        0 if item["role"] == "diagnostic" else 1 if item["warnings"] else 2,
        -len(item["boxes"]),
        item["source_image"],
    ))
    selected = crowded[:args.limit]
    if not selected:
        raise ValueError("No crowded images found")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    for item in selected:
        source_image = source / item["image"]
        target_image = output / item["image"]
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_image, target_image)
    focused = dict(manifest)
    focused["items"] = selected
    focused["counts"] = {"selected": len(selected)}
    focused["policy"] = (
        "Focused manual review: one annotation per visible fruit; "
        "verify crowded banana instances before training."
    )
    (output / "manifest.json").write_text(
        json.dumps(focused, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "manifest.js").write_text(
        "window.INSTANCE_DATA = " + json.dumps(focused, ensure_ascii=False) + ";\n",
        encoding="utf-8",
    )
    shutil.copy2(source / "review.html", output / "review.html")
    print(json.dumps({"review": str(output / "review.html"), "selected": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
