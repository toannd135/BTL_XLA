"""Verify submitted trial reports and any locally available trial weights."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-models", action="store_true", help="Fail if any trial weight is missing")
    args = parser.parse_args()
    registry = json.loads((ROOT / "exports/trial_models.json").read_text(encoding="utf-8"))
    missing = []
    for item in registry["models"]:
        report = ROOT / item["validation_report"]
        if digest(report) != item["validation_report_sha256"]:
            raise ValueError(f"Report changed: {report}")
        data = json.loads(report.read_text(encoding="utf-8-sig"))
        matches = [row for row in data["results"] if row["sha256"] == item["expected_sha256"]]
        assert len(matches) == 1, item["id"]
        row = matches[0]
        assert row["mAP50"] == item["mAP50"] and row["mAP50_95"] == item["mAP50_95"]
        print("REPORT OK:", item["id"], item["validation_report"])
        model = ROOT / item["checkpoint_path"]
        if not model.is_file():
            missing.append(item["checkpoint_path"])
            print("MODEL MISSING:", item["checkpoint_path"])
        elif digest(model) != item["expected_sha256"]:
            raise ValueError(f"Wrong model bytes: {model}")
        else:
            print("MODEL OK:", item["checkpoint_path"])
    print(f"Verified {len(registry['models'])} trial reports; missing weights: {len(missing)}")
    return 1 if args.require_models and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
