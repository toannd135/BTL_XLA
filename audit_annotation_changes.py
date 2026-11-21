
from pathlib import Path
from collections import Counter

OLD = Path("work/obb/seg_data_candidate/train/labels")
NEW = Path("work/obb/seg_data_v2/train/labels")

NAMES = {0: "banana", 1: "apple", 2: "orange"}

def read_counts(path):
    counts = Counter()
    if not path.exists():
        return counts

    for line in path.read_text().splitlines():
        parts = line.split()
        if parts:
            counts[int(parts[0])] += 1

    return counts

changed = []
old_total = Counter()
new_total = Counter()
count_changes = []

for new_file in sorted(NEW.glob("*.txt")):
    old_file = OLD / new_file.name

    old_counts = read_counts(old_file)
    new_counts = read_counts(new_file)

    old_total.update(old_counts)
    new_total.update(new_counts)

    old_text = old_file.read_text() if old_file.exists() else ""
    new_text = new_file.read_text()

    if old_text.strip() != new_text.strip():
        changed.append(new_file.stem)

    if old_counts != new_counts:
        count_changes.append(
            (new_file.stem, dict(old_counts), dict(new_counts))
        )

print("===== ANNOTATION CHANGE AUDIT =====")
print("Total train labels:", len(list(NEW.glob("*.txt"))))
print("Changed annotation files:", len(changed))
print("Files with changed instance counts:", len(count_changes))

print("\n===== OLD VS NEW INSTANCE COUNTS =====")
for cls, name in NAMES.items():
    print(
        f"{name}: old={old_total[cls]}, "
        f"new={new_total[cls]}, "
        f"delta={new_total[cls] - old_total[cls]}"
    )

print("\n===== IMAGES WITH CHANGED INSTANCE COUNTS =====")
for stem, old, new in count_changes:
    print(f"{stem}: old={old} -> new={new}")

print("\n===== FIRST 30 CHANGED FILES =====")
for stem in changed[:30]:
    print(stem)
