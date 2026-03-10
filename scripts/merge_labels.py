import json
import glob
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "zotgpt_labels_all.json"

def main():
    print("=" * 80)
    print("Merging ZotGPT Label Files")
    print("=" * 80)

    pattern = str(PROCESSED_DIR / "zotgpt_id_to_label_*.json")
    label_files = sorted(glob.glob(pattern))

    if not label_files:
        print(" No label files found!")
        print(f"Looking for: {pattern}")
        return

    print(f"\nFound {len(label_files)} label files:")
    for f in label_files:
        print(f"  - {Path(f).name}")

    all_labels = {}
    file_stats = []

    print("\nMerging files...")
    for label_file in label_files:
        try:
            with open(label_file, "r", encoding="utf-8") as f:
                labels = json.load(f)

            before_count = len(all_labels)
            all_labels.update(labels)
            after_count = len(all_labels)
            new_labels = after_count - before_count

            file_stats.append(
                {
                    "file": Path(label_file).name,
                    "labels_in_file": len(labels),
                    "new_labels": new_labels,
                    "duplicates": len(labels) - new_labels,
                }
            )

            print(
                f"{Path(label_file).name}: {len(labels)} labels ({new_labels} new, {len(labels) - new_labels} duplicates)"
            )

        except Exception as e:
            print(f"Error reading {Path(label_file).name}: {e}")

    print("\n" + "=" * 80)
    print("Merged Label Statistics")
    print("=" * 80)
    print(f"Total unique articles labeled: {len(all_labels)}")

    label_counts = Counter(all_labels.values())

    print("\nLabel distribution:")
    label_names = {
        0: "Negative",
        1: "Leaning Negative",
        2: "Neutral",
        3: "Leaning Positive",
        4: "Positive",
    }

    for label in range(5):
        count = label_counts.get(label, 0)
        percentage = 100 * count / len(all_labels) if all_labels else 0
        print(f"  {label} - {label_names[label]}: {count:4d} ({percentage:5.1f}%)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_labels, f, indent=2, ensure_ascii=False)

    print(f"\n Merged labels saved to: {OUTPUT_FILE}")
    print("=" * 80)

    print("\nFile details:")
    for stats in file_stats:
        print(f"  {stats['file']}")
        print(f"    - Labels: {stats['labels_in_file']}")
        print(f"    - New: {stats['new_labels']}")
        print(f"    - Duplicates: {stats['duplicates']}")


if __name__ == "__main__":
    main()
