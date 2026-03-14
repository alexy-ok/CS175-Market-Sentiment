import json
from pathlib import Path


def load_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def average_labels(gemini_labels, zotgpt_labels):
    averaged = {}
    
    all_ids = set(gemini_labels.keys()) | set(zotgpt_labels.keys())
    
    for article_id in all_ids:
        gemini_label = gemini_labels.get(article_id)
        zotgpt_label = zotgpt_labels.get(article_id)
        
        if gemini_label is not None and zotgpt_label is not None:
            avg = (gemini_label + zotgpt_label) / 2
            averaged[article_id] = round(avg)
        elif gemini_label is not None:
            averaged[article_id] = gemini_label
        elif zotgpt_label is not None:
            averaged[article_id] = zotgpt_label
    
    return averaged


def main():
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data" / "processed"
    
    gemini_file = data_dir / "gemini_id_to_label.json"
    zotgpt_file = data_dir / "zotgpt_labels_all.json"
    output_file = data_dir / "averaged_labels.json"

    print(f"Loading Gemini labels from {gemini_file}...")
    gemini_labels = load_labels(gemini_file)
    print(f"  Loaded {len(gemini_labels)} labels")
    
    print(f"Loading ZotGPT labels from {zotgpt_file}...")
    zotgpt_labels = load_labels(zotgpt_file)
    print(f"  Loaded {len(zotgpt_labels)} labels")

    print("Averaging labels...")
    averaged_labels = average_labels(gemini_labels, zotgpt_labels)
    print(f"  Averaged {len(averaged_labels)} labels")
    
    both_count = sum(1 for aid in averaged_labels 
                     if aid in gemini_labels and aid in zotgpt_labels)
    gemini_only = sum(1 for aid in averaged_labels 
                      if aid in gemini_labels and aid not in zotgpt_labels)
    zotgpt_only = sum(1 for aid in averaged_labels 
                      if aid not in gemini_labels and aid in zotgpt_labels)
    
    print(f"\nStatistics:")
    print(f"  Articles with both labels: {both_count}")
    print(f"  Articles with only Gemini labels: {gemini_only}")
    print(f"  Articles with only ZotGPT labels: {zotgpt_only}")
    
    if both_count > 0:
        differences = []
        for aid in averaged_labels:
            if aid in gemini_labels and aid in zotgpt_labels:
                diff = abs(gemini_labels[aid] - zotgpt_labels[aid])
                differences.append(diff)
        
        print(f"\nLabel differences (when both exist):")
        print(f"  Average absolute difference: {sum(differences) / len(differences):.2f}")
        print(f"  Max difference: {max(differences)}")
        print(f"  Articles with different labels: {sum(1 for d in differences if d > 0)}")
    
    print(f"\nSaving averaged labels to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(averaged_labels, f, indent=2, ensure_ascii=False)
    
    print("Done!")


if __name__ == "__main__":
    main()
