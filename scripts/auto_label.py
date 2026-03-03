import json
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_FILE = BASE_DIR / "data" / "raw" / "guardian_articles_20260210_224419.json"

def count_articles():
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    return len(articles)

def main():
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    print("="*80)
    print("Automated ZotGPT Labeling")
    print("="*80)
    
    total_articles = count_articles()
    print(f"Total articles in dataset: {total_articles}")
    print(f"Starting from index: {start_idx}")
    print(f"Batch size: {batch_size}")
    print(f"Articles to label: {total_articles - start_idx}")
    print("="*80)
    
    if start_idx >= total_articles:
        print(f"\nStart index {start_idx} is beyond total articles {total_articles}")
        return
    
    input("\nPress Enter to start labeling")
    
    current_idx = start_idx
    batch_num = 1
    successful_batches = 0
    failed_batches = 0
    
    while current_idx < total_articles:
        remaining = total_articles - current_idx
        current_batch = min(batch_size, remaining)
        
        print(f"\n{'='*80}")
        print(f"BATCH {batch_num}: Articles {current_idx}-{current_idx + current_batch - 1}")
        print(f"Progress: {current_idx}/{total_articles} ({100*current_idx/total_articles:.1f}% complete)")
        print(f"Successful: {successful_batches} | Failed: {failed_batches}")
        print(f"{'='*80}\n")
        
        cmd = [sys.executable, "src/gpt_label.py", str(current_idx), str(batch_size)]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=BASE_DIR,
                check=True
            )
            
            print(f"\nBatch {batch_num} completed successfully!")
            successful_batches += 1
            
        except subprocess.CalledProcessError as e:
            print(f"\nError in batch {batch_num}")
            print(f"Exit code: {e.returncode}")
            failed_batches += 1
        
        current_idx += batch_size
        batch_num += 1
    
    print("\n" + "="*80)
    print("🎉 ALL BATCHES COMPLETED!")
    print("="*80)
    print(f"Total articles labeled: {current_idx - start_idx}")
    print(f"Successful batches: {successful_batches}")
    print(f"Failed batches: {failed_batches}")
    print(f"\nLabel files saved in: data/processed/")
    print("\nTo merge all label files, run:")
    print("  python scripts/merge_labels.py")
    print("="*80)


if __name__ == "__main__":
    main()
