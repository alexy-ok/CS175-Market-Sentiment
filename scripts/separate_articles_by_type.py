"""
Script to separate Guardian articles by type (liveblog vs article).
Creates two separate JSON files for each article type.
"""

import json
import sys
from pathlib import Path
from collections import Counter

# Ensure UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


def load_articles(file_path):
    """Load articles from JSON file."""
    print(f"Loading articles from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def separate_by_type(articles):
    """
    Separate articles by type.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Dictionary with article types as keys and lists of articles as values
    """
    separated = {}
    
    for article in articles:
        article_type = article.get('type', 'unknown')
        if article_type not in separated:
            separated[article_type] = []
        separated[article_type].append(article)
    
    return separated


def save_articles(articles, output_path, article_type):
    """Save articles to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(articles)} {article_type} articles to: {output_path}")


def calculate_statistics(articles):
    """Calculate statistics for a list of articles."""
    if not articles:
        return None
    
    # Parse dates
    dates = []
    for article in articles:
        date_str = article.get('webPublicationDate', '')
        if date_str:
            dates.append(date_str[:10])  # YYYY-MM-DD
    
    # Word counts
    word_counts = []
    for article in articles:
        if 'fields' in article and 'bodyText' in article['fields']:
            body = article['fields']['bodyText']
            word_count = len(body.split())
            word_counts.append(word_count)
    
    stats = {
        'count': len(articles),
        'date_range': f"{min(dates)} to {max(dates)}" if dates else "N/A",
        'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
        'min_word_count': min(word_counts) if word_counts else 0,
        'max_word_count': max(word_counts) if word_counts else 0,
    }
    
    return stats


def main():
    # Paths
    base_path = Path(__file__).parent.parent
    input_file = base_path / 'data' / 'raw' / 'guardian_articles_20260210_224419.json'
    output_dir = base_path / 'data' / 'raw'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not input_file.exists():
        print(f"Error: File not found at {input_file}")
        return
    
    # Load articles
    articles = load_articles(input_file)
    print(f"Loaded {len(articles)} total articles\n")
    
    # Separate by type
    print("Separating articles by type...")
    separated = separate_by_type(articles)
    
    # Count types
    print("\nArticle type distribution:")
    for article_type, article_list in separated.items():
        print(f"  - {article_type}: {len(article_list)} articles")
    
    # Save separated files
    print("\nSaving separated files...")
    print("="*80)
    
    stats_summary = {}
    
    for article_type, article_list in separated.items():
        # Create filename
        output_file = output_dir / f'guardian_{article_type}s_20260210.json'
        
        # Save articles
        save_articles(article_list, output_file, article_type)
        
        # Calculate statistics
        stats = calculate_statistics(article_list)
        stats_summary[article_type] = stats
    
    # Print detailed statistics
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)
    
    for article_type, stats in stats_summary.items():
        if stats:
            print(f"\n{article_type.upper()}:")
            print(f"  Total count: {stats['count']}")
            print(f"  Date range: {stats['date_range']}")
            print(f"  Average word count: {stats['avg_word_count']:.0f}")
            print(f"  Shortest article: {stats['min_word_count']} words")
            print(f"  Longest article: {stats['max_word_count']} words")
    
    print("\n" + "="*80)
    print(f"All separated files saved to: {output_dir}")
    print("="*80)
    
    # Create a summary file
    summary_file = output_dir / 'separation_summary.json'
    summary_data = {
        'total_articles': len(articles),
        'types': {k: {'count': v['count'], 'file': f'guardian_{k}s_20260210.json'} 
                  for k, v in stats_summary.items()},
        'statistics': stats_summary
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
