import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

ZOTGPT_API_KEY = os.getenv("GPT_KEY")
ZOTGPT_API_URL = "https://azureapi.zotgpt.uci.edu/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"

BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


MAX_RETRIES = 3
RETRY_DELAY = 2 
REQUEST_DELAY = 1

INPUT_TOKEN_COST = 0.0025 / 1000   # $0.0025 per 1K input tokens
OUTPUT_TOKEN_COST = 0.010 / 1000   # $0.010 per 1K output tokens  


class ZotGPTLabeler:
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("ZotGPT API key not found. Please set GPT_KEY in .env file")
        
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "api-key": self.api_key
        }
        

        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_api_calls = 0
        self.failed_api_calls = 0
        
    def create_prompt(self, article: Dict) -> str:
        headline = article.get("webTitle", "")
        fields = article.get("fields", {})
        standfirst = fields.get("standfirst", "")
        body_text = fields.get("bodyText", "")
        

        text_for_analysis = f"Headline: {headline}\n\n"
        
        if standfirst:
            import re
            clean_standfirst = re.sub(r'<[^>]+>', '', standfirst)
            text_for_analysis += f"Summary: {clean_standfirst}\n\n"
        
        if body_text:
            text_for_analysis += f"Excerpt: {body_text[:1500]}\n\n"
        
        prompt = f"""Analyze the following article and classify its sentiment/outlook on the US stock market.

{text_for_analysis}

Task: Classify the article's sentiment toward the US stock market into one of five categories:

POSITIVE - The article suggests optimistic outlook, growth, gains, bullish sentiment, or favorable conditions for US stocks
LEANING POSITIVE - The article suggests an optimistic outlook, growth, gains, bullish sentiment, or favorable conditions for US stocks, but with some caveats or qualifiers
NEUTRAL - The article presents balanced view, mixed signals, or is not directly related to US stock market sentiment
LEANING NEGATIVE - The article suggests a pessimistic outlook, losses, bearish sentiment, concerns, or unfavorable conditions for US stocks, but with some caveats or qualifiers
NEGATIVE - The article suggests pessimistic outlook, losses, bearish sentiment, concerns, or unfavorable conditions for US stocks

Important Guidelines:
- Focus on implications for US stock market, not just general economic news
- Consider the overall tone and market implications
- Look for keywords like: stocks, shares, markets, investors, Wall Street, S&P, Dow Jones, Nasdaq, trading, earnings

Respond with ONLY ONE NUMBER: 4 - POSITIVE, 3 - LEANING POSITIVE, 2 - NEUTRAL, 1 - LEANING NEGATIVE, 0 - NEGATIVE"""
        
        return prompt
    
    def parse_sentiment(self, response: str) -> str:
        if not response:
            return "neutral"
        
        response_upper = response.upper().strip()
        
        # Check for numeric response (1-5)
        if "4" in response_upper or "POSITIVE" in response_upper:
            return "positive"
        elif "3" in response_upper or "LEANING POSITIVE" in response_upper:
            return "leaning_positive"
        elif "2" in response_upper or "NEUTRAL" in response_upper:
            return "neutral"
        elif "1" in response_upper or "LEANING NEGATIVE" in response_upper:
            return "leaning_negative"
        elif "0" in response_upper or "NEGATIVE" in response_upper:
            return "negative"
        else:
            return "neutral" 
    
    def estimate_tokens(self, text: str) -> int:
        return len(text) // 4
    
    def call_zotgpt_api(self, prompt: str) -> Optional[Dict]:
        payload = {
            "messages": [{
                "role": "user", 
                "content": prompt
            }]
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.post(
                    ZOTGPT_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()

                    self.total_api_calls += 1
                    
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    
                    if prompt_tokens == 0:
                        prompt_tokens = self.estimate_tokens(prompt)
                    if completion_tokens == 0:
                        response_text = result.get("response", "")
                        completion_tokens = self.estimate_tokens(response_text)
                    
                    self.total_input_tokens += prompt_tokens
                    self.total_output_tokens += completion_tokens
                    
                    return {
                        "response": result.get("response", "").strip(),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens
                    }
                elif response.status_code == 429:  # Rate limit
                    print(f"Rate limit hit, waiting {RETRY_DELAY * (attempt + 1)} seconds...")
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    print(f"API error (attempt {attempt + 1}/{MAX_RETRIES}): Status {response.status_code}")
                    print(f"Response: {response.text}")
                    time.sleep(RETRY_DELAY)
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                time.sleep(RETRY_DELAY)
        
        # Track failed call
        self.failed_api_calls += 1
        return None
    
    
    def label_article(self, article: Dict) -> Dict:
        article_id = article.get("id", "unknown")
        
        prompt = self.create_prompt(article)
        
        api_result = self.call_zotgpt_api(prompt)
        
        if api_result:
            response_text = api_result["response"]
            sentiment = self.parse_sentiment(response_text)
        else:
            response_text = None
            sentiment = "neutral"  # Default on failure
        
        return {
            "article_id": article_id,
            "sentiment": sentiment,
            "raw_response": response_text,
            "headline": article.get("webTitle", ""),
            "url": article.get("webUrl", "")
        }
    
    def estimate_cost(self, articles: List[Dict]) -> Dict:

        sample_prompts = [self.create_prompt(article) for article in articles[:min(5, len(articles))]]
        avg_prompt_tokens = sum(self.estimate_tokens(p) for p in sample_prompts) / len(sample_prompts)

        estimated_completion_tokens = 30

        total_input_tokens = int(avg_prompt_tokens * len(articles))
        total_output_tokens = int(estimated_completion_tokens * len(articles))
        
        input_cost = total_input_tokens * INPUT_TOKEN_COST
        output_cost = total_output_tokens * OUTPUT_TOKEN_COST
        total_cost = input_cost + output_cost
        
        return {
            "num_articles": len(articles),
            "estimated_input_tokens": total_input_tokens,
            "estimated_output_tokens": total_output_tokens,
            "estimated_total_tokens": total_input_tokens + total_output_tokens,
            "estimated_input_cost": input_cost,
            "estimated_output_cost": output_cost,
            "estimated_total_cost": total_cost
        }
    
    def label_articles(self, articles: List[Dict], output_file: str, 
                      resume_from: Optional[str] = None) -> List[Dict]:
        labeled_articles = []
        checkpoint_file = PROCESSED_DATA_DIR / f"{output_file}.checkpoint.json"
        
        start_idx = 0
        if resume_from and Path(resume_from).exists():
            print(f"Resuming from checkpoint: {resume_from}")
            with open(resume_from, 'r', encoding='utf-8') as f:
                labeled_articles = json.load(f)
            start_idx = len(labeled_articles)
            print(f"Resuming from article {start_idx + 1}/{len(articles)}")
        
        print(f"\nLabeling {len(articles) - start_idx} articles using ZotGPT API...")
        
        for i, article in enumerate(tqdm(articles[start_idx:], initial=start_idx, total=len(articles))):
            try:
                labeled = self.label_article(article)
                labeled_articles.append(labeled)
                
                if (i + 1) % 10 == 0:
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(labeled_articles, f, indent=2, ensure_ascii=False)
                
                time.sleep(REQUEST_DELAY)
                
            except Exception as e:
                print(f"\nError processing article {article.get('id', 'unknown')}: {e}")
                # Save checkpoint on error
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump(labeled_articles, f, indent=2, ensure_ascii=False)
                continue

        output_path = PROCESSED_DATA_DIR / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeled_articles, f, indent=2, ensure_ascii=False)
        
        print(f"\nLabeled {len(labeled_articles)} articles")
        print(f"Results saved to: {output_path}")
        
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        return labeled_articles
    
    def get_cost_summary(self) -> Dict:

        input_cost = self.total_input_tokens * INPUT_TOKEN_COST
        output_cost = self.total_output_tokens * OUTPUT_TOKEN_COST
        total_cost = input_cost + output_cost
        
        return {
            "total_api_calls": self.total_api_calls,
            "failed_api_calls": self.failed_api_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
    
    def print_cost_summary(self):
        """Print detailed cost summary."""
        cost_info = self.get_cost_summary()
        
        print("\n" + "="*60)
        print("COST SUMMARY")
        print("="*60)
        print(f"API Calls:")
        print(f"  Successful: {cost_info['total_api_calls']}")
        print(f"  Failed:     {cost_info['failed_api_calls']}")
        print(f"\nToken Usage:")
        print(f"  Input tokens:  {cost_info['total_input_tokens']:,}")
        print(f"  Output tokens: {cost_info['total_output_tokens']:,}")
        print(f"  Total tokens:  {cost_info['total_tokens']:,}")
        print(f"\nEstimated Cost (GPT-4o pricing):")
        print(f"  Input cost:  ${cost_info['input_cost']:.4f}")
        print(f"  Output cost: ${cost_info['output_cost']:.4f}")
        print(f"  Total cost:  ${cost_info['total_cost']:.4f}")
        print("="*60)
    
    def print_statistics(self, labeled_articles: List[Dict]):
        """Print labeling statistics."""
        from collections import Counter
        
        sentiments = [a['sentiment'] for a in labeled_articles]
        counts = Counter(sentiments)
        
        print("\n" + "="*60)
        print("LABELING STATISTICS")
        print("="*60)
        print(f"Total articles labeled: {len(labeled_articles)}")
        print(f"\nSentiment distribution:")
        
        sentiment_labels = [
            ('positive', 'Positive'),
            ('leaning_positive', 'Leaning Positive'),
            ('neutral', 'Neutral'),
            ('leaning_negative', 'Leaning Negative'),
            ('negative', 'Negative')
        ]
        
        for key, label in sentiment_labels:
            count = counts.get(key, 0)
            pct = (count / len(labeled_articles) * 100) if len(labeled_articles) > 0 else 0
            print(f"  {label:18s}: {count:3d} ({pct:5.1f}%)")
        
        print("="*60)


def load_articles(filepath: Path) -> List[Dict]:
    """Load articles from JSON file."""
    print(f"Loading articles from: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    print(f"Loaded {len(articles)} articles")
    return articles


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Label Guardian articles using ZotGPT API"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/guardian_articles_20260210_224419.json",
        help="Input JSON file with articles (relative to project root)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="zotgpt_labeled_articles.json",
        help="Output filename for labeled articles (saved in data/processed/)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from checkpoint file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of articles to process (for testing)"
    )
    
    args = parser.parse_args()
    
    input_path = BASE_DIR / args.input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return
    
    articles = load_articles(input_path)
    
    if args.limit:
        articles = articles[:args.limit]
        print(f"Limited to first {args.limit} articles")

    try:
        labeler = ZotGPTLabeler(ZOTGPT_API_KEY)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Show cost estimate
    print("\n" + "="*60)
    print("COST ESTIMATE")
    print("="*60)
    estimate = labeler.estimate_cost(articles)
    print(f"Articles to process: {estimate['num_articles']}")
    print(f"\nEstimated token usage:")
    print(f"  Input tokens:  {estimate['estimated_input_tokens']:,}")
    print(f"  Output tokens: {estimate['estimated_output_tokens']:,}")
    print(f"  Total tokens:  {estimate['estimated_total_tokens']:,}")
    print(f"\nEstimated cost (GPT-4o pricing):")
    print(f"  Input cost:  ${estimate['estimated_input_cost']:.4f}")
    print(f"  Output cost: ${estimate['estimated_output_cost']:.4f}")
    print(f"  Total cost:  ${estimate['estimated_total_cost']:.4f}")
    print("="*60)
    
    # Confirmation prompt for costs over $1
    if estimate['estimated_total_cost'] > 1.0:
        response = input(f"\n  Estimated cost is ${estimate['estimated_total_cost']:.2f}. Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Cancelled by user.")
            return

    resume_checkpoint = args.resume or (PROCESSED_DATA_DIR / f"{args.output}.checkpoint.json")
    labeled_articles = labeler.label_articles(
        articles,
        args.output,
        resume_from=resume_checkpoint if Path(resume_checkpoint).exists() else None
    )
    

    labeler.print_statistics(labeled_articles)
    labeler.print_cost_summary()
    
    print("\nLabeling complete!")
    print(f"Output file: {PROCESSED_DATA_DIR / args.output}")


if __name__ == "__main__":
    main()
