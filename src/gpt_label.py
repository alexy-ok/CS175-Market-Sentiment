import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()

ZOTGPT_API_KEY = os.getenv("ZOTGPT_API_KEY")
ZOTGPT_BASE_URL = "https://azureapi.zotgpt.uci.edu/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-01"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


class ZotGPTLabeler:

    def __init__(self, api_key, base_url=ZOTGPT_BASE_URL):
        if not api_key:
            raise ValueError(
                "ZotGPT API key not found. Please set ZOTGPT_API_KEY in .env file"
            )

        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
            "Cache-Control": "no-cache",
        }

    def create_sentiment_prompt(self, article):
        title = article.get("webTitle", "")
        body = article.get("fields", {}).get("bodyText", "")

        text_to_analyze = f"{title}\n\n{body[:2000]}"

        prompt = f"""You are a financial sentiment analysis expert. Analyze the following Guardian business article and classify its sentiment regarding the US stock market.

Article:
{text_to_analyze}

Classify the sentiment into one of these categories:
4 - POSITIVE: Optimistic outlook, growth, gains, bullish sentiment, or favorable conditions for US stocks
3 - LEANING POSITIVE: Mostly positive but with some caveats or qualifiers
2 - NEUTRAL: Balanced view, mixed signals, or not directly related to US stock market sentiment
1 - LEANING NEGATIVE: Mostly negative but with some caveats or qualifiers
0 - NEGATIVE: Pessimistic outlook, losses, bearish sentiment, concerns, or unfavorable conditions for US stocks

Respond with ONLY the number (0, 1, 2, 3, or 4) and nothing else."""

        return prompt

    def label_article(self, article):

        article_id = article.get("id", "")
        prompt = self.create_sentiment_prompt(article)

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a financial sentiment analysis expert. Respond only with a number: 0 for negative, 1 for leaning negative, 2 for neutral, 3 for leaning positive, or 4 for positive.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "max_tokens": 10,
        }

        try:
            response = requests.post(
                f"{self.base_url}", headers=self.headers, json=payload, timeout=30
            )
            response.raise_for_status()

            result = response.json()
            label_text = result["choices"][0]["message"]["content"].strip()

            try:
                label = int(label_text)
                if label not in [0, 1, 2, 3, 4]:
                    print(
                        f"Warning: Invalid label {label} for article {article_id}. Defaulting to 2 (neutral)"
                    )
                    label = 2
            except ValueError:
                print(
                    f"Warning: Could not parse label '{label_text}' for article {article_id}. Defaulting to 2 (neutral)"
                )
                label = 2

            return {
                "article_id": article_id,
                "label": label,
                "raw_response": label_text,
            }

        except requests.exceptions.RequestException as e:
            print(f"Error labeling article {article_id}: {str(e)}")
            return {"article_id": article_id, "label": 2, "error": str(e)}

    def label_articles_batch(self, articles, start_idx=0, count=50, delay=1.0):
        labels = {}
        total = len(articles)
        end_idx = min(start_idx + count, total)

        print(f"\nLabeling articles {start_idx} to {end_idx-1} of {total}")
        print("=" * 80)

        for idx in range(start_idx, end_idx):
            article = articles[idx]
            article_id = article.get("id", f"article_{idx}")

            if "type" in article and article["type"] == "liveblog":
                print(f"[{idx+1}/{total}] Skipping liveblog: {article_id[:60]}...")
                continue

            print(f"[{idx+1}/{total}] Labeling: {article_id[:60]}...")

            result = self.label_article(article)
            labels[result["article_id"]] = result["label"]

            if "error" not in result:
                label_names = [
                    "Negative",
                    "Leaning Negative",
                    "Neutral",
                    "Leaning Positive",
                    "Positive",
                ]
                print(f"  → Label: {result['label']} ({label_names[result['label']]})")
            else:
                print(f"  → Error: {result['error']}")

            time.sleep(delay)

        print("=" * 80)
        print(f"Labeled {len(labels)} articles")

        return labels

    def save_labels(self, labels, filename):
        filepath = PROCESSED_DATA_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)

        print(f"\nSaved labels to: {filepath}")

        label_counts = [0] * 5
        for label in labels.values():
            if 0 <= label <= 4:
                label_counts[label] += 1

        print("\nLabel distribution:")
        print(f"  Negative (0): {label_counts[0]}")
        print(f"  Leaning Negative (1): {label_counts[1]}")
        print(f"  Neutral (2): {label_counts[2]}")
        print(f"  Leaning Positive (3): {label_counts[3]}")
        print(f"  Positive (4): {label_counts[4]}")


def load_articles(file_path):
    print(f"Loading articles from: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    start_idx = 0
    count = 50

    if len(sys.argv) > 1:
        start_idx = int(sys.argv[1])
    if len(sys.argv) > 2:
        count = int(sys.argv[2])

    try:
        labeler = ZotGPTLabeler(ZOTGPT_API_KEY)

        input_file = RAW_DATA_DIR / "guardian_articles_20260210_224419.json"

        if not input_file.exists():
            print(f"Error: File not found at {input_file}")
            return

        articles = load_articles(input_file)
        print(f"Loaded {len(articles)} articles")
        print(f"Starting at index {start_idx}, labeling {count} articles\n")

        labels = labeler.label_articles_batch(
            articles, start_idx=start_idx, count=count, delay=1.0
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"zotgpt_id_to_label_{start_idx}to{start_idx+count}_{timestamp}.json"
        labeler.save_labels(labels, filename)

        print("\nLabeling complete!")

        if start_idx + count < len(articles):
            next_idx = start_idx + count
            print(f"\nTo continue, run: python src/gpt_label.py {next_idx} {count}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
