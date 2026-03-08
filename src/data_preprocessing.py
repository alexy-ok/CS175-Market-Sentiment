import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

sys.path.append(str(Path(__file__).parent.parent))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
)


class DataPreprocessor:

    def __init__(self):
        self.data = None

    def load_articles(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            articles = json.load(f)

        print(f"Loaded {len(articles)} articles")
        return articles

    def extract_text_features(self, articles):
        processed_data = []

        for article in articles:
            article_id = article.get("id", "")
            headline = article.get("webTitle", "")

            fields = article.get("fields", {})
            body_text = fields.get("bodyText", "")
            standfirst = fields.get("standfirst", "")

            combined_text = f"{headline}. {standfirst}".strip()

            if not combined_text:
                continue

            processed_data.append(
                {
                    "article_id": article_id,
                    "headline": headline,
                    "text": combined_text,
                    "full_body": body_text,
                    "published_date": article.get("webPublicationDate", ""),
                    "url": article.get("webUrl", ""),
                }
            )

        df = pd.DataFrame(processed_data)
        print(f"Extracted {len(df)} articles with valid text")
        return df

    def clean_text(self, df):
        df = df.drop_duplicates(subset=["text"])

        df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
        df["text"] = df["text"].str.strip()

        df = df[df["text"].str.len() >= 20]

        print(f"After cleaning: {len(df)} articles")
        return df

    def save_for_labeling(self, df, output_file):
        labeling_df = df[["article_id", "headline", "text"]].copy()
        labeling_df["sentiment"] = ""

        output_path = PROCESSED_DATA_DIR / output_file
        labeling_df.to_csv(output_path, index=False, encoding="utf-8")

        print(f"Saved {len(labeling_df)} articles for labeling to {output_path}")
        print(
            "\nPlease label the 'sentiment' column with: positive, neutral, or negative"
        )
        print("Based on the article's sentiment towards US markets/economy")

    def load_labeled_data(self, filepath):
        df = pd.read_csv(filepath, encoding="utf-8")

        df = df[df["sentiment"].notna() & (df["sentiment"] != "")]

        df["sentiment"] = df["sentiment"].str.lower().str.strip()

        valid_labels = ["positive", "neutral", "negative"]
        df = df[df["sentiment"].isin(valid_labels)]

        print(f"Loaded {len(df)} labeled articles")
        print(f"Label distribution:\n{df['sentiment'].value_counts()}")

        return df

    def create_train_val_test_split(self, df):

        train_val_df, test_df = train_test_split(
            df, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=df["sentiment"]
        )

        val_ratio = VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=RANDOM_SEED,
            stratify=train_val_df["sentiment"],
        )

        print(f"\nDataset split:")
        print(f"Train: {len(train_df)} samples")
        print(f"Validation: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")

        return train_df, val_df, test_df

    def save_splits(self, train_df, val_df, test_df):
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
        val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

        print(f"\nSaved splits to {PROCESSED_DATA_DIR}")


def process_raw_data():
    preprocessor = DataPreprocessor()

    raw_files = list(RAW_DATA_DIR.glob("guardian_articles_*.json"))

    if not raw_files:
        print("No raw data files found. Please run data_collection.py first.")
        return

    latest_file = max(raw_files, key=lambda x: x.stat().st_mtime)
    print(f"Processing {latest_file.name}...")

    articles = preprocessor.load_articles(latest_file)
    df = preprocessor.extract_text_features(articles)
    df = preprocessor.clean_text(df)

    preprocessor.save_for_labeling(df, "articles_for_labeling.csv")


def create_training_data():
    preprocessor = DataPreprocessor()

    labeled_file = PROCESSED_DATA_DIR / "articles_labeled.csv"

    if not labeled_file.exists():
        print(f"Labeled data file not found: {labeled_file}")
        print(
            "Please label the data in 'articles_for_labeling.csv' and save as 'articles_labeled.csv'"
        )
        return

    df = preprocessor.load_labeled_data(labeled_file)

    train_df, val_df, test_df = preprocessor.create_train_val_test_split(df)

    preprocessor.save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Guardian articles")
    parser.add_argument(
        "--mode",
        choices=["prepare", "split"],
        default="prepare",
        help="Mode: 'prepare' to create labeling file, 'split' to create train/val/test splits",
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        process_raw_data()
    else:
        create_training_data()
