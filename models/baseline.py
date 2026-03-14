import json
import os


import logistic_regression
import finbert_baseline

# Load JSON 
script_dir = os.path.dirname(os.path.abspath(__file__))

# File Structure
file_path = os.path.join(
    script_dir, "..", "data", "raw", "guardian_articles_20260210.json"
)
file_path = os.path.abspath(file_path)

with open(file_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Populating Texts and Labels
articles = articles[:100]
texts = []
labels = []
for art in articles:
    headline = art["fields"].get("headline", "")
    standfirst = art["fields"].get("standfirst", "")
    body = art["fields"].get("bodyText", "")

    full_text = headline + " " + standfirst + " " + body
    texts.append(full_text)
    labels.append(art.get("sentiment", "Neutral"))  # Default to Neutral if not present

# Label Mapping
label2id = {
    "Negative": 0,
    "Leaning Negative": 1,
    "Neutral": 2,
    "Leaning Positive": 3,
    "Positive": 4
}

id2label = {v: k for k, v in label2id.items()}
y_true = [label2id[label] for label in labels]

logistic_regression.run_logistic_regression(texts, y_true)
finbert_baseline.run_finbert(texts, y_true)

