import json
import os
import logistic_regression
import finbert_baseline

script_dir = os.path.dirname(os.path.abspath(__file__))

# Open articles
file_path = os.path.join(
    script_dir, "..", "data", "raw", "guardian_articles_20260210.json"
)
file_path = os.path.abspath(file_path)

with open(file_path, "r", encoding="utf-8") as f:
    articles = json.load(f)

# Get label mapping
def load_averaged_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label_map = {}
    for id, score in data.items():
        label_map[id] = int(score)

    return label_map

# Map labels to articles w/ texts
def build_dataset_from_labels(articles, label_map):
    texts = []
    labels = []

    for art in articles:
        id = art.get("id")
        if id not in label_map:
            continue

        headline = art["fields"].get("headline", "")
        standfirst = art["fields"].get("standfirst", "")
        body = art["fields"].get("bodyText", "")

        text = headline + " " + standfirst + " " + body

        texts.append(text)
        labels.append(label_map[id])

    return texts, labels

# Remove later
articles = articles[:100]

averaged_labels = load_averaged_labels(
    "data/processed/averaged_labels.json"
)
texts_eval, y_eval = build_dataset_from_labels(articles, averaged_labels)

# Run models
logistic_regression.run_logistic_regression(texts_eval, y_eval)
finbert_baseline.run_finbert(texts_eval, y_eval)

