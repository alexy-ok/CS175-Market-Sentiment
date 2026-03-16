from transformers import pipeline
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.scripts.model_tester import evaluate_model

def run_finbert(texts, y_true):
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert"
    )
    y_pred = []
    for text in texts:
        result = finbert(
            text,
            truncation=True,
            max_length=512
        )[0]

        label = result["label"].lower()
        score = result["score"]

        if label == "negative":
            if score > 0.75:
                y_pred.append(0)
            else:
                y_pred.append(1)

        elif label == "neutral":
            y_pred.append(2)

        elif label == "positive":
            if score > 0.75:
                y_pred.append(4)
            else:
                y_pred.append(3)

    evaluate_model("FinBERT Baseline", y_true, y_pred)