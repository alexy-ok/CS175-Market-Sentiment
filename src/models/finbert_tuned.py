import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

CHECKPOINT = "ProsusAI/finbert"
NUM_LABELS = 5
BATCH_SIZE = 4
NUM_EPOCHS = 1

ROOT = Path(__file__).resolve().parents[1]
ARTICLES_FILE = ROOT / "data" / "raw" / "guardian_articles_20260210.json"
LABEL_FILE = ROOT / "data" / "processed" / "averaged_labels.json"

def load_data():
    with open(ARTICLES_FILE, "r", encoding="utf-8") as f:
        articles = json.load(f)
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    rows = []
    for art in articles:
        key = art.get("id")
        if key is None or key not in label_map:
            continue
        fields = art.get("fields", {})
        headline = fields.get("headline", "")
        standfirst = fields.get("standfirst", "")
        body = fields.get("bodyText", "")
        text = " ".join([headline, standfirst, body]).strip()
        if not text:
            continue
        rows.append({"text": text, "label": int(label_map[key])})
    return rows

def build_datasets(rows):
    train_val, test = train_test_split(
        rows,
        test_size=0.15,
        random_state=42,
        stratify=[r["label"] for r in rows],
    )
    train, val = train_test_split(
        train_val,
        test_size=0.1765,
        random_state=42,
        stratify=[r["label"] for r in train_val],
    )
    return {
        "train": Dataset.from_list(train),
        "validation": Dataset.from_list(val),
        "test": Dataset.from_list(test),
    }


def tokenize_dataset(dataset, tokenizer):
    return dataset.map(lambda x: tokenizer(x["text"], truncation=True, max_length=512), batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "mae": mean_absolute_error(labels, preds),
    }

def main():
    rows = load_data()
    print(f"Total labeled rows: {len(rows)}")

    ds = build_datasets(rows)
    print("Split sizes:", {k: len(v) for k, v in ds.items()})

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    tokenized = {k: tokenize_dataset(v, tokenizer) for k, v in ds.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    args = TrainingArguments(
        output_dir="finbert-tuned-out",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="mae",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint="finbert-tuned-out/checkpoint-44")
    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"])

    print("Validation:", val_metrics)
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
