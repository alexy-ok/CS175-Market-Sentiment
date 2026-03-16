import json, random, sys, datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# mac-config
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

RAW_DATA_FILE = max(
    RAW_DATA_DIR.glob("guardian_articles_*.json"),
    key=lambda x: x.stat().st_mtime
)

# config
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE = 32

INT_TO_LABEL = {
    0: "negative",
    1: "leaning negative",
    2: "neutral",
    3: "leaning positive",
    4: "positive",
}

LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}
LABELS = [INT_TO_LABEL[i] for i in range(5)]

SYSTEM_PROMPT = (
    "Analyze the following article and classify its sentiment/outlook on the US stock market."
    "Task: Classify the article's sentiment toward the US stock market into one of five categories:"
    "POSITIVE - The article suggests optimistic outlook, growth, gains, bullish sentiment, or favorable conditions for US stocks"
    "LEANING POSITIVE - The article suggests an optimistic outlook, growth, gains, bullish sentiment, or favorable conditions for US stocks, but with some caveats or qualifiers"
    "NEUTRAL - The article presents balanced view, mixed signals, or is not directly related to US stock market sentiment"
    "LEANING NEGATIVE - The article suggests a pessimistic outlook, losses, bearish sentiment, concerns, or unfavorable conditions for US stocks, but with some caveats or qualifiers"
    "NEGATIVE - The article suggests pessimistic outlook, losses, bearish sentiment, concerns, or unfavorable conditions for US stocks"
    "Respond with ONLY ONE NUMBER: 4 - POSITIVE, 3 - LEANING POSITIVE, 2 - NEUTRAL, 1 - LEANING NEGATIVE, 0 - NEGATIVE"
)

def sample_few_shot_examples(articles, labels, seed=42):
    rng = random.Random(seed)

    # build pool per class
    pool: dict[int, list[dict]] = {i : [] for i in range(5)}

    for article in articles:
        if article.get("type") == "liveblog":
            continue

        article_id = article.get("id", "")
        if article_id in labels:
            pool[labels[article_id]].append(article)
        
    examples = []
    
    for int_label in range(5):
        candidates = pool[int_label]

        if not candidates:
            print(f"Warning: no articles found for label {int_label} ({INT_TO_LABEL[int_label]})")
            continue
            
        article = rng.choice(candidates)
        text = format_article_text(article)
        str_label = " ".join(w.capitalize() for w in INT_TO_LABEL[int_label].split())
        examples.append((text, str_label))
        print(f"  Few-shot [{str_label}]: {article.get('webTitle', '')[:70]}")

    return examples

def get_few_shot_ids(articles, labels, seed=42):
    rng = random.Random(seed)

    pool: dict[int, list[str]] = {i: [] for i in range(5)}

    for article in articles:
        if article.get("type") == "liveblog":
            continue
        article_id = article.get("id", "")
        if article_id in labels:
            pool[labels[article_id]].append(article_id)
    chosen = set()
    for candidates in pool.values():
        if candidates:
            chosen.add(rng.choice(candidates))

    return chosen

# load data
def load_articles(path=RAW_DATA_FILE):
    print(f"Loading articles from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_labels(label_file):
    print(f"Loading labels from: {label_file}")
    with open(label_file, "r", encoding="utf-8") as f:
        return json.load(f)

def format_article_text(article: dict) -> str:
    """Mirrors ZotGPT's text construction: title + body[:2000]"""
    title = article.get("webTitle", "")
    body  = article.get("fields", {}).get("bodyText", "")
    return f"{title}\n\n{body[:2000]}"

def prepare_dataset(articles, labels, exclude_ids: set[str] | None = None):
    exclude_ids = exclude_ids or set()
    ids, texts, string_labels = [], [], []

    for article in articles:
        article_id   = article.get("id", "")
        article_type = article.get("type", "")

        if article_type == "liveblog":
            continue
        if article_id not in labels:
            continue
        if article_id in exclude_ids:
            continue

        ids.append(article_id)
        texts.append(format_article_text(article))
        string_labels.append(INT_TO_LABEL[labels[article_id]])

    print(f"Matched {len(ids)} labeled, non-liveblog articles")
    return ids, texts, string_labels

# load model
def load_pipeline(model_id=MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tokenizer,
        dtype=torch.float16, # change to float16 for MPS (Apple silicon gpu)
        # device="cpu", # add for mac
        max_new_tokens=8,
        do_sample=False
    )
    return pipe

# build prompt
def build_zero_shot_msgs(article_text):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Article:\n{article_text}"},
    ]

def build_few_shot_msgs(article_text, examples):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex_text, ex_label in examples:
        messages.append({"role": "user",      "content": f"Article:\n{ex_text}"})
        messages.append({"role": "assistant", "content": ex_label})
    messages.append({"role": "user", "content": f"Article:\n{article_text}"})

    return messages

def parse_label(raw):
    text = raw.lower().strip()
    text = text.replace("sentiment:", "").replace("-", " ")

    for label in sorted(LABELS, key=len, reverse=True):
        if label in text:
            return label

    return "unknown"

def predict_batch(pipe, article_texts, mode, few_shot_examples=None):
    all_messages = [
        build_zero_shot_msgs(text) if mode == "zero_shot"
        else build_few_shot_msgs(text, few_shot_examples)
        for text in article_texts
    ]

    predictions = []
    total = len(all_messages)
    for i in range(0, total, BATCH_SIZE):
        batch = all_messages[i : i + BATCH_SIZE]
        print(f"  Inferring {i}–{min(i + BATCH_SIZE, total) - 1} / {total}…")

        outputs = pipe(batch)
        for out in outputs:
            last_msg = out[0]["generated_text"][-1]["content"]
            predictions.append(parse_label(last_msg))

    return predictions

# evaluate
def evaluate(predictions, true_labels):
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, labels=LABELS, zero_division=0)
    cm = confusion_matrix(true_labels, predictions, labels=LABELS)
    return {"accuracy": acc, "report": report, "confusion matrix": cm}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        label_file = Path(sys.argv[1])
    else:
        label_file = PROCESSED_DATA_DIR / "zotgpt_labels_all.json"
        if not label_file.exists():
            print(f"Label file not found: {label_file}")
            print("Run scripts/merge_labels.py first, or use --test for dummy data.")
            sys.exit(1)
    articles      = load_articles()
    zotgpt_labels = load_labels(label_file)

    # sample few-shot examples, exclude from evaluation
    print("\nSampling few-shot examples from labeled dataset:")
    few_shot_examples = sample_few_shot_examples(articles, zotgpt_labels)
    few_shot_ids = get_few_shot_ids(articles, zotgpt_labels)

    ids, texts, true_labels = prepare_dataset(
        articles, zotgpt_labels, exclude_ids=few_shot_ids
    )

    print(f"\nLabel distribution in ground truth ({len(ids)} articles):")
    for label in LABELS:
        print(f"{label}: {true_labels.count(label)}")

    # load model
    print("\nLoading model...")
    pipe = load_pipeline()

    # zero-shot
    print("\n" + "=" * 80)
    print("0-SHOT BASELINE")
    print("=" * 80)
    zero_preds = predict_batch(pipe, texts, mode="zero_shot")

    unknown_0 = zero_preds.count("unknown")
    if unknown_0:
        print(f"Warning: {unknown_0} unparseable predictions — check raw outputs")

    results_0 = evaluate(zero_preds, true_labels)
    print(f"\nAccuracy: {results_0['accuracy']:.3f}")
    print(results_0["report"])

    # few-shot
    print("\n" + "=" * 80)
    print("FEW-SHOT BASELINE")
    print("=" * 80)
    few_preds = predict_batch(pipe, texts, mode="few_shot", few_shot_examples=few_shot_examples)

    unknown_k = few_preds.count("unknown")
    if unknown_k:
        print(f"Warning: {unknown_k} unparseable predictions — check raw outputs")

    results_k = evaluate(few_preds, true_labels)
    print(f"\nAccuracy: {results_k['accuracy']:.3f}")
    print(results_k["report"])

    # save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "model": MODEL_ID,
        "timestamp": timestamp,
        "batch_size": BATCH_SIZE,
        "zero_shot": {
            "accuracy": results_0["accuracy"],
            "predictions": dict(zip(ids, zero_preds)),
        },
        "few_shot": {
            "accuracy": results_k["accuracy"],
            "predictions": dict(zip(ids, few_preds)),
        },
        "ground_truth": dict(zip(ids, true_labels)),
    }

    out_path = PROCESSED_DATA_DIR / f"llama_baseline_results_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")