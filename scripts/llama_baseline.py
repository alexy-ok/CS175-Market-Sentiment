import json, random, sys, datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.metrics import accuracy_score, classification_report

# mac-config
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_FILE = BASE_DIR / "data" / "raw" / "guardian_articles_20260315_054452.json"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# config
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
BATCH_SIZE = 8

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
    "You are a financial news sentiment classifier."
    "Given a news headline, respond with exactly one of these labels:"
    "Positive, Leaning Positive, Neutral, Leaning Negative, Negative."
    "Do not provide an explanation, just give the label."
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
        str_label = INT_TO_LABEL[int_label].title().replace(" N", " N")
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
        torch_dtype=torch.float32, # change to float16 for MPS (Apple silicon gpu)
        device="cpu",
        do_sample=False,
        max_new_tokens=8,
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
    text = raw.strip().lower()

    for label in sorted(LABELS, key=len, reverse=True):
        if label in text:
            return label
        
    return "unknown"

def predict_batch(pipe, article_texts, mode):
    all_messages = []
    for text in article_texts:
        if mode == "zero_shot":
            msgs = build_zero_shot_msgs(text)
        else:
            msgs = build_few_shot_msgs(text)
        all_messages.append(msgs)

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
    acc    = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, labels=LABELS, zero_division=0)
    return {"accuracy": acc, "report": report}

# dummy data
DUMMY_ARTICLES = [
    {"id": "test/positive-1",        "type": "article", "webTitle": "Markets surge on blowout jobs report",           "fields": {"bodyText": "Wall Street surged Friday after the Labor Department reported 350,000 jobs added, far exceeding expectations. The S&P 500 climbed 2.1%, led by financials and consumer discretionary stocks. Unemployment fell to 3.4%."}},
    {"id": "test/leaning-positive-1","type": "article", "webTitle": "Stocks edge higher after dovish Fed comments",    "fields": {"bodyText": "Equity markets drifted slightly higher Thursday after Fed minutes suggested policymakers are open to pausing rate hikes. Gains were modest, with the Dow up 0.4%, as investors remained cautious."}},
    {"id": "test/neutral-1",         "type": "article", "webTitle": "Fed holds rates steady, offers no new guidance", "fields": {"bodyText": "The Federal Reserve held its benchmark rate steady at 5.25-5.5% on Wednesday. Chair Powell said the committee would remain data-dependent and gave no clear signal on future cuts or hikes."}},
    {"id": "test/leaning-negative-1","type": "article", "webTitle": "Consumer spending misses forecasts",             "fields": {"bodyText": "Consumer spending rose just 0.1% in October, below the 0.3% forecast, as higher borrowing costs weighed on households. Analysts warned the soft reading could drag on fourth-quarter GDP."}},
    {"id": "test/negative-1",        "type": "article", "webTitle": "Markets plunge on recession fears",              "fields": {"bodyText": "Stocks tumbled sharply Friday as weak manufacturing data stoked recession fears. The Nasdaq fell 3.5%, the S&P 500 dropped 2.8%, and bond yields spiked to their highest since 2008. Analysts warned of further downside."}},
    {"id": "test/positive-2",        "type": "article", "webTitle": "Tech stocks rally on strong earnings",           "fields": {"bodyText": "Technology shares led a broad market rally after several major firms reported better-than-expected quarterly earnings. Investors cheered the results, pushing the Nasdaq up 1.8% on the session."}},
    {"id": "test/neutral-2",         "type": "article", "webTitle": "Oil prices steady amid mixed signals",           "fields": {"bodyText": "Crude oil prices held steady Thursday as OPEC output data and US inventory figures sent conflicting signals to traders. Analysts said the market was in a wait-and-see mode ahead of next week's Fed meeting."}},
    {"id": "test/negative-2",        "type": "article", "webTitle": "Bank shares fall on credit concern warnings",    "fields": {"bodyText": "Financial sector stocks slid after several large banks flagged rising credit card delinquencies and tightening lending standards. The KBW Bank Index fell 2.1%, with analysts cutting price targets across the sector."}},
    {"id": "test/leaning-negative-2","type": "article", "webTitle": "Retail sales dip, missing analyst forecasts",   "fields": {"bodyText": "US retail sales fell 0.2% in September, slightly below the flat reading economists had forecast. While the labour market remains resilient, the data added to concerns about slowing consumer momentum heading into the holiday season."}},
    {"id": "test/leaning-positive-2","type": "article", "webTitle": "Inflation cools slightly, boosting rate-cut hopes", "fields": {"bodyText": "Consumer prices rose 3.1% year-on-year in November, down from 3.2% the prior month. The modest decline raised hopes that the Fed could begin cutting rates in the first half of next year, sending stocks modestly higher."}},
]

DUMMY_LABELS = {
    "test/positive-1":         4,
    "test/leaning-positive-1": 3,
    "test/neutral-1":          2,
    "test/leaning-negative-1": 1,
    "test/negative-1":         0,
    "test/positive-2":         4,
    "test/neutral-2":          2,
    "test/negative-2":         0,
    "test/leaning-negative-2": 1,
    "test/leaning-positive-2": 3,
}

if __name__ == "__main__":
    test_mode = "--test" in sys.argv

    if test_mode:
        print("=" * 80)
        print("RUNNING IN TEST MODE (dummy data, no API key or files needed)")
        print("=" * 80)
        articles      = DUMMY_ARTICLES
        zotgpt_labels = DUMMY_LABELS
    else:
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