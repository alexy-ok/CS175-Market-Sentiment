import json
import os
import random
import pandas as pd
from transformers import pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# --- 1. Load the JSON file ---
'''
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct absolute path to JSON file
file_path = os.path.join(script_dir, "..", "raw", "guardian_articles_20260210_224419.json")

# Normalize path
file_path = os.path.abspath(file_path)
with open(file_path, "r", encoding="utf-8") as f:
    articles = json.load(f)
''' 
file_path = 'C:\\Users\\robbi\\source\\repos\\CS175-Market-Sentiment\\data\\raw\\guardian_articles_20260210_224419.json'
with open(file_path, "r", encoding="utf-8") as f:
    articles = json.load(f)
# --- 2. Extract text for prediction ---
articles = random.sample(articles, 500)
texts = []
for art in articles:
    headline = art["fields"].get("headline", "")
    standfirst = art["fields"].get("standfirst", "")
    body = art["fields"].get("bodyText", "")
    
    full_text = headline + " " + standfirst + " " + body
    texts.append(full_text)

# --- 3. Initialize FinBERT sentiment classifier ---
finbert = pipeline("text-classification", model="ProsusAI/finbert")

# --- 4. Generate sentiment labels automatically ---
labels = []
for text in texts:
    result = finbert(
        text, 
        truncation=True,
        max_length=512
    )[0]  
    label = result["label"]  # 'POSITIVE', 'NEUTRAL', 'NEGATIVE'
    
    if label == "negative":
        labels.append(0)
    elif label == "neutral":
        labels.append(1)
    else:
        labels.append(2)

# --- 5. Create DataFrame ---
data = pd.DataFrame({
    "text": texts,
    "label": labels
})

print(data.head())

# --- 6. Train/validation/test split ---
X = data["text"]
y = data["label"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

# --- 7. TF-IDF vectorization ---
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# --- 8. Logistic Regression baseline ---
baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
print(data["label"].value_counts())
baseline_model.fit(X_train_vec, y_train)

# --- 9. Evaluate ---
val_preds = baseline_model.predict(X_val_vec)
print("Validation Accuracy:", accuracy_score(y_val, val_preds))
print(classification_report(y_val, val_preds))

test_preds = baseline_model.predict(X_test_vec)
print("Test Accuracy:", accuracy_score(y_test, test_preds))
print(classification_report(y_test, test_preds))
