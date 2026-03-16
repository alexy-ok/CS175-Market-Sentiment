import json
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.finbert_baseline import run_finbert
from src.scripts.model_tester import evaluate_model
from transformers import pipeline

# mapping for LLaMA string predictions to numeric
llama_mapping = {
    "negative": 0,
    "leaning negative": 1,
    "neutral": 2,
    "leaning positive": 3,
    "positive": 4
}

def run_finbert_get_preds(texts):
    """Modified version that returns predictions"""
    finbert = pipeline(
        "text-classification",
        model="ProsusAI/finbert"
    )
    y_pred = []
    for text in texts:
        if not text.strip():
            y_pred.append(2)  # neutral for empty
            continue
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
    return y_pred

def load_raw_articles():
    """Load all raw Guardian articles into a dict by ID"""
    raw_dir = Path('data/raw')
    articles = {}
    for file_path in raw_dir.glob('guardian_articles_*.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            for article in data:
                article_id = article.get('id')
                if article_id:
                    articles[article_id] = article
    return articles

def plot_model_comparison(llama_preds, finbert_preds, y_true, title_prefix, save_path):
    llama_numeric = pd.Series(llama_preds)
    finbert_series = pd.Series(finbert_preds)
    
    print("\nPrediction Distributions:")
    print(f"LLaMA {title_prefix} predictions: {llama_numeric.value_counts().sort_index()}")
    print(f"FinBERT predictions: {finbert_series.value_counts().sort_index()}")
    
    # agreement
    agreement = (llama_numeric == finbert_series).mean()
    print(f"Agreement between LLaMA {title_prefix} and FinBERT: {agreement:.3f}")
    
    # correlation
    corr = llama_numeric.corr(finbert_series)
    print(f"Correlation between LLaMA {title_prefix} and FinBERT predictions: {corr:.3f}")

    # compute accuracies
    llama_acc = accuracy_score(y_true, llama_preds)
    finbert_acc = accuracy_score(y_true, finbert_preds)
    
    # plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # prediction distributions
    llama_counts = llama_numeric.value_counts().sort_index()
    finbert_counts = finbert_series.value_counts().sort_index()
    all_scores = sorted(set(llama_counts.index).union(set(finbert_counts.index)))
    
    llama_vals = [llama_counts.get(score, 0) for score in all_scores]
    finbert_vals = [finbert_counts.get(score, 0) for score in all_scores]
    
    axes[0].bar([x - 0.2 for x in all_scores], llama_vals, width=0.4, label='LLaMA', alpha=0.7, color='blue')
    axes[0].bar([x + 0.2 for x in all_scores], finbert_vals, width=0.4, label='FinBERT', alpha=0.7, color='orange')
    axes[0].set_title('Prediction Distributions')
    axes[0].set_xlabel('Sentiment Score (0-4)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(all_scores)
    axes[0].legend()
    
    # scatter plot
    axes[1].scatter(llama_numeric, finbert_series, alpha=0.6, color='green')
    axes[1].set_title(f'LLaMA {title_prefix} vs FinBERT Predictions\nCorrelation: {corr:.3f}')
    axes[1].set_xlabel(f'LLaMA {title_prefix} Score')
    axes[1].set_ylabel('FinBERT Score')
    axes[1].set_xlim(-0.5, 4.5)
    axes[1].set_ylim(-0.5, 4.5)
    axes[1].grid(True, alpha=0.3)
    
    # accuracy comparison
    models = [f'LLaMA {title_prefix}', 'FinBERT']
    accuracies = [llama_acc, finbert_acc]
    bars = axes[2].bar(models, accuracies, color=['blue', 'orange'])
    axes[2].set_title('Model Accuracies')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def main():
    # load LLaMA results
    with open('data/processed/llama_baseline_results.json', 'r') as f:
        llama_results = json.load(f)
    
    # load true labels
    with open('data/processed/hand_labels_for_few_shot.json', 'r') as f:
        true_labels = json.load(f)
    
    # load raw articles
    articles = load_raw_articles()
    
    # prepare data for comparison
    llama_zero_preds = []
    llama_few_preds = []
    finbert_texts = []
    y_true = []

    # load prediction sets
    zero_shot_preds = llama_results['zero_shot']['predictions']
    few_shot_preds = llama_results['few_shot']['predictions']
    
    for article_id in zero_shot_preds:
        if article_id in true_labels and article_id in articles:
            y_true.append(true_labels[article_id])
            
            llama_zero_preds.append(zero_shot_preds[article_id])
            llama_few_preds.append(few_shot_preds[article_id])
            
            # get text for FinBERT
            article = articles[article_id]
            body_text = article.get('fields', {}).get('bodyText', '')
            if body_text:
                finbert_texts.append(body_text[:2000])  # limit length
            else:
                finbert_texts.append('')
    
    print(f"Found {len(y_true)} articles with complete data")
    
    # evaluate LLaMA
    evaluate_model("LLaMA Zero-Shot", y_true, llama_zero_preds)
    evaluate_model("LLaMA Few-Shot", y_true, llama_few_preds)
    
    # run FinBERT
    finbert_preds = []
    if finbert_texts:
        finbert_preds = run_finbert_get_preds(finbert_texts)
        evaluate_model("FinBERT Baseline", y_true, finbert_preds)
    
    plot_model_comparison(llama_zero_preds, finbert_preds, y_true,
                          title_prefix="0-shot", save_path="results/0-shot_model_comparison.png")
    plot_model_comparison(llama_few_preds, finbert_preds, y_true,
                          title_prefix="Few-shot", save_path="results/few-shot_model_comparison.png")

if __name__ == "__main__":
    main()