Libraries
- torch
- transformers
- datasets
- scikit-learn
- numpy
- pandas
- requests
- python-dotenv
- matplotlib
- tqdm
- streamlit
- yfinance

Code Entirely Written by Us
Core Source Files (/src/)
- `dashboard.py` - Web GUI using Streamlit for interactive sentiment analysis (20 lines)
- `data_collection.py` - Collects business articles from The Guardian API (109 lines)
- `gpt_label.py` - Requests sentiment labels from ZotGPT API (166 lines)
- `stock_data_collection.py` - Retrieves stock market data using yfinance (33 lines)

Model Implementations (/models/)
- `baseline.py` - Orchestrates baseline model evaluation pipeline (44 lines)
- `logistic_regression.py` - TF-IDF + Logistic Regression baseline model (37 lines)
- `finbert_baseline.py` - Pre-trained FinBERT zero-shot inference (32 lines)
- `finbert_tuned.py` - Fine-tuned FinBERT for 5-class sentiment classification (107 lines)

Utility Scripts (/scripts/)
- `model_tester.py` - Evaluation metrics (MAE, accuracy, classification report) (34 lines)
- `plot.py` - Visualization of classification results (55 lines)
- `label_with_zotgpt.py` - Automated labeling using ZotGPT (390 lines)
- `llm_average_labels.py` - Aggregates multiple LLM label predictions (72 lines)
- `merge_labels.py` - Combines labels from different sources (73 lines)
- `sentiment_stock_correlation.py` - Analyzes correlation between sentiment and stock prices (124 lines)
- `separate_articles_by_type.py` - Categorizes articles by content type (96 lines)
- `compare_llama_finbert.py` - Compares Llama and FinBERT model performance (176 lines)
- `llama_baseline.py` - Llama model baseline implementation (227 lines)
- `test_zotgpt.py` - ZotGPT API integration testing (91 lines)
- `zotgpt_auto_label.py` - Automated ZotGPT labeling workflow (65 lines)
