# CS175 Market Sentiment Analysis using NLP

This project uses Natural Language Processing (NLP) to predict market sentiment based on business articles from The Guardian.

## Code Details

### Libraries
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

### Code Entirely Written by Us

#### Core Source Files (`/src/`)
- `dashboard.py` - Web GUI using Streamlit for interactive sentiment analysis (20 lines)
- `data_collection.py` - Collects business articles from The Guardian API (109 lines)
- `gpt_label.py` - Requests sentiment labels from ZotGPT API (166 lines)
- `stock_data_collection.py` - Retrieves stock market data using yfinance (33 lines)

#### Model Implementations (`/src/models/`)
- `baseline.py` - Orchestrates baseline model evaluation pipeline (47 lines)
- `logistic_regression.py` - TF-IDF + Logistic Regression baseline model (37 lines)
- `finbert_baseline.py` - Pre-trained FinBERT zero-shot inference (32 lines)
- `finbert_tuned.py` - Fine-tuned FinBERT for 5-class sentiment classification (107 lines)

#### Utility Scripts (`/src/scripts/`)
- `model_tester.py` - Evaluation metrics (MAE, accuracy, classification report) (34 lines)
- `label_with_zotgpt.py` - Automated labeling using ZotGPT (390 lines)
- `llm_average_labels.py` - Aggregates multiple LLM label predictions (72 lines)
- `merge_labels.py` - Combines labels from different sources (73 lines)
- `sentiment_stock_correlation.py` - Analyzes correlation between sentiment and stock prices (124 lines)
- `separate_articles_by_type.py` - Categorizes articles by content type (96 lines)
- `compare_llama_finbert.py` - Compares Llama and FinBERT model performance (176 lines)
- `llama_baseline.py` - Llama model baseline implementation (227 lines)
- `test_zotgpt.py` - ZotGPT API integration testing (91 lines)
- `zotgpt_auto_label.py` - Automated ZotGPT labeling workflow (65 lines)

**Total Lines of Original Code: ~1,900+ lines**


## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd CS175-Market-Sentiment
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the Data

- Download the data from [https://drive.google.com/file/d/1g2rZhCMd6BJlYWuIUppel8YsZwC5Hnt0/view?usp=sharing]
- Place it in `data/raw/` directory

### 4. Start Application

```bash
python ./src/dashboard.py
```

## Project Structure

```
CS175-Market-Sentiment/
├── data/
│   ├── raw/              # Raw articles from API
│   └── processed/        # Processed and labeled data
├── src/
│   ├── models/           # Model implementations
│   ├── scripts/          # Utility scripts for analysis and testing
│   ├── dashboard.py      # Streamlit web application
│   ├── data_collection.py
│   ├── gpt_label.py
│   └── stock_data_collection.py
├── results/              # Evaluation results
├── logs/                 # Training logs
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
└── README.md            # This file
```

## Usage

### Data Collection

```bash
python src/data_collection.py
```
