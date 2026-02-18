# CS175 Market Sentiment Analysis using NLP

This project uses Natural Language Processing (NLP) to predict market sentiment based on business articles from The Guardian.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd CS175-Market-Sentiment
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get the Data
- Download the data from [https://drive.google.com/file/d/1g2rZhCMd6BJlYWuIUppel8YsZwC5Hnt0/view?usp=sharing]
- Place it in `data/raw/` directory

## Project Structure
```
CS175-Market-Sentiment/
├── data/
│   ├── raw/              # Raw articles from API
│   ├── processed/        # Processed and labeled data
│   └── README.md
├── src/
│   ├── data_collection.py      # Collect articles from Guardian API
│   ├── data_preprocessing.py   # Preprocess and prepare data
│   └── stock_data_collection.py # Collect stock market data from Yahoo Finance API
├── models/               # Trained model checkpoints
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