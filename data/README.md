# Data Directory

This directory contains the datasets for the CS175 Market Sentiment Analysis project.

## Directory Structure

```
data/
├── raw/                    # Raw articles from The Guardian API
│   └── .gitkeep
├── processed/              # Processed and labeled data
│   └── .gitkeep
└── README.md              # This file
```

## Getting the Data

1. Download the shared data file from [https://drive.google.com/file/d/1g2rZhCMd6BJlYWuIUppel8YsZwC5Hnt0/view?usp=sharing]
2. Place it in the `data/raw/` directory
3. The file should be named: `guardian_articles_YYYYMMDD_HHMMSS.json`

## Raw Data Format

The raw data file contains a JSON array of articles from The Guardian API with the following structure:

```json
[
  {
    "id": "business/2024/...",
    "webTitle": "Article Headline",
    "webPublicationDate": "2024-01-01T12:00:00Z",
    "webUrl": "https://...",
    "fields": {
      "headline": "Article Headline",
      "bodyText": "Full article text...",
      "standfirst": "Article summary..."
    }
  },
  ...
]
```

## Processed Data Format

After running `data_preprocessing.py`, you'll have:
- `articles_for_labeling.csv` - Cleaned articles ready for sentiment labeling
- `articles_labeled.csv` - Manually labeled data (after team labeling)
- `train.csv` - Training set
- `val.csv` - Validation set
- `test.csv` - Test set

## Data Collection Parameters

Current settings (see `src/data_collection.py`):
- **Source:** The Guardian API
- **Section:** Business
- **Tags:** business/stock-markets
- **Date Range:** Last 2 years
- **Max Articles:** 2000
- **Fields:** headline, bodyText, standfirst