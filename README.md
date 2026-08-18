# YouTube Content Intelligence Platform

## Project Overview
This project is a complete, production-quality data science pipeline designed to analyze YouTube video performance. It collects data via the official YouTube Data API v3, cleans it, engineers features, trains machine learning models to predict video performance, and presents the results in an interactive Streamlit dashboard.

The overarching goal is to answer: *"What characteristics are associated with successful YouTube videos, and can we predict whether a new video is likely to perform well?"*

## Architecture & Data Pipeline
1. **Data Collection (`src/collector.py`)**: Fetches videos, statistics, and channel data. Handles API pagination and prevents redundant fetching.
2. **Data Cleaning (`src/cleaner.py`)**: Handles missing values, ISO 8601 duration parsing, type casting, and outlier detection.
3. **Feature Engineering (`src/feature_engineering.py`)**: Extracts predictive features (title characteristics, upload timing, channel size) and analytical features (engagement rates).
4. **Machine Learning (`src/train.py`)**: 
   - *Classification*: Predicts if a video is "High Performing" (top 25%).
   - *Regression*: Estimates `log_views`.
5. **Dashboard (`dashboard/app.py`)**: Interactive UI for exploration and prediction.
6. **Orchestration (`run.py`)**: CLI entry point supporting single-runs or continuous daemon schedules.

## Tech Stack
- **Python 3.10+**
- **Data Collection:** `google-api-python-client` (YouTube Data API v3)
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `joblib`
- **Visualization:** `plotly`
- **Dashboard:** `streamlit`
- **Automation:** `schedule`

## Setup Instructions

### 1. Clone & Environment
```bash
git clone <repository_url>
cd youtube-content-intelligence
python -m venv .venv
# Activate: 
# Windows: .\.venv\Scripts\Activate.ps1
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2. API Configuration
1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project and enable the **YouTube Data API v3**.
3. Create API Credentials (API Key).
4. Rename `.env.example` to `.env` and add your key:
```
YOUTUBE_API_KEY=your_actual_api_key_here
```
*Note: Never commit your `.env` file to version control.*

### 3. Usage
Run the pipeline to collect, clean, engineer, and train:
```bash
python run.py --collect
```

Launch the Streamlit dashboard:
```bash
python run.py --dashboard
```

Run as a continuous background daemon (e.g., every 24 hours):
```bash
python run.py --daemon 24
```

Run unit tests:
```bash
pytest tests/
```

## Machine Learning & Data Leakage Prevention
**CRITICAL:** The models are built specifically to prevent **data leakage**.
- **Prediction Features:** Factors known *at or before* publication (e.g., title length, upload day/hour, video duration, channel subscriber count).
- **Analysis Features:** Factors known only *after* publication (e.g., likes, comments, views, engagement rate). These are used in the EDA dashboard but **strictly excluded** from the ML models.

## Limitations & Ethical Considerations
- **API Quota:** The YouTube API has a daily quota (typically 10,000 units). The collector is optimized to batch requests, but large-scale frequent updates will exhaust the quota.
- **Algorithm Bias:** Models are trained on a snapshot of historical data. YouTube's recommendation algorithms evolve constantly, making view prediction inherently probabilistic, not deterministic.
- **Correlation vs. Causation:** Feature importance (e.g., title length) indicates correlation with success in this dataset, not a guaranteed causal mechanism.

## Future Improvements
- **NLP & Sentiment Analysis:** Generate embeddings from video descriptions and titles.
- **Thumbnail Analysis:** Extract visual features from video thumbnails.
- **Time-Series Forecasting:** Track video performance metrics continuously over time.
- **Database Migration:** Move from flat CSVs to PostgreSQL for the raw and processed data layers.
