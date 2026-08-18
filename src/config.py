import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Search configuration
SEARCH_QUERIES = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "generative AI",
    "data science",
    "Python programming",
    "large language models",
    "AI tools",
    "computer science"
]

# ML configuration
HIGH_PERFORMING_PERCENTILE = 75 # Top 25%

# Ensure directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
