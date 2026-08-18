import pandas as pd
import numpy as np
import logging
import isodate
from pathlib import Path

from .config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self):
        self.raw_data_path = RAW_DATA_DIR / "youtube_raw.csv"
        self.clean_data_path = PROCESSED_DATA_DIR / "youtube_clean.csv"

    def _parse_duration(self, duration_str):
        """Convert ISO 8601 duration string to total seconds."""
        if pd.isna(duration_str):
            return np.nan
        try:
            parsed_duration = isodate.parse_duration(duration_str)
            return parsed_duration.total_seconds()
        except Exception:
            return np.nan

    def clean(self) -> pd.DataFrame:
        """Load, clean, and save the dataset."""
        logger.info(f"Loading raw data from {self.raw_data_path}")
        if not self.raw_data_path.exists():
            logger.error("Raw data file not found!")
            return pd.DataFrame()

        df = pd.read_csv(self.raw_data_path)
        initial_len = len(df)

        # 1. Deduplication
        df = df.drop_duplicates(subset=["video_id"], keep="last")
        logger.info(f"Dropped {initial_len - len(df)} duplicate videos.")

        # 2. Handle missing values
        # Fill missing text fields with empty strings
        df['title'] = df['title'].fillna("")
        df['description'] = df['description'].fillna("")
        df['tags'] = df['tags'].fillna("")
        
        # Numeric conversions and filling missing stats
        numeric_cols = ["views", "likes", "comments", "subscriber_count", "channel_video_count", "channel_view_count"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

        # 3. Handle Dates
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df = df.dropna(subset=['published_at']) # Drop if publish date is missing entirely

        # 4. Handle Duration
        df['duration_seconds'] = df['duration'].apply(self._parse_duration)
        df['duration_minutes'] = df['duration_seconds'] / 60.0
        
        # Drop malformed durations
        df = df.dropna(subset=['duration_seconds'])
        
        # Filter out 0 length or unreasonably long (e.g. > 10 hours) streams which might distort analysis
        df = df[(df['duration_seconds'] > 0) & (df['duration_seconds'] <= 36000)]

        # 5. Handle outliers / Zero views
        # Videos with zero views are kept, but we might log them.
        zero_views = len(df[df['views'] == 0])
        if zero_views > 0:
            logger.warning(f"Found {zero_views} videos with 0 views.")

        # Remove "extreme" outliers only if they're clearly errors. 
        # In YouTube, extreme views (e.g. billions) are valid, so we don't drop them blindly.

        logger.info(f"Final dataset size after cleaning: {len(df)}")
        
        # Save to processed directory
        df.to_csv(self.clean_data_path, index=False)
        logger.info(f"Saved cleaned data to {self.clean_data_path}")
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaner = DataCleaner()
    cleaner.clean()
