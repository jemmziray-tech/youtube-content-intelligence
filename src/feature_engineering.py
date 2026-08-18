import pandas as pd
import numpy as np
import logging
from .config import PROCESSED_DATA_DIR, HIGH_PERFORMING_PERCENTILE

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.clean_data_path = PROCESSED_DATA_DIR / "youtube_clean.csv"
        self.features_data_path = PROCESSED_DATA_DIR / "youtube_features.csv"

    def _safe_divide(self, a, b):
        return np.where(b == 0, 0, a / b)

    def engineer_features(self) -> pd.DataFrame:
        """Create features and save to new dataset."""
        logger.info(f"Loading cleaned data from {self.clean_data_path}")
        if not self.clean_data_path.exists():
            logger.error("Clean data file not found!")
            return pd.DataFrame()

        df = pd.read_csv(self.clean_data_path)
        
        # Ensure correct types
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['title'] = df['title'].fillna("").astype(str)
        df['description'] = df['description'].fillna("").astype(str)

        # 1. Title Features
        logger.info("Extracting Title features...")
        df['title_length'] = df['title'].apply(len)
        df['title_word_count'] = df['title'].apply(lambda x: len(x.split()))
        df['title_character_count'] = df['title'].apply(lambda x: len(x.replace(" ", "")))
        
        df['uppercase_ratio'] = df['title'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1)
        )
        df['exclamation_count'] = df['title'].apply(lambda x: x.count('!'))
        df['question_count'] = df['title'].apply(lambda x: x.count('?'))
        df['number_count'] = df['title'].apply(lambda x: sum(c.isdigit() for c in x))

        # 2. Description Features
        logger.info("Extracting Description features...")
        df['description_length'] = df['description'].apply(len)
        df['description_word_count'] = df['description'].apply(lambda x: len(x.split()))

        # 3. Publishing Features
        logger.info("Extracting Publishing features...")
        df['upload_year'] = df['published_at'].dt.year
        df['upload_month'] = df['published_at'].dt.month
        df['upload_day'] = df['published_at'].dt.day
        df['upload_day_of_week'] = df['published_at'].dt.dayofweek
        df['upload_hour'] = df['published_at'].dt.hour

        # 4. Engagement Features (ANALYSIS ONLY - DO NOT USE FOR PREDICTION)
        logger.info("Extracting Engagement features...")
        df['likes_per_view'] = self._safe_divide(df['likes'], df['views'])
        df['comments_per_view'] = self._safe_divide(df['comments'], df['views'])
        # Simplified engagement rate: (likes + comments) / views
        df['engagement_rate'] = self._safe_divide((df['likes'] + df['comments']), df['views'])

        # 5. Target Variables
        logger.info("Extracting Target variables...")
        # Classification Target: high_performing
        threshold = np.percentile(df['views'], HIGH_PERFORMING_PERCENTILE)
        df['high_performing'] = (df['views'] >= threshold).astype(int)
        
        # Regression Target: log_views
        df['log_views'] = np.log1p(df['views'])
        
        logger.info(f"Target 'high_performing' threshold (top {100-HIGH_PERFORMING_PERCENTILE}%): {threshold} views")
        logger.info(f"High performing count: {df['high_performing'].sum()} / {len(df)}")

        # Save features
        df.to_csv(self.features_data_path, index=False)
        logger.info(f"Saved engineered features to {self.features_data_path}")
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engineer = FeatureEngineer()
    engineer.engineer_features()
