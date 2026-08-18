import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import PROCESSED_DATA_DIR, MODELS_DIR

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.features_data_path = PROCESSED_DATA_DIR / "youtube_features.csv"
        
        # VERY IMPORTANT: PREDICTION FEATURES ONLY. 
        # No post-publication stats (views, likes, comments, engagement)
        self.prediction_features = [
            'title_length', 'title_word_count', 'title_character_count', 
            'uppercase_ratio', 'exclamation_count', 'question_count', 'number_count',
            'description_length', 'description_word_count',
            'duration_minutes', 
            'upload_year', 'upload_month', 'upload_day', 'upload_day_of_week', 'upload_hour',
            'subscriber_count', 'channel_video_count'
        ]
        
    def load_data(self):
        df = pd.read_csv(self.features_data_path)
        # Drop rows with NaN in target variables (just in case)
        df = df.dropna(subset=['high_performing', 'log_views'])
        return df

    def create_preprocessor(self):
        """Creates a preprocessing pipeline for numeric features."""
        # We use RobustScaler because YouTube data (like subscriber_count) has heavy outliers
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.prediction_features)
            ])
        return preprocessor

    def train_classifiers(self, X_train, y_train, X_test, y_test):
        """Train and compare classification models."""
        logger.info("Training Classification Models...")
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        results = []
        best_f1 = -1
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.create_preprocessor()),
                ('classifier', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            # For ROC-AUC, we need probabilities if available
            if hasattr(model, "predict_proba"):
                y_prob = pipeline.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, y_prob)
            else:
                roc = np.nan
                
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1-score": f1,
                "ROC-AUC": roc
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline
                best_name = name
                
        results_df = pd.DataFrame(results)
        logger.info(f"\nClassification Results:\n{results_df.to_string()}")
        logger.info(f"Best Classifier: {best_name}")
        
        joblib.dump(best_model, MODELS_DIR / "best_classifier.joblib")
        logger.info("Saved best classifier to models/best_classifier.joblib")
        
        # Save evaluation metrics to use in dashboard
        results_df.to_csv(MODELS_DIR / "classification_metrics.csv", index=False)
        
        # Save feature importance if possible
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            importances = best_model.named_steps['classifier'].feature_importances_
            imp_df = pd.DataFrame({
                "Feature": self.prediction_features,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)
            imp_df.to_csv(MODELS_DIR / "classifier_feature_importance.csv", index=False)

    def train_regressors(self, X_train, y_train, X_test, y_test):
        """Train and compare regression models."""
        logger.info("Training Regression Models...")
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        results = []
        best_r2 = -float("inf")
        best_model = None
        best_name = ""
        
        for name, model in models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', self.create_preprocessor()),
                ('regressor', model)
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2
            })
            
            if r2 > best_r2:
                best_r2 = r2
                best_model = pipeline
                best_name = name
                
        results_df = pd.DataFrame(results)
        logger.info(f"\nRegression Results:\n{results_df.to_string()}")
        logger.info(f"Best Regressor: {best_name}")
        
        joblib.dump(best_model, MODELS_DIR / "best_regressor.joblib")
        logger.info("Saved best regressor to models/best_regressor.joblib")
        
        results_df.to_csv(MODELS_DIR / "regression_metrics.csv", index=False)
        
    def run(self):
        df = self.load_data()
        
        X = df[self.prediction_features]
        y_class = df['high_performing']
        y_reg = df['log_views']
        
        # Split for Classification
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X, y_class, test_size=0.2, random_state=42, stratify=y_class
        )
        
        self.train_classifiers(X_train_c, y_train_c, X_test_c, y_test_c)
        
        # Split for Regression
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        self.train_regressors(X_train_r, y_train_r, X_test_r, y_test_r)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = ModelTrainer()
    trainer.run()
