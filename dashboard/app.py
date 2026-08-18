import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
import subprocess

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.config import PROCESSED_DATA_DIR, MODELS_DIR
from src.eda import (
    plot_views_distribution, 
    plot_views_vs_duration, 
    plot_performance_by_day, 
    plot_top_channels,
    plot_correlation_matrix,
    plot_performance_donut,
    plot_prediction_gauge
)

# --- Page Configuration ---
st.set_page_config(
    page_title="YouTube Content Intelligence",
    page_icon="▶️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Advanced Custom CSS (Glassmorphism & Polish) ---
st.markdown("""
<style>
    /* Global Background tweaking for deep dark mode */
    .stApp {
        background-color: #0e1117;
    }
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(30, 30, 30, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        margin-bottom: 24px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(255, 0, 0, 0.15);
        border: 1px solid rgba(255, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #ff0000;
        margin-bottom: 8px;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 1rem;
        color: #aaaaaa;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #ff0000;
        padding-bottom: 0.5rem;
        display: inline-block;
    }
    /* Hide top padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    features_path = PROCESSED_DATA_DIR / "youtube_features.csv"
    if features_path.exists():
        return pd.read_csv(features_path)
    return pd.DataFrame()

@st.cache_resource
def load_models():
    classifier_path = MODELS_DIR / "best_classifier.joblib"
    regressor_path = MODELS_DIR / "best_regressor.joblib"
    classifier = joblib.load(classifier_path) if classifier_path.exists() else None
    regressor = joblib.load(regressor_path) if regressor_path.exists() else None
    return classifier, regressor

@st.cache_data
def load_metrics():
    class_metrics = MODELS_DIR / "classification_metrics.csv"
    reg_metrics = MODELS_DIR / "regression_metrics.csv"
    feat_imp = MODELS_DIR / "classifier_feature_importance.csv"
    
    cm = pd.read_csv(class_metrics) if class_metrics.exists() else pd.DataFrame()
    rm = pd.read_csv(reg_metrics) if reg_metrics.exists() else pd.DataFrame()
    fi = pd.read_csv(feat_imp) if feat_imp.exists() else pd.DataFrame()
    
    return cm, rm, fi

df = load_data()
classifier, regressor = load_models()
class_metrics, reg_metrics, feat_imp = load_metrics()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg", width=150)
    st.markdown("<br>", unsafe_allow_html=True)
    
    page = st.radio(
        "Navigation", 
        ["Overview", "Video Explorer", "Performance Analysis", "Channel Analysis", "ML Prediction", "Model Performance", "Data Refresh", "About"]
    )
    
    st.markdown("---")
    st.markdown("### Portfolio Project")
    st.markdown("Built by an expert AI Engineer.")
    st.markdown("Powered by Streamlit & Scikit-Learn.")

if df.empty and page not in ["Data Refresh", "About"]:
    st.warning("No data found! Please go to 'Data Refresh' to run the initial data collection.")
    st.stop()

# --- Page: Overview ---
if page == "Overview":
    st.title("Executive Overview")
    st.markdown("A high-level view of the collected YouTube dataset, focusing on AI and Tech content.")
    
    # Top Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{len(df):,}</div><div class='metric-label'>Total Videos</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{df['channel_title'].nunique():,}</div><div class='metric-label'>Channels</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{int(df['views'].median()):,}</div><div class='metric-label'>Median Views</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-value'>{int(df['engagement_rate'].mean() * 100):.2f}%</div><div class='metric-label'>Avg Engagement</div></div>", unsafe_allow_html=True)

    # Main Charts
    st.markdown("<div class='section-header'>Data Distributions</div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.plotly_chart(plot_views_distribution(df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_performance_donut(df), use_container_width=True)

# --- Page: Video Explorer ---
elif page == "Video Explorer":
    st.title("Dataset Explorer")
    st.markdown("Deep dive into the raw data.")
    
    # Filters in an expander for clean UI
    with st.expander("🔍 Search & Filter", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            channels = ["All"] + sorted(df["channel_title"].unique().tolist())
            selected_channel = st.selectbox("Filter by Channel", channels)
        with col2:
            search_kw = st.text_input("Search Title/Description Keyword")
            
    filtered_df = df.copy()
    if selected_channel != "All":
        filtered_df = filtered_df[filtered_df["channel_title"] == selected_channel]
    if search_kw:
        filtered_df = filtered_df[filtered_df["title"].str.contains(search_kw, case=False, na=False) | 
                                  filtered_df["description"].str.contains(search_kw, case=False, na=False)]
        
    display_cols = ["title", "channel_title", "views", "likes", "engagement_rate", "duration_minutes", "published_at"]
    
    # Use st.dataframe with column config for polished tables
    st.dataframe(
        filtered_df[display_cols].sort_values("views", ascending=False),
        column_config={
            "views": st.column_config.ProgressColumn("Views", format="%d", min_value=0, max_value=int(filtered_df['views'].max())),
            "engagement_rate": st.column_config.NumberColumn("Engagement", format="%.4f"),
            "duration_minutes": st.column_config.NumberColumn("Duration (m)", format="%.1f")
        },
        use_container_width=True,
        hide_index=True
    )

# --- Page: Performance Analysis ---
elif page == "Performance Analysis":
    st.title("Performance Analytics")
    
    st.markdown("<div class='section-header'>Duration vs Views</div>", unsafe_allow_html=True)
    st.plotly_chart(plot_views_vs_duration(df), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Timing Impact</div>", unsafe_allow_html=True)
        st.plotly_chart(plot_performance_by_day(df), use_container_width=True)
    with col2:
        st.markdown("<div class='section-header'>Feature Correlation</div>", unsafe_allow_html=True)
        features = ["log_views", "duration_minutes", "title_length", "subscriber_count", "engagement_rate"]
        st.plotly_chart(plot_correlation_matrix(df, features), use_container_width=True)

# --- Page: Channel Analysis ---
elif page == "Channel Analysis":
    st.title("Channel Analytics")
    st.markdown("Discover which channels are driving the highest engagement and views.")
    
    st.plotly_chart(plot_top_channels(df), use_container_width=True)
    
    st.markdown("<div class='section-header'>Detailed Channel Statistics</div>", unsafe_allow_html=True)
    channel_agg = df.groupby("channel_title").agg(
        videos=("video_id", "count"),
        median_views=("views", "median"),
        avg_engagement=("engagement_rate", "mean"),
        high_perf_pct=("high_performing", lambda x: x.mean() * 100)
    ).sort_values("median_views", ascending=False).reset_index()
    
    st.dataframe(
        channel_agg,
        column_config={
            "median_views": st.column_config.ProgressColumn("Median Views", format="%d", min_value=0, max_value=int(channel_agg['median_views'].max())),
            "avg_engagement": st.column_config.NumberColumn("Avg Engagement", format="%.4f"),
            "high_perf_pct": st.column_config.NumberColumn("High Perf. (%)", format="%.1f")
        },
        use_container_width=True,
        hide_index=True
    )

# --- Page: ML Prediction ---
elif page == "ML Prediction":
    st.title("AI Video Performance Predictor")
    st.info("💡 **Strict Data Leakage Prevention:** This model predicts future performance using *only* pre-publication features (Title, Timing, Channel Size). Post-publication stats like Likes or Comments are explicitly excluded.")
    
    if not classifier or not regressor:
        st.error("Models not found! Please run the training pipeline first.")
        st.stop()
        
    with st.form("prediction_form"):
        st.markdown("<div class='section-header'>Video Metadata Input</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            title = st.text_input("📝 Video Title", "Building a YouTube ML Dashboard in Python")
            duration_min = st.number_input("⏱️ Duration (minutes)", min_value=1.0, value=15.0)
        with col2:
            subscribers = st.number_input("👥 Channel Subscribers", min_value=0, value=50000)
            channel_vids = st.number_input("🎥 Total Channel Videos", min_value=0, value=100)
        with col3:
            upload_day = st.selectbox("📅 Upload Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            upload_hour = st.slider("🕒 Upload Hour (24h)", 0, 23, 14)
            
        desc = st.text_area("📄 Video Description", "In this video we build a complete ML pipeline...")
            
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Run ML Prediction", use_container_width=True)
        
    if submitted:
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        # Engineer features
        input_data = {
            'title_length': len(title),
            'title_word_count': len(title.split()),
            'title_character_count': len(title.replace(" ", "")),
            'uppercase_ratio': sum(1 for c in title if c.isupper()) / max(len(title), 1),
            'exclamation_count': title.count('!'),
            'question_count': title.count('?'),
            'number_count': sum(c.isdigit() for c in title),
            'description_length': len(desc),
            'description_word_count': len(desc.split()),
            'duration_minutes': duration_min,
            'upload_year': 2024,
            'upload_month': 1,
            'upload_day': 15,
            'upload_day_of_week': day_map[upload_day],
            'upload_hour': upload_hour,
            'subscriber_count': subscribers,
            'channel_video_count': channel_vids
        }
        
        input_df = pd.DataFrame([input_data])
        
        prob_high = classifier.predict_proba(input_df)[0][1]
        log_views_pred = regressor.predict(input_df)[0]
        views_pred = np.expm1(log_views_pred)
        
        st.markdown("<div class='section-header'>Prediction Results</div>", unsafe_allow_html=True)
        
        r1, r2 = st.columns([1, 1])
        with r1:
            st.plotly_chart(plot_prediction_gauge(prob_high), use_container_width=True)
        with r2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{int(views_pred):,}</div><div class='metric-label'>Estimated Views</div></div>", unsafe_allow_html=True)
            if prob_high > 0.5:
                st.success("🎯 The model classifies this video as highly likely to perform well!")
            else:
                st.warning("📉 The model classifies this video as likely to have average or below-average performance.")

# --- Page: Model Performance ---
elif page == "Model Performance":
    st.title("Model Diagnostics")
    st.markdown("Deep dive into the underlying machine learning models and feature importances.")
    
    t1, t2 = st.tabs(["Classification (Top 25%)", "Regression (Log Views)"])
    
    with t1:
        st.markdown("### Classifier Evaluation")
        st.markdown("**Metric Focus:** ROC-AUC is prioritized over Accuracy due to class imbalance.")
        st.dataframe(
            class_metrics,
            column_config={
                "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                "F1-score": st.column_config.NumberColumn(format="%.3f"),
                "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
            },
            use_container_width=True, hide_index=True
        )
        
        if not feat_imp.empty:
            st.markdown("### Global Feature Importance")
            import plotly.express as px
            fig = px.bar(
                feat_imp.head(10).sort_values("Importance", ascending=True),
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=px.colors.sequential.Reds
            )
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
    with t2:
        st.markdown("### Regressor Evaluation")
        st.markdown("**Note:** Predicting exact views is extremely hard. The $R^2$ score indicates how much variance our pre-publication features explain.")
        st.dataframe(
            reg_metrics,
            column_config={
                "MAE": st.column_config.NumberColumn(format="%.3f"),
                "RMSE": st.column_config.NumberColumn(format="%.3f"),
                "R2": st.column_config.NumberColumn(format="%.3f"),
            },
            use_container_width=True, hide_index=True
        )

# --- Page: Data Refresh ---
elif page == "Data Refresh":
    st.title("Data Pipeline Orchestration")
    st.markdown("Trigger the entire ELT and ML pipeline directly from the UI.")
    
    st.info("The pipeline will: **Fetch Data (YouTube API)** ➔ **Clean Data** ➔ **Engineer Features** ➔ **Retrain Models**")
    
    if st.button("🚀 Run Full Data Pipeline Now", type="primary"):
        with st.spinner("Executing pipeline... This will take a few minutes. Check the terminal for detailed logs."):
            try:
                subprocess.run([sys.executable, "-m", "src.collector"], check=True)
                subprocess.run([sys.executable, "-m", "src.cleaner"], check=True)
                subprocess.run([sys.executable, "-m", "src.feature_engineering"], check=True)
                subprocess.run([sys.executable, "-m", "src.train"], check=True)
                
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Data pipeline completed successfully! Models and datasets have been updated.")
            except subprocess.CalledProcessError as e:
                st.error(f"❌ An error occurred during the pipeline run: {e}")

# --- Page: About ---
elif page == "About":
    st.title("About This Project")
    st.markdown("""
    ### YouTube Content Intelligence Platform
    This project is a complete, production-quality data science pipeline designed to analyze YouTube video performance.
    
    **Architecture:**
    - **Data Collection:** YouTube Data API v3 (Incremental updates, Pagination, Deduplication)
    - **Data Processing:** Pandas, NumPy (Cleaning, Feature Engineering)
    - **Machine Learning:** Scikit-Learn (Classification & Regression pipelines)
    - **Dashboard:** Streamlit & Plotly
    
    **Data Leakage Prevention:**
    A strict boundary is maintained between *analysis features* (likes, comments, views) and *prediction features* (title, duration, subscriber count). Models are trained solely on information available *before* publication.
    
    **Limitations:**
    - Models predict estimates based on historical samples; they cannot perfectly account for the YouTube recommendation algorithm.
    - API Quota limits the amount of data we can refresh daily.
    """)
