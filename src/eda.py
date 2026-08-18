import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Common template settings for a polished look
TEMPLATE = "plotly_dark"
CHART_MARGINS = dict(l=20, r=20, t=40, b=20)

def _clean_layout(fig: go.Figure) -> go.Figure:
    """Applies a clean, portfolio-ready layout to a Plotly figure."""
    fig.update_layout(
        template=TEMPLATE,
        margin=CHART_MARGINS,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=12, color="#e0e0e0"),
        hoverlabel=dict(bgcolor="#1e1e1e", font_size=13, font_family="sans-serif")
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)", zeroline=False)
    return fig

def plot_performance_donut(df: pd.DataFrame) -> go.Figure:
    """Donut chart showing High Performing vs Normal videos."""
    counts = df['high_performing'].value_counts().reset_index()
    counts['high_performing'] = counts['high_performing'].map({1: 'High Performing (Top 25%)', 0: 'Normal'})
    
    fig = px.pie(
        counts, 
        values='count', 
        names='high_performing',
        hole=0.6,
        color='high_performing',
        color_discrete_map={'High Performing (Top 25%)': '#ff0000', 'Normal': '#333333'}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
    fig.update_layout(title_text="Performance Split", title_x=0.5)
    return _clean_layout(fig)

def plot_prediction_gauge(probability: float) -> go.Figure:
    """Gauge chart for ML probability."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "High Performance Probability (%)"},
        number = {'suffix': "%", 'font': {'size': 40, 'color': '#ff0000'}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#ff0000"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#333",
            'steps': [
                {'range': [0, 25], 'color': "rgba(255,0,0,0.1)"},
                {'range': [25, 50], 'color': "rgba(255,0,0,0.25)"},
                {'range': [50, 75], 'color': "rgba(255,0,0,0.5)"},
                {'range': [75, 100], 'color': "rgba(255,0,0,0.75)"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100}
        }
    ))
    return _clean_layout(fig)

def plot_views_distribution(df: pd.DataFrame) -> go.Figure:
    """Plot the distribution of views (log-scaled)."""
    fig = px.histogram(
        df, 
        x="log_views", 
        nbins=50, 
        title="Distribution of Views (Log Scaled)",
        labels={"log_views": "Log(Views)", "count": "Number of Videos"},
        color_discrete_sequence=["#ff0000"]
    )
    fig.update_layout(bargap=0.1)
    return _clean_layout(fig)

def plot_views_vs_duration(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of views vs video duration."""
    df_plot = df.copy()
    df_plot['Performance'] = df_plot['high_performing'].map({1: 'High', 0: 'Normal'})
    
    fig = px.scatter(
        df_plot,
        x="duration_minutes",
        y="log_views",
        color="Performance",
        color_discrete_map={"High": "#ff0000", "Normal": "#555555"},
        hover_data=["title", "channel_title", "views"],
        title="Views vs. Video Duration",
        labels={"duration_minutes": "Duration (Minutes)", "log_views": "Log(Views)"},
        opacity=0.7
    )
    return _clean_layout(fig)

def plot_performance_by_day(df: pd.DataFrame) -> go.Figure:
    """Box plot of performance by day of week."""
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    df_copy = df.copy()
    df_copy['day_name'] = df_copy['upload_day_of_week'].map(day_map)
    df_copy['day_name'] = pd.Categorical(df_copy['day_name'], categories=list(day_map.values()), ordered=True)
    
    fig = px.box(
        df_copy,
        x="day_name",
        y="log_views",
        title="Performance by Upload Day",
        labels={"day_name": "Day of Week", "log_views": "Log(Views)"},
        color_discrete_sequence=["#ff0000"]
    )
    return _clean_layout(fig)

def plot_top_channels(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart of top channels by median views."""
    channel_stats = df.groupby("channel_title").agg(
        median_views=("views", "median"),
        video_count=("video_id", "count")
    ).reset_index()
    
    channel_stats = channel_stats[channel_stats["video_count"] >= 2]
    top_channels = channel_stats.nlargest(top_n, "median_views")
    
    fig = px.bar(
        top_channels,
        x="median_views",
        y="channel_title",
        orientation='h',
        title=f"Top {top_n} Channels by Median Views",
        labels={"median_views": "Median Views", "channel_title": "Channel"},
        color="median_views",
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return _clean_layout(fig)

def plot_correlation_matrix(df: pd.DataFrame, features: list) -> go.Figure:
    """Plot correlation matrix for selected features."""
    corr = df[features].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale=px.colors.diverging.RdBu_r
    )
    return _clean_layout(fig)
