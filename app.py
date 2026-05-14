import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AnomalyGuard",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

MAX_ROWS = 100000
CHART_SAMPLE = 30000
TABLE_ROWS = 1000

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.title("🛡️ AnomalyGuard")
st.caption("Isolation Forest Based Network Anomaly Detection")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:

    st.header("⚙️ Configuration")

    scaler_path = st.text_input(
        "Scaler Path",
        value="scaler.pkl"
    )

    model_path = st.text_input(
        "Model Path",
        value="isolation_model.pkl"
    )

    cols_path = st.text_input(
        "Columns Path",
        value="columns.pkl"
    )

    st.divider()

    show_raw = st.checkbox("Show Raw Data", False)
    anomalies_only = st.checkbox("Show Only Anomalies", False)

# ─────────────────────────────────────────────────────────────
# CACHE MODEL LOADING
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_artifacts():

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    columns = joblib.load(cols_path)

    return scaler, model, columns

# ─────────────────────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV dataset to begin analysis.")
    st.stop()

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────

try:

    with st.spinner("Loading dataset..."):

        df_raw = pd.read_csv(
            uploaded_file,
            low_memory=False
        )

except Exception as e:

    st.error(f"Failed to load CSV: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# LIMIT HUGE DATASETS
# ─────────────────────────────────────────────────────────────

if len(df_raw) > MAX_ROWS:

    df_raw = df_raw.sample(
        MAX_ROWS,
        random_state=42
    )

    st.warning(
        f"Dataset too large. Sampled {MAX_ROWS:,} rows for stability."
    )

# ─────────────────────────────────────────────────────────────
# RAW PREVIEW
# ─────────────────────────────────────────────────────────────

if show_raw:

    st.subheader("Raw Dataset")

    st.dataframe(
        df_raw.head(20),
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

drop_cols = [
    'Src IP dec',
    'Dst IP dec',
    'Timestamp',
    'Attempted Category'
]

with st.spinner("Preprocessing data..."):

    df = df_raw.copy()

    df.drop(
        columns=drop_cols,
        errors='ignore',
        inplace=True
    )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(inplace=True)

# ─────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────

X = df.drop(
    columns=['Label'],
    errors='ignore'
)

# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────

try:

    scaler, model, columns = load_artifacts()

except Exception as e:

    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────
# FIX COLUMN ORDER
# ─────────────────────────────────────────────────────────────

for col in columns:

    if col not in X.columns:
        X[col] = 0

X = X[columns]

# ─────────────────────────────────────────────────────────────
# MEMORY OPTIMIZATION
# ─────────────────────────────────────────────────────────────

X = X.astype(np.float32)

# ─────────────────────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────────────────────

progress = st.progress(
    0,
    text="Running inference..."
)

try:

    progress.progress(30)

    X_scaled = scaler.transform(X)

    progress.progress(60)

    y_pred_raw = model.predict(X_scaled)

    progress.progress(80)

    scores = model.decision_function(X_scaled)

    progress.progress(100)

except Exception as e:

    st.error(f"Inference failed: {e}")
    st.stop()

progress.empty()

# ─────────────────────────────────────────────────────────────
# PREDICTION CONVERSION
# ─────────────────────────────────────────────────────────────

y_pred = np.where(
    y_pred_raw == 1,
    0,
    1
)

df['Prediction'] = y_pred

df['Status'] = np.where(
    df['Prediction'] == 1,
    '🚨 Anomaly',
    '✅ Normal'
)

df['Anomaly Score'] = scores

# ─────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────

total = len(df)

n_normal = int((y_pred == 0).sum())
n_anomaly = int((y_pred == 1).sum())

normal_pct = n_normal / total * 100
anomaly_pct = n_anomaly / total * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Records",
    f"{total:,}"
)

col2.metric(
    "Features",
    X.shape[1]
)

col3.metric(
    "Normal",
    f"{n_normal:,}"
)

col4.metric(
    "Anomalies",
    f"{n_anomaly:,}"
)

# ─────────────────────────────────────────────────────────────
# CHART DATA SAMPLE
# ─────────────────────────────────────────────────────────────

df_chart = df.sample(
    min(CHART_SAMPLE, len(df)),
    random_state=42
)

# ─────────────────────────────────────────────────────────────
# PIE CHART
# ─────────────────────────────────────────────────────────────

st.subheader("Traffic Distribution")

pie_df = pd.DataFrame({
    'Category': ['Normal', 'Anomaly'],
    'Count': [n_normal, n_anomaly]
})

fig_pie = px.pie(
    pie_df,
    names='Category',
    values='Count',
    hole=0.6
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# HISTOGRAM
# ─────────────────────────────────────────────────────────────

numeric_cols = df_chart.select_dtypes(
    include=np.number
).columns.tolist()

ignore_cols = [
    'Prediction'
]

numeric_cols = [
    c for c in numeric_cols
    if c not in ignore_cols
]

if numeric_cols:

    selected_feature = st.selectbox(
        "Feature Distribution",
        numeric_cols
    )

    fig_hist = px.histogram(
        df_chart,
        x=selected_feature,
        color='Status',
        nbins=20
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────
# ANOMALY SCORE DISTRIBUTION
# ─────────────────────────────────────────────────────────────

st.subheader("Anomaly Score Distribution")

fig_score = px.histogram(
    df_chart,
    x='Anomaly Score',
    color='Status',
    nbins=30
)

st.plotly_chart(
    fig_score,
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# TOP ANOMALIES
# ─────────────────────────────────────────────────────────────

st.subheader("Top Suspicious Records")

top_anomalies = df[
    df['Prediction'] == 1
].sort_values(
    by='Anomaly Score'
).head(20)

st.dataframe(
    top_anomalies.head(20),
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────────────────────

st.subheader("Detailed Results")

display_df = df

if anomalies_only:
    display_df = df[df['Prediction'] == 1]

st.dataframe(
    display_df.head(TABLE_ROWS),
    use_container_width=True
)

# ─────────────────────────────────────────────────────────────
# DOWNLOADS
# ─────────────────────────────────────────────────────────────

st.subheader("Export Results")

csv_full = df.to_csv(
    index=False
).encode('utf-8')

st.download_button(
    "Download Full Results",
    csv_full,
    "results.csv",
    "text/csv"
)

csv_anomaly = df[
    df['Prediction'] == 1
].to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Anomalies Only",
    csv_anomaly,
    "anomalies.csv",
    "text/csv"
)

st.success("Analysis completed successfully.")
