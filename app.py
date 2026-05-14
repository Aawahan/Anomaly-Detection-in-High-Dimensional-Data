import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="AnomalyGuard",
    page_icon="🛡️",
    layout="wide"
)

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────

MAX_ROWS = 30000
CHART_ROWS = 5000
TABLE_ROWS = 200

# ─────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────

st.title("🛡️ AnomalyGuard")
st.caption("Isolation Forest Network Anomaly Detection")

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:

    st.header("⚙️ Configuration")

    show_raw = st.checkbox(
        "Show Raw Data",
        False
    )

    anomaly_only = st.checkbox(
        "Show Only Anomalies",
        False
    )

# ─────────────────────────────────────────────
# MODEL CACHE
# ─────────────────────────────────────────────

@st.cache_resource
def load_artifacts():

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("isolation_model.pkl")
    columns = joblib.load("columns.pkl")

    return scaler, model, columns

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

# ─────────────────────────────────────────────
# LOAD CSV
# ─────────────────────────────────────────────

try:

    df_raw = pd.read_csv(
        uploaded_file,
        low_memory=False
    )

except Exception as e:

    st.error(f"CSV loading failed: {e}")
    st.stop()

# ─────────────────────────────────────────────
# DATA SAMPLING
# ─────────────────────────────────────────────

if len(df_raw) > MAX_ROWS:

    df_raw = df_raw.sample(
        MAX_ROWS,
        random_state=42
    )

    st.warning(
        f"Large dataset detected. Sampled {MAX_ROWS:,} rows for stability."
    )

# ─────────────────────────────────────────────
# RAW PREVIEW
# ─────────────────────────────────────────────

if show_raw:

    st.subheader("Raw Dataset")

    st.dataframe(
        df_raw.head(20),
        width='stretch'
    )

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

drop_cols = [
    'Src IP dec',
    'Dst IP dec',
    'Timestamp',
    'Attempted Category'
]

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

X = df.drop(
    columns=['Label'],
    errors='ignore'
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

try:

    scaler, model, columns = load_artifacts()

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()

# ─────────────────────────────────────────────
# FIX COLUMNS
# ─────────────────────────────────────────────

for col in columns:

    if col not in X.columns:
        X[col] = 0

X = X[columns]

# ─────────────────────────────────────────────
# MEMORY OPTIMIZATION
# ─────────────────────────────────────────────

X = X.astype(np.float32)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────

with st.spinner("Running inference..."):

    X_scaled = scaler.transform(X)

    y_pred_raw = model.predict(X_scaled)

    scores = model.decision_function(X_scaled)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

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

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

total = len(df)

normal_count = int((y_pred == 0).sum())
anomaly_count = int((y_pred == 1).sum())

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Records",
    f"{total:,}"
)

col2.metric(
    "Normal",
    f"{normal_count:,}"
)

col3.metric(
    "Anomalies",
    f"{anomaly_count:,}"
)

# ─────────────────────────────────────────────
# SMALL PIE CHART
# ─────────────────────────────────────────────

st.subheader("Traffic Distribution")

pie_df = pd.DataFrame({
    'Type': ['Normal', 'Anomaly'],
    'Count': [normal_count, anomaly_count]
})

fig = px.pie(
    pie_df,
    names='Type',
    values='Count',
    hole=0.5
)

st.plotly_chart(
    fig,
    width='stretch'
)

# ─────────────────────────────────────────────
# TOP ANOMALIES
# ─────────────────────────────────────────────

st.subheader("Top Suspicious Records")

top_anomalies = df[
    df['Prediction'] == 1
].sort_values(
    by='Anomaly Score'
).head(20)

st.dataframe(
    top_anomalies,
    width='stretch'
)

# ─────────────────────────────────────────────
# RESULTS TABLE
# ─────────────────────────────────────────────

st.subheader("Results")

display_df = df

if anomaly_only:
    display_df = df[df['Prediction'] == 1]

st.dataframe(
    display_df.head(TABLE_ROWS),
    width='stretch'
)

# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

csv = df.to_csv(
    index=False
).encode('utf-8')

st.download_button(
    "Download Results",
    csv,
    "results.csv",
    "text/csv"
)

st.success("Analysis completed successfully.")
