import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AnomalyGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #0f1629;
    --bg-card: #131929;
    --bg-card-hover: #1a2235;
    --border: #1e2d45;
    --border-glow: #2563eb;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #475569;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: var(--sans);
    background-color: var(--bg-primary);
    color: var(--text-primary);
}

.main .block-container {
    padding: 2rem 2.5rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
}

/* ── Hero Header ── */
.hero-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
    padding-bottom: 2rem;
    border-bottom: 1px solid var(--border);
}

.hero-icon {
    font-size: 2.8rem;
    line-height: 1;
}

.hero-title {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.1;
}

.hero-sub {
    font-size: 0.85rem;
    color: var(--accent-cyan);
    font-family: var(--mono);
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.25rem;
}

/* ── Status Pill ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    border: 1px solid;
    margin-left: auto;
}

.status-ready {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.35);
    color: var(--accent-green);
}

.status-idle {
    background: rgba(100,116,139,0.12);
    border-color: rgba(100,116,139,0.35);
    color: var(--text-muted);
}

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}

.metric-card.blue::before  { background: var(--accent-blue); }
.metric-card.cyan::before  { background: var(--accent-cyan); }
.metric-card.green::before { background: var(--accent-green); }
.metric-card.red::before   { background: var(--accent-red); }

.metric-label {
    font-family: var(--mono);
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.6rem;
}

.metric-value {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
    color: var(--text-primary);
}

.metric-value.green { color: var(--accent-green); }
.metric-value.red   { color: var(--accent-red); }
.metric-value.cyan  { color: var(--accent-cyan); }

.metric-sub {
    font-size: 0.78rem;
    color: var(--text-secondary);
    margin-top: 0.4rem;
}

/* ── Section Headings ── */
.section-title {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Upload Zone ── */
.upload-zone {
    background: var(--bg-card);
    border: 2px dashed var(--border);
    border-radius: 16px;
    padding: 3rem 2rem;
    text-align: center;
    transition: border-color 0.2s;
}

/* ── Alert Boxes ── */
.alert {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.88rem;
    margin: 1rem 0;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 0.7rem;
}

.alert-success {
    background: rgba(16,185,129,0.1);
    border-color: var(--accent-green);
    color: #6ee7b7;
}

.alert-danger {
    background: rgba(239,68,68,0.1);
    border-color: var(--accent-red);
    color: #fca5a5;
}

.alert-info {
    background: rgba(59,130,246,0.1);
    border-color: var(--accent-blue);
    color: #93c5fd;
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
    background: var(--bg-card) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s !important;
}

.stFileUploader > div:hover {
    border-color: var(--accent-blue) !important;
}

div[data-testid="stDataFrameResizable"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
}

.stButton > button {
    background: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover {
    opacity: 0.85 !important;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)) !important;
}

/* hide streamlit branding */
#MainMenu, footer { visibility: hidden; }

/* ── Separator ── */
hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 2rem 0;
}

/* ── Sidebar elements ── */
.sidebar-section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin: 0.2rem;
}

.tag-normal {
    background: rgba(16,185,129,0.15);
    color: var(--accent-green);
    border: 1px solid rgba(16,185,129,0.3);
}

.tag-anomaly {
    background: rgba(239,68,68,0.15);
    color: var(--accent-red);
    border: 1px solid rgba(239,68,68,0.3);
}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:1.1rem; font-weight:700;
                color:#f1f5f9; margin-bottom:0.3rem;">🛡️ AnomalyGuard</div>
    <div style="font-size:0.7rem; color:#06b6d4; font-family:'Space Mono',monospace;
                letter-spacing:0.1em; text-transform:uppercase; margin-bottom:1.5rem;">
        Isolation Forest Engine</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">⚙️ Model Configuration</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

    scaler_path = st.text_input("Scaler path", value="scaler.pkl", help="Path to joblib scaler file")
    model_path  = st.text_input("Model path",  value="isolation_model.pkl")
    cols_path   = st.text_input("Columns path", value="columns.pkl")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">🗑️ Columns to Drop</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

    default_drop = "Src IP dec, Dst IP dec, Timestamp, Attempted Category"
    drop_cols_raw = st.text_area("Comma-separated", value=default_drop, height=80)
    drop_cols = [c.strip() for c in drop_cols_raw.split(",") if c.strip()]

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">📊 Display Options</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)

    preview_rows = st.slider("Preview rows", 5, 100, 10)
    show_raw     = st.checkbox("Show raw data table", value=True)
    show_anomaly_only = st.checkbox("Filter anomalies only", value=False)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem; padding-top:1.5rem; border-top:1px solid #1e2d45;
                font-family:'Space Mono',monospace; font-size:0.65rem;
                color:#475569; text-align:center;">
        Isolation Forest · Unsupervised<br>Anomaly Detection Pipeline
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-icon">🛡️</div>
    <div>
        <div class="hero-title">AnomalyGuard</div>
        <div class="hero-sub">Network Traffic Anomaly Detection System</div>
    </div>
    <div class="status-pill status-idle" id="status-pill">⬤ &nbsp;Awaiting Input</div>
</div>
""", unsafe_allow_html=True)

# ─── Upload Section ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">01 &nbsp;Data Ingestion</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your network traffic CSV here",
    type=["csv"],
    help="Upload a CSV file containing network traffic features. The model expects pre-defined columns.",
)
MAX_ROWS = 100000

if len(df_raw) > MAX_ROWS:
    df_raw = df_raw.sample(MAX_ROWS, random_state=42)
    st.warning(f"Dataset too large. Randomly sampled {MAX_ROWS:,} rows.")
    
# ─── Main Logic ───────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="alert alert-info">
        ℹ️ Upload a CSV file above to begin analysis. Configure model paths in the sidebar if needed.
    </div>
    """, unsafe_allow_html=True)

    # Empty state metric cards
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-card blue">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">—</div>
            <div class="metric-sub">Awaiting data</div>
        </div>
        <div class="metric-card cyan">
            <div class="metric-label">Features</div>
            <div class="metric-value">—</div>
            <div class="metric-sub">Awaiting data</div>
        </div>
        <div class="metric-card green">
            <div class="metric-label">Normal Traffic</div>
            <div class="metric-value green">—</div>
            <div class="metric-sub">Awaiting inference</div>
        </div>
        <div class="metric-card red">
            <div class="metric-label">Anomalies</div>
            <div class="metric-value red">—</div>
            <div class="metric-sub">Awaiting inference</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Load & Preprocess ───────────────────────────────────────────────────────
with st.spinner("Loading data…"):
    df_raw = pd.read_csv(uploaded_file, low_memory=False)

st.markdown('<div class="section-title">02 &nbsp;Preprocessing</div>', unsafe_allow_html=True)

col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.metric("Raw Rows", f"{len(df_raw):,}")
with col_info2:
    st.metric("Raw Columns", f"{df_raw.shape[1]}")
with col_info3:
    inf_count = np.isinf(df_raw.select_dtypes(include=[np.number])).sum().sum()
    st.metric("Inf / NaN Values", f"{int(inf_count) + df_raw.isnull().sum().sum():,}")

if show_raw:
    with st.expander("🔍 Raw Data Preview", expanded=False):
        st.dataframe(df_raw.head(preview_rows), use_container_width=True)

# ─── Preprocessing Pipeline ──────────────────────────────────────────────────
with st.spinner("Cleaning data…"):
    df = df_raw.copy()
    df = df.drop(columns=drop_cols, errors='ignore')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    rows_before = len(df)
    df.dropna(inplace=True)
    rows_dropped = rows_before - len(df)

if rows_dropped > 0:
    st.markdown(f"""
    <div class="alert alert-info">
        ⚠️ Dropped <strong>{rows_dropped:,}</strong> rows containing NaN or Inf values
        ({rows_dropped/rows_before*100:.1f}% of data).
    </div>
    """, unsafe_allow_html=True)

X = df.drop(columns=['Label'], errors='ignore')

# ─── Load Models ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">03 &nbsp;Model Inference</div>', unsafe_allow_html=True)

model_status = st.empty()
try:
    with st.spinner("Loading model artefacts…"):
        scaler  = joblib.load(scaler_path)
        model   = joblib.load(model_path)
        columns = joblib.load(cols_path)

    model_status.markdown("""
    <div class="alert alert-success">✅ Model artefacts loaded successfully.</div>
    """, unsafe_allow_html=True)
except FileNotFoundError as e:
    model_status.markdown(f"""
    <div class="alert alert-danger">
        ❌ Could not load model files: <code>{e}</code><br>
        Ensure <code>scaler.pkl</code>, <code>isolation_model.pkl</code>,
        and <code>columns.pkl</code> are in the working directory.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── Align Columns & Scale ───────────────────────────────────────────────────
for col in columns:
    if col not in X.columns:
        X[col] = 0
X = X[columns]

progress_bar = st.progress(0, text="Running inference…")
for pct in range(0, 101, 10):
    time.sleep(0.04)
    progress_bar.progress(pct, text=f"Running inference… {pct}%")

X_scaled = scaler.transform(X)
y_pred_raw = model.predict(X_scaled)
y_pred = [0 if x == 1 else 1 for x in y_pred_raw]   # 0 = Normal, 1 = Anomaly

df['Prediction'] = y_pred
df['Status'] = df['Prediction'].map({0: '✅ Normal', 1: '🚨 Anomaly'})

progress_bar.empty()

# ─── KPI Cards ───────────────────────────────────────────────────────────────
total         = len(y_pred)
n_normal      = y_pred.count(0)
n_anomaly     = y_pred.count(1)
normal_pct    = n_normal  / total * 100
anomaly_pct   = n_anomaly / total * 100
n_features    = X.shape[1]

st.markdown(f"""
<div class="metric-grid">
    <div class="metric-card blue">
        <div class="metric-label">Total Records</div>
        <div class="metric-value">{total:,}</div>
        <div class="metric-sub">After preprocessing</div>
    </div>
    <div class="metric-card cyan">
        <div class="metric-label">Features Used</div>
        <div class="metric-value cyan">{n_features}</div>
        <div class="metric-sub">Model input dimensions</div>
    </div>
    <div class="metric-card green">
        <div class="metric-label">Normal Traffic</div>
        <div class="metric-value green">{n_normal:,}</div>
        <div class="metric-sub">{normal_pct:.1f}% of total</div>
    </div>
    <div class="metric-card red">
        <div class="metric-label">Anomalies Detected</div>
        <div class="metric-value red">{n_anomaly:,}</div>
        <div class="metric-sub">{anomaly_pct:.1f}% of total</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="alert alert-success">
    🎯 Inference complete — results ready for review below.
</div>
""", unsafe_allow_html=True)

# ─── Charts ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">04 &nbsp;Visual Analysis</div>', unsafe_allow_html=True)

chart_col1, chart_col2 = st.columns([1, 2])

# Donut chart
with chart_col1:
    fig_donut = go.Figure(go.Pie(
        labels=['Normal', 'Anomaly'],
        values=[n_normal, n_anomaly],
        hole=0.65,
        marker=dict(
            colors=['#10b981', '#ef4444'],
            line=dict(color='#0a0e1a', width=3)
        ),
        textinfo='label+percent',
        textfont=dict(family='Space Mono', size=12, color='#f1f5f9'),
        hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>',
    ))
    fig_donut.update_layout(
        title=dict(text='Traffic Split', font=dict(family='Space Mono', size=13, color='#94a3b8')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        showlegend=False,
        margin=dict(t=50, b=20, l=20, r=20),
        annotations=[dict(
            text=f"<b>{anomaly_pct:.1f}%</b><br><span style='font-size:10px'>anomalous</span>",
            x=0.5, y=0.5,
            font=dict(family='Space Mono', size=14, color='#ef4444'),
            showarrow=False
        )]
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# Bar chart by feature (score distribution or numeric col)
with chart_col2:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c != 'Prediction']

    if num_cols:
        selected_feat = st.selectbox(
            "Feature to inspect",
            num_cols,
            index=0,
            help="Select a numeric feature to compare its distribution across normal vs anomaly traffic"
        )

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=df[df['Prediction'] == 0][selected_feat],
            name='Normal',
            marker_color='rgba(16,185,129,0.7)',
            nbinsx=50,
            hovertemplate='Value: %{x}<br>Count: %{y}<extra>Normal</extra>',
        ))
        fig_hist.add_trace(go.Histogram(
            x=df[df['Prediction'] == 1][selected_feat],
            name='Anomaly',
            marker_color='rgba(239,68,68,0.7)',
            nbinsx=50,
            hovertemplate='Value: %{x}<br>Count: %{y}<extra>Anomaly</extra>',
        ))
        fig_hist.update_layout(
            barmode='overlay',
            title=dict(text=f'Distribution of <b>{selected_feat}</b>', font=dict(family='Space Mono', size=13, color='#94a3b8')),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(19,25,41,0.6)',
            font=dict(family='DM Sans', color='#94a3b8'),
            xaxis=dict(gridcolor='#1e2d45', title=selected_feat),
            yaxis=dict(gridcolor='#1e2d45', title='Count'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(family='Space Mono', size=11)),
            margin=dict(t=50, b=40, l=40, r=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

# ─── Anomaly Timeline (if index is meaningful) ───────────────────────────────
if total > 1:
    fig_timeline = go.Figure()
    window = 200  # rolling window for density

    anomaly_series = pd.Series(y_pred)
    rolling_anomaly = anomaly_series.rolling(window, min_periods=1).mean() * 100

    fig_timeline.add_trace(go.Scatter(
        x=list(range(len(rolling_anomaly))),
        y=rolling_anomaly,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#ef4444', width=1.5),
        fillcolor='rgba(239,68,68,0.12)',
        name='Anomaly density (%)',
        hovertemplate='Record #%{x}<br>Anomaly density: %{y:.1f}%<extra></extra>',
    ))
    fig_timeline.update_layout(
        title=dict(text=f'Rolling Anomaly Density (window={window})', font=dict(family='Space Mono', size=13, color='#94a3b8')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(19,25,41,0.6)',
        font=dict(family='DM Sans', color='#94a3b8'),
        xaxis=dict(gridcolor='#1e2d45', title='Record Index'),
        yaxis=dict(gridcolor='#1e2d45', title='Anomaly %', ticksuffix='%'),
        margin=dict(t=50, b=40, l=50, r=20),
        hovermode='x unified',
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# ─── Results Table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">05 &nbsp;Detailed Results</div>', unsafe_allow_html=True)

display_df = df[df['Prediction'] == 1] if show_anomaly_only else df

st.dataframe(
    display_df[['Status'] + [c for c in display_df.columns if c not in ('Prediction', 'Status')]].head(preview_rows),
    use_container_width=True,
    height=350,
)

# ─── Download ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">06 &nbsp;Export</div>', unsafe_allow_html=True)

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    csv_all = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Download Full Results (CSV)",
        data=csv_all,
        file_name="anomaly_detection_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

with dl_col2:
    anomalies_only = df[df['Prediction'] == 1]
    csv_anomalies = anomalies_only.to_csv(index=False).encode('utf-8')
    st.download_button(
        "🚨 Download Anomalies Only (CSV)",
        data=csv_anomalies,
        file_name="anomalies_only.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("""
<div style="margin-top:3rem; padding-top:1.5rem; border-top:1px solid #1e2d45;
            font-family:'Space Mono',monospace; font-size:0.65rem;
            color:#475569; text-align:center;">
    AnomalyGuard · Isolation Forest · Unsupervised Anomaly Detection
</div>
""", unsafe_allow_html=True)
