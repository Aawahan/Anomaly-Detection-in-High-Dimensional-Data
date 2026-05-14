import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🚨 Anomaly Detection System")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, low_memory=False)

    st.write("### Uploaded Data")
    st.dataframe(df.head())

    df = df.drop(columns=['Src IP dec', 'Dst IP dec', 'Timestamp', 'Attempted Category'], errors='ignore')

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=['Label'], errors='ignore')

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("isolation_model.pkl")
    columns = joblib.load("columns.pkl")

    for col in columns:
        if col not in X.columns:
            X[col] = 0

    X = X[columns]

    X_scaled = scaler.transform(X)

    y_pred = model.predict(X_scaled)
    y_pred = [0 if x == 1 else 1 for x in y_pred]

    df['Prediction'] = y_pred

    st.success("Prediction completed!")

    st.write("### Results")
    st.dataframe(df.head())

    st.write(f"Normal: {y_pred.count(0)}")
    st.write(f"Anomalies: {y_pred.count(1)}")

total = len(y_pred)

normal_percent = (y_pred.count(0) / total) * 100
anomaly_percent = (y_pred.count(1) / total) * 100

st.write(f"Normal Traffic: {normal_percent:.2f}%")
st.write(f"Anomalous Traffic: {anomaly_percent:.2f}%")
