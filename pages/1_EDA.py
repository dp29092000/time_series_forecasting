import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("📊 Exploratory Data Analysis")
st.markdown("---")

@st.cache_data
def load_data():
    data = yf.download("V", period="5y", progress=False)
    data.columns = data.columns.get_level_values(0)
    df = data[['Volume']].copy().sort_index()
    df_weekly = df.resample('W').sum()
    return df, df_weekly

df, df_weekly = load_data()

# ── RAW VOLUME ────────────────────────────────────────
st.subheader("Daily Trading Volume")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['Volume'], alpha=0.7)
ax.set_title("Visa Daily Trading Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
st.pyplot(fig)

# ── ROLLING MEAN ──────────────────────────────────────
st.subheader("Rolling Mean (30-day)")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['Volume'], alpha=0.5, label='Volume')
ax.plot(df['Volume'].rolling(window=30).mean(), color='red', label='30-day Rolling Mean')
ax.set_title("Visa Volume with Rolling Mean")
ax.legend()
st.pyplot(fig)

# ── DECOMPOSITION ─────────────────────────────────────
st.subheader("Seasonal Decomposition")
result = seasonal_decompose(df['Volume'], model='additive', period=252)
fig = result.plot()
fig.set_size_inches(12, 8)
plt.tight_layout()
st.pyplot(fig)

# ── ADF TEST ──────────────────────────────────────────
st.subheader("Stationarity Test (ADF)")
adf_result = adfuller(df['Volume'])
col1, col2 = st.columns(2)
with col1:
    st.metric("ADF p-value", f"{adf_result[1]:.6f}")
with col2:
    if adf_result[1] < 0.05:
        st.success("✅ Series is Stationary (p < 0.05)")
    else:
        st.error("❌ Series is Non-Stationary (p > 0.05)")
