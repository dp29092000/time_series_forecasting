import streamlit as st

st.set_page_config(
    page_title="Visa Volume Forecasting",
    page_icon="📈",
    layout="wide"
)

# header
st.title("📈 Visa Stock Volume Forecasting")
st.markdown("#### Predicting daily trading volume using Classical, ML, and Deep Learning approaches")
st.markdown("---")

# about section in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🎯 Objective")
    st.markdown("""
    Forecast next-day Visa (V) stock trading volume, a proxy for payment network activity, using three distinct modeling approaches and compare their performance.
    """)

with col2:
    st.markdown("### 📊 Dataset")
    st.markdown("""
    - **Source:** Yahoo Finance via yfinance  
    - **Ticker:** Visa Inc. (V)  
    - **Daily Data:** 1,254 rows  
    - **Weekly Data:** 262 rows  
    - **Target:** Trading Volume  
    """)

with col3:
    st.markdown("### 🤖 Models")
    st.markdown("""
    - **SARIMA**: Classical baseline (weekly)  
    - **XGBoost**: ML with lag features (daily)  
    - **LSTM**: Deep learning sequence model (daily)  
    """)

st.markdown("---")
st.markdown("### 🗺️ Navigate")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**📊 EDA**: Explore volume trends, seasonality, and stationarity")
with col2:
    st.info("**🏆 Model Comparison**: Compare SARIMA, XGBoost, and LSTM performance")
with col3:
    st.info("**🔮 Predict**: Forecast next day's Visa trading volume")

st.markdown("---")
st.caption("Built by Prasanna D | IIT Gandhinagar | University of Maryland (MAML, Fall 2026)")