import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Model Comparison", layout = "wide", page_icon = "🏆")
st.title("🏆 Model Comparison")
st.markdown("---")

st.subheader("Model Performance Metrics")

metrics_df = pd.DataFrame({
    'Model': ['SARIMA (Weekly)', 'XGBoost (Daily)', 'XGBoost Tuned (Daily)', 'LSTM (Daily)'],
    'MAE': ['6,842,800', '1,644,265', '1,590,303', '1,666,351'],
    'RMSE': ['9,003,515', '2,406,246', '2,316,892', '2,313,774'],
    'MAPE': ['22.37%', '23.87%', '23.60%', '22.12%']
})

st.dataframe(metrics_df, use_container_width=True, hide_index=True)
st.markdown("---")

sarima_preds_df = pd.read_csv("data/sarima_predictions.csv")
st.subheader("Actual vs SARIMA Predictions (Weekly)")
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(sarima_preds_df['actual'], label = 'Actual')
ax.plot(sarima_preds_df['predicted'], label='Predicted', color='red')
ax.set_xlabel("Week")
ax.set_ylabel("Volume")
ax.legend()
st.pyplot(fig)
st.caption("Test period: 53 weeks of weekly data")
st.markdown("---")

xgboost_preds_df = pd.read_csv("data/xgb_predictions.csv")
st.subheader("Actual vs XGBoost Predictions")
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(xgboost_preds_df['actual'], label = 'Actual')
ax.plot(xgboost_preds_df['predicted'], label='Predicted', color='red')
ax.set_xlabel("Day")
ax.set_ylabel("Volume")
ax.legend()
st.pyplot(fig)
st.caption("Test period: 215 trading days")
st.markdown("---")

lstm_preds_df = pd.read_csv("data/lstm_predictions.csv")
st.subheader("Actual vs LSTM Predictions")
fig,ax = plt.subplots(figsize=(12,4))
ax.plot(lstm_preds_df['actual'], label = 'Actual')
ax.plot(lstm_preds_df['predicted'], label='Predicted', color='red')
ax.set_xlabel("Day")
ax.set_ylabel("Volume")
ax.legend()
st.pyplot(fig)
st.caption("Test period: 71 trading days (reduced due to 180-day sequence window)")