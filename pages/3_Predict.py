import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Prediction", layout="wide", page_icon = "🔮")
st.title("🔮 Prediction")

@st.cache_data
def load_data():
    data = yf.download("V", period = "1y", progress = False)
    data.columns = data.columns.get_level_values(0)
    df = data[["Volume"]].copy().sort_index()
    return df

df = load_data()

xgb_model = joblib.load('models/xgb_tuned.pkl')
scaler = joblib.load('models/scaler.pkl')
lstm_model = load_model('models/lstm_model.keras')


option = st.selectbox(
    'Which model would you like to use?',
    [None,'XGBoost Tuned','LSTM']
)
if option is not None:
    if option == 'XGBoost Tuned':
        # get last 180 days of volume
        last_180 = df['Volume'].tail(180).values[::-1]

        features = {}
        for i in range(1, 181):
            features[f'lag_{i}'] = last_180[i-1]

        latest_date = df.index[-1]
        tomorrow = latest_date + pd.Timedelta(days=1)
        features['month'] = tomorrow.month
        features['week'] = tomorrow.isocalendar()[1]
        features['quarter'] = tomorrow.quarter

        input_df = pd.DataFrame([features])
        
        xgb_prediction = xgb_model.predict(input_df)[0]

        st.markdown("---")
        st.metric(label="📈 Predicted Next Day Volume", value=f"{xgb_prediction:,.0f}")

        st.subheader("Last 180 Days - Visa Trading Volume")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['Volume'].tail(180).values, label='Historical Volume')
        ax.axhline(y=xgb_prediction, color='red', linestyle='--', label='Predicted Next Day')
        ax.set_xlabel("Day")
        ax.set_ylabel("Volume")
        ax.legend()
        st.pyplot(fig)

    elif option == 'LSTM':
        test_scaled = scaler.transform(df['Volume'].tail(180).values.reshape(-1,1))
        test_scaled = test_scaled.reshape((1,180,1))
        lstm_pred_scaled = lstm_model.predict(test_scaled)
        lstm_prediction = scaler.inverse_transform(lstm_pred_scaled)[0][0]

        st.markdown("---")
        st.metric(label="📈 Predicted Next Day Volume", value=f"{lstm_prediction:,.0f}")

        st.subheader("Last 180 Days - Visa Trading Volume")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df['Volume'].tail(180).values, label='Historical Volume')
        ax.axhline(y=lstm_prediction, color='red', linestyle='--', label='Predicted Next Day')
        ax.set_xlabel("Day")
        ax.set_ylabel("Volume")
        ax.legend()
        st.pyplot(fig)

