# Visa Stock Trading Volume Forecasting

A time series forecasting app that predicts next-day Visa (V) trading volume using three models — SARIMA, XGBoost, and LSTM — and compares their performance.

🔗 **[Live App](https://timeseriesforecasting-aeoedqbgoxhq9ice2qgsve.streamlit.app/)** | 📂 **[GitHub Repo](https://github.com/dp29092000/time_series_forecasting)**

---

## Problem Statement

Trading volume is a key market signal used in algorithmic trading and risk management. This project forecasts next-day Visa (V) trading volume using classical statistical, tree-based, and deep learning approaches, then compares their accuracy.

---

## Dataset

- **Source:** Yahoo Finance via `yfinance`
- **Ticker:** V (Visa Inc.)
- **Frequency:** Daily and Weekly
- **Saved locally:** `visa_daily_volume.csv`, `visa_weekly_volume.csv` for reproducibility

---

## Models

### SARIMA(0,1,2)(1,0,0)[52]
- Trained on weekly aggregated volume
- Captures seasonality at a 52-week cycle
- Baseline statistical model

### XGBoost (Tuned)
- 180 lag features + month, ISO week, quarter
- TimeSeriesSplit cross-validation
- Hyperparameters: `n_estimators=200`, `max_depth=3`, `learning_rate=0.01`, `subsample=0.9`, `colsample_bytree=0.8`

### LSTM
- Architecture: `LSTM(64) → Dense(32, relu) → Dense(1)`
- Input shape: `(180, 1)` — 180 days of scaled volume
- Trained for 50 epochs, batch size 16
- Scaled with `MinMaxScaler` fit on training data only

---

## Model Comparison

| Model | Data | MAE | RMSE | MAPE |
|-------|------|-----|------|------|
| SARIMA | Weekly | 6,842,800 | 9,003,515 | 22.37% |
| XGBoost | Daily | 1,590,303 | 2,316,892 | 23.60% |
| LSTM | Daily | 1,666,351 | 2,313,774 | 22.12% |

XGBoost and LSTM perform comparably on error metrics. XGBoost is more stable across different input windows; LSTM is sensitive to the exact tail-180 data fetched, reflecting the high volatility of trading volume.

---

## App Structure

```
time_series_forecasting/
├── app.py                  # Landing page
├── pages/
│   ├── 1_EDA.py            # Exploratory Data Analysis
│   ├── 2_Model_Comparison.py  # Metrics and visualizations
│   └── 3_Predict.py        # Live prediction using latest 180 days
├── models/
│   ├── xgb_tuned.pkl
│   ├── lstm_model.keras
│   └── scaler.pkl
├── data/
│   ├── sarima_predictions.csv
│   ├── xgb_predictions.csv
│   └── lstm_predictions.csv
├── requirements.txt
└── runtime.txt
```

---

## Key Engineering Decisions

- **Feature engineering:** 180 lag features capture long-term volume patterns; calendar features (month, week, quarter) add seasonality signal
- **Train/test split:** `TimeSeriesSplit` used for XGBoost to prevent data leakage
- **Scaler:** `MinMaxScaler` fit on training data only; applied to test and live prediction data
- **LSTM weights:** Saved as `.keras` format; architecture rebuilt in code to avoid Keras version mismatch issues
- **Version pinning:** XGBoost 3.2.0 pinned to match Colab training environment and avoid prediction mismatch

---

## How to Run Locally

```bash
git clone https://github.com/dp29092000/time_series_forecasting.git
cd time_series_forecasting
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## Tech Stack

`Python` · `Streamlit` · `XGBoost` · `TensorFlow/Keras` · `Statsmodels` · `yfinance` · `scikit-learn` · `pandas` · `matplotlib`
