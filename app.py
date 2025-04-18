import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

model = ['random_forest_model.pkl', 'hist_gradient_boosting_model.pkl']
i = 0

# ——————————————————————————————————————————————
# 1) Load your trained RandomForest model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('random_forest_model.pkl')

rf_model = load_model()

# ——————————————————————————————————————————————
st.title("Stock Close Price Analysis & Prediction")

st.markdown(
    "Upload a CSV with columns: `Date`, `Open`, `High`, `Low`, `Close`, "
    "`Volume`, `% Change` (optional)."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    # ——————————————————————————————————————
    # 2) Read & feature‐engineer
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)

    for lag in [1, 2, 3, 5]:
        df[f'Lag{lag}_Close'] = df['Close'].shift(lag)

    df['Pct_Change'] = df['Close'].pct_change()
    df['HL_Range'] = df['High'] - df['Low']

    df = df.dropna()

    features = ["Open","High","Low","Close","Volume",
                "MA_5","MA_10","Volatility","Momentum",
                "Lag1_Close","Lag2_Close","Lag3_Close","Lag5_Close",
                "Pct_Change","HL_Range"]
    X = df[features]
    y = df['Close']

    # ——————————————————————————————————————
    # 3) Predict
    df['Predicted_Close'] = rf_model.predict(X)

    # ——————————————————————————————————————
    # 4) Plot 1: Close with MAs
    st.subheader("Close Price with 5‑Day & 10‑Day Moving Averages")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df["Close"], label="Close Price")
    ax1.plot(df.index, df["MA_5"],   label="5‑Day MA")
    ax1.plot(df.index, df["MA_10"],  label="10‑Day MA")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)

    # ——————————————————————————————————————
    # 5) Plot 2: Correlation matrix
    st.subheader("Correlation Matrix of Features")
    corr = df[features + ['Close']].corr()
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    cax = ax2.imshow(corr, cmap='coolwarm', interpolation='none')
    fig2.colorbar(cax, ax=ax2)
    ax2.set_xticks(range(len(corr)))
    ax2.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax2.set_yticks(range(len(corr)))
    ax2.set_yticklabels(corr.columns)
    st.pyplot(fig2)

    # ——————————————————————————————————————
    # 6) Plot 3: Actual vs Predicted
    st.subheader("Actual vs Predicted Close Price")
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    ax3.plot(df.index, df['Close'],           label="Actual")
    ax3.plot(df.index, df['Predicted_Close'], linestyle="--", label="Predicted")
    ax3.set_xlabel("Date")
    ax3.set_ylabel("Close Price")
    ax3.legend()
    st.pyplot(fig3)

    # ——————————————————————————————————————
    # 7) Plot 4: Feature importances
    st.subheader("Feature Importances")
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    ax4.bar(range(len(features)), importances[idx], align='center')
    ax4.set_xticks(range(len(features)))
    ax4.set_xticklabels([features[i] for i in idx], rotation=45, ha='right')
    ax4.set_ylabel("Importance")
    st.pyplot(fig4)

    # ——————————————————————————————————————
    # 8) Metrics on hold‑out (last 20%)
    split_idx = int(len(df) * 0.8)
    y_true = df['Close'].iloc[split_idx:]
    y_pred = df['Predicted_Close'].iloc[split_idx:]
    mse = mean_squared_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    st.subheader("Test Set Performance (last 20%)")
    st.markdown(f"- **MSE:** {mse:.4f}  \n- **R²:** {r2:.4f}")

    # ——————————————————————————————————————
    # 9) Show recent data
    st.subheader("Latest Data & Predictions")
    st.dataframe(df.tail())

else:
    st.info("🔹 Please upload a CSV file to run the analysis.")