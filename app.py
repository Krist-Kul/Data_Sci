import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Cache the model loading for performance
def load_model():
    return joblib.load('random_forest_model.pkl')

model = st.cache(load_model)()

st.title("Stock Close Price Prediction App")

st.markdown(
    "Upload a CSV file containing historical stock data with columns: `Date`, `Open`, `High`, `Low`, `Close`, `Volume (Shares)`, `Value ('000 Baht)`, `% Change`"
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')

    # Feature engineering
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['Volatility'] = df['Close'].rolling(window=5).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    df = df.dropna()

    # Prepare features for prediction
    features = [
        'Open', 'High', 'Low', 'Volume (Shares)', "Value ('000 Baht)",
        'MA_5', 'MA_10', 'Volatility', 'Momentum'
    ]
    X = df[features]

    # Predict Close Price
    df['Predicted_Close'] = model.predict(X)

    # Show data and predictions
    st.subheader("Data with Predictions")
    st.dataframe(df.tail())

    # Plot actual vs predicted
    st.subheader("Actual vs Predicted Close Price")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.index, df['Close'], label='Actual Close')
    ax.plot(df.index, df['Predicted_Close'], linestyle='--', label='Predicted Close')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file to get started.")
