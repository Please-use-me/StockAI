import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# --- 1. SETUP & DATA FETCHING ---
st.set_page_config(page_title="All-in-One Stock App", layout="wide")
st.title("📈 Pro Stock Analyzer & ML Predictor")

ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
days_back = st.sidebar.slider("Days of Data", 365, 1000, 500)

@st.cache_data
def load_data(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    return df

data = load_data(ticker, days_back)

if not data.empty:
    # --- 2. TECHNICAL INDICATORS & SIGNALS ---
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    
    data['Signal'] = 0.0
    data['Signal'] = (data['MA20'] > data['MA50']).astype(float)
    data['Position'] = data['Signal'].diff()

    # --- 3. MAIN DASHBOARD ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price & Crossover Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'].squeeze(), name="Close", line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'].squeeze(), name="20-Day MA", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'].squeeze(), name="50-Day MA", line=dict(color='orange')))
        
        # Plot Buy/Sell signals
        fig.add_trace(go.Scatter(x=data[data['Position'] == 1].index, y=data['MA20'][data['Position'] == 1], 
                                 mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy'))
        fig.add_trace(go.Scatter(x=data[data['Position'] == -1].index, y=data['MA20'][data['Position'] == -1], 
                                 mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 Statistics")
        latest_price = float(data['Close'].iloc[-1].item())
        st.metric("Latest Price", f"${latest_price:.2f}")

        avg_return = float((data['Close'].pct_change().mean() * 100).item())

                
        daily_ret = data['Close'].pct_change().dropna()
        avg_return = float(daily_ret.mean().item() * 100)
        st.metric("Avg Daily Return", f"{avg_return:.2f}%")


        st.write(data.tail(5))

             # --- 4. EDA SECTION (Simplified to fix errors) ---
    st.divider()
    st.subheader("🔍 Market Overview")
    
    # We display a simple table instead of the broken charts
    st.write("Recent Daily Returns (%)")
    display_returns = (data['Close'].pct_change() * 100).dropna().tail(10)
    st.dataframe(display_returns)

    # --- 5. ML PREDICTION ---
    st.divider()
    st.subheader("🤖 AI Price Direction Prediction")
    
    # Clean data for ML
    ml_df = data.copy().dropna()
    # Ensure we have 1D arrays for the target
    ml_df['Target'] = (ml_df['Close'].shift(-1) > ml_df['Close']).astype(int)
    
    # Force features to be a clean 2D array and target to be 1D
    X = ml_df[['MA20', 'MA50']].iloc[:-1].values
    y = ml_df['Target'].iloc[:-1].values.ravel()
    
    if len(X) > 0:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Predict for the next day
        last_features = ml_df[['MA20', 'MA50']].tail(1).values
        pred = model.predict(last_features)[0]
        
        if pred == 1:
            st.success("Model Predicts: **UP** (Bullish Movement)")
        else:
            st.error("Model Predicts: **DOWN** (Bearish Movement)")
    else:
        st.info("Not enough data for AI prediction yet.")
else:
    st.warning("Please enter a valid stock ticker symbol.")
