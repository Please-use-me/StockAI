import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# --- 1. SETUP ---
st.set_page_config(page_title="Stox Analyzer", layout="wide")
st.title("📈 Pro Stock Analyzer & ML Predictor")

ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
days_back = st.sidebar.slider("Days of Data", 365, 1000, 500)

@st.cache_data
def load_data(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    # Fix for multi-index columns in newer yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(ticker, days_back)

if not data.empty:
    # --- 2. INDICATORS ---
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA50'] = data['Close'].rolling(50).mean()
    data['Returns'] = data['Close'].pct_change()
    
    data['Signal'] = (data['MA20'] > data['MA50']).astype(float)
    data['Position'] = data['Signal'].diff()

    # --- 3. DASHBOARD ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Price & Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Close"))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name="20-Day MA"))
        
        # Plot Buy/Sell signals
        buy_signals = data[data['Position'] == 1]
        sell_signals = data[data['Position'] == -1]
        
        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['MA20'], 
                                 mode='markers', marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['MA20'], 
                                 mode='markers', marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell'))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("📊 Stats")
        # Fixed: Using .item() ensures we get a single number
        latest_price = float(data['Close'].iloc[-1])
        st.metric("Latest Price", f"${latest_price:.2f}")
        
        avg_ret = float(data['Returns'].mean() * 100)
        st.metric("Avg Daily Return", f"{avg_ret:.2f}%")
        st.dataframe(data.tail(5))

    # --- 4. EDA ---
    st.divider()
    st.subheader("🔍 Exploratory Data Analysis")
    e1, e2 = st.columns(2)
    with e1:
        fig_hist = px.histogram(x=data['Returns'].dropna(), title="Return Distribution")
        st.plotly_chart(fig_hist, width='stretch')
    with e2:
        fig_box = px.box(y=data['Returns'].dropna(), title="Volatility Range")
        st.plotly_chart(fig_box, width='stretch')

    # --- 5. ML PREDICTION ---
    st.divider()
    st.subheader("🤖 AI Price Direction Prediction")
    ml_df = data.dropna()
    ml_df['Target'] = (ml_df['Close'].shift(-1) > ml_df['Close']).astype(int)
    
    X = ml_df[['MA20', 'MA50', 'Returns']].iloc[:-1].values
    y = ml_df['Target'].iloc[:-1].values
    model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    
    pred = model.predict(ml_df[['MA20', 'MA50', 'Returns']].tail(1).values)
    if pred == 1:
        st.success("Model Predicts: **UP** tomorrow (Bullish)")
    else:
        st.error("Model Predicts: **DOWN** tomorrow (Bearish)")
else:
    st.error("Ticker not found. Please try another symbol like TSLA or MSFT.")
