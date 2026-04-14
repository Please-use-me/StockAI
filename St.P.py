import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# --- 1. SETUP & DATA FETCHING ---
st.set_page_config(page_title="Stox Analyzer", layout="wide")
st.title("📈 Pro Stock Analyzer & ML Predictor")

ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL").upper()
days_back = st.sidebar.slider("Days of Data", 365, 1000, 500)

@st.cache_data
def load_data(symbol, days):
    end = datetime.now()
    start = end - timedelta(days=days)
    df = yf.download(symbol, start=start, end=end)
    # Critical Fix: Flatten multi-index columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

data = load_data(ticker, days_back)

if not data.empty:
    # --- 2. TECHNICAL INDICATORS ---
    # .squeeze() ensures we are working with 1D series
    close_prices = data['Close'].squeeze()
    data['MA20'] = close_prices.rolling(20).mean()
    data['MA50'] = close_prices.rolling(50).mean()
    
    data['Signal'] = (data['MA20'] > data['MA50']).astype(float)
    data['Position'] = data['Signal'].diff()

    # --- 3. MAIN DASHBOARD ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Price & Crossover Signals")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=close_prices, name="Close", line=dict(color='gray', width=1)))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], name="20-Day MA", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=data.index, y=data['MA50'], name="50-Day MA", line=dict(color='orange')))
        
        # Plot Buy/Sell signals
        buy_sig = data[data['Position'] == 1]
        sell_sig = data[data['Position'] == -1]
        fig.add_trace(go.Scatter(x=buy_sig.index, y=buy_sig['MA20'], mode='markers', 
                                 marker=dict(symbol='triangle-up', size=12, color='green'), name='Buy'))
        fig.add_trace(go.Scatter(x=sell_sig.index, y=sell_sig['MA20'], mode='markers', 
                                 marker=dict(symbol='triangle-down', size=12, color='red'), name='Sell'))
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("📊 Statistics")
        # .item() extracts the pure number for the metric
        curr_price = float(close_prices.iloc[-1])
        st.metric("Latest Price", f"${curr_price:.2f}")

        returns = close_prices.pct_change().dropna()
        avg_ret = float(returns.mean() * 100)
        st.metric("Avg Daily Return", f"{avg_ret:.2f}%")
        st.write("Recent Data History", data.tail(5))

    # --- 4. EDA SECTION (Safe Charts) ---
    st.divider()
    st.subheader("🔍 Market Insights")
    e1, e2 = st.columns(2)
    
    with e1:
        # Pass values directly to fix 1D-dimensional error
        fig_hist = px.histogram(x=returns.values.flatten(), title="Return Distribution", labels={'x': 'Daily Return'})
        st.plotly_chart(fig_hist, width='stretch')
    with e2:
        fig_box = px.box(y=returns.values.flatten(), title="Volatility Range", labels={'y': 'Daily Return'})
        st.plotly_chart(fig_box, width='stretch')

    # --- 5. ML PREDICTION ---
    st.divider()
    st.subheader("🤖 AI Price Direction Prediction")
    
    ml_df = data.copy().dropna()
    ml_df['Target'] = (ml_df['Close'].squeeze().shift(-1) > ml_df['Close'].squeeze()).astype(int)
    
    # Force 2D features and 1D target
    X = ml_df[['MA20', 'MA50']].iloc[:-1].values
    y = ml_df['Target'].iloc[:-1].values.ravel()
    
    if len(X) > 30:
        model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
        last_feat = ml_df[['MA20', 'MA50']].tail(1).values
        pred = model.predict(last_feat)[0]
        
        if pred == 1:
            st.success(f"Model Predicts: **UP** tomorrow for {ticker} (Bullish)")
        else:
            st.error(f"Model Predicts: **DOWN** tomorrow for {ticker} (Bearish)")
    else:
        st.info("Insufficient data for a reliable AI prediction.")
else:
    st.error("Invalid Ticker. Please check the symbol and try again.")
