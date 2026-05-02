import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

st.title("📈 Intra-Day Stock Price Predictor")
st.caption("Random Forest ML Model · Indian Stocks (NSE) · Built by D. Tarun Swamy, MCA - Vignan University")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    stocks = {
        "Infosys": "INFY.NS",
        "TCS": "TCS.NS",
        "Reliance": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Wipro": "WIPRO.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "SBI": "SBIN.NS",
        "HCL Tech": "HCLTECH.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Asian Paints": "ASIANPAINT.NS",
    }
    stock_name = st.selectbox("Select Stock", list(stocks.keys()))
    ticker = stocks[stock_name]
    custom = st.text_input("Or enter custom (e.g. TATAMOTORS.NS)")
    if custom:
        ticker = custom.upper()
    period = st.selectbox("Data period", ["6mo", "1y", "2y"], index=1)
    n_trees = st.slider("Number of trees", 50, 200, 100, 50)
    st.markdown("---")
    st.markdown("**MCA Final Project**")
    st.markdown("Intra-Day Stock Prediction using Random Forest ML")
    st.markdown("*D. Tarun Swamy, Vignan University*")

@st.cache_data(ttl=3600)
def get_data_and_predict(ticker, period, n_trees):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_pos'] = (df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-9)

    df['Return']   = df['Close'].pct_change()
    df['HL_range'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['Vol_ratio'] = df['Volume'] / (df['Volume'].rolling(10).mean() + 1e-9)

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['MA5','MA20','MA50','RSI','MACD','BB_pos','Return','HL_range','Vol_ratio']
    X = df[features].values
    y = df['Target'].values

    split = int(len(X) * 0.8)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split])
    X_test  = scaler.transform(X[split:])

    model = RandomForestClassifier(n_estimators=n_trees, max_depth=6, random_state=42, class_weight='balanced')
    model.fit(X_train, y[:split])

    acc = accuracy_score(y[split:], model.predict(X_test))
    next_pred = model.predict(scaler.transform(X[-1:]))[0]
    next_prob = model.predict_proba(scaler.transform(X[-1:]))[0]

    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)

    return {'df': df, 'acc': acc, 'pred': next_pred, 'prob': next_prob, 'fi': fi}

with st.spinner(f"Loading {stock_name} data and training model..."):
    result = get_data_and_predict(ticker, period, n_trees)

if result is None:
    st.error("Could not fetch data. Check the ticker and try again.")
    st.stop()

df   = result['df']
acc  = result['acc']
pred = result['pred']
prob = result['prob']
fi   = result['fi']

price   = float(df['Close'].iloc[-1])
prev    = float(df['Close'].iloc[-2])
change  = ((price - prev) / prev) * 100

# Metrics row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"₹{price:,.2f}", f"{change:+.2f}%")
c2.metric("Model Accuracy", f"{acc*100:.1f}%", "on test set")
c3.metric("RSI", f"{float(df['RSI'].iloc[-1]):.1f}")
c4.metric("Data Points", str(len(df)))

st.markdown("---")

# Prediction + Chart
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🔮 Tomorrow's Prediction")
    if pred == 1:
        st.success(f"### 📈 BUY / UP")
    else:
        st.error(f"### 📉 SELL / DOWN")

    st.metric("Confidence", f"{prob[pred]*100:.1f}%")
    st.write(f"P(UP) = **{prob[1]*100:.1f}%** | P(DOWN) = **{prob[0]*100:.1f}%**")
    st.caption("⚠️ Educational purposes only. Not financial advice.")

    st.markdown("---")
    st.subheader("📊 Indicators")
    rsi = float(df['RSI'].iloc[-1])
    macd = float(df['MACD'].iloc[-1])
    bb = float(df['BB_pos'].iloc[-1])
    st.write(f"**RSI ({rsi:.1f}):** {'🔴 Overbought' if rsi>70 else ('🟢 Oversold' if rsi<30 else '🟡 Neutral')}")
    st.write(f"**MACD ({macd:.2f}):** {'🟢 Bullish' if macd>0 else '🔴 Bearish'}")
    st.write(f"**Bollinger:** {'🔴 Near Upper' if bb>0.8 else ('🟢 Near Lower' if bb<0.2 else '🟡 Middle')}")

with col2:
    st.subheader("📉 Price Chart")
    recent = df.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent['Open'], high=recent['High'],
        low=recent['Low'],   close=recent['Close'],
        name='Price',
        increasing_line_color='#00b09b',
        decreasing_line_color='#ff4b4b'
    ))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MA20'], line=dict(color='#185FA5', width=1.5), name='MA20'))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MA50'], line=dict(color='#EF9F27', width=1.5), name='MA50'))
    fig.update_layout(height=400, xaxis_rangeslider_visible=False, plot_bgcolor='white', paper_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Feature importance
col3, col4 = st.columns(2)

with col3:
    st.subheader("🧠 Feature Importance")
    fig2 = go.Figure(go.Bar(
        x=fi['Importance'], y=fi['Feature'],
        orientation='h', marker_color='#185FA5'
    ))
    fig2.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white', yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    st.subheader("📈 RSI (Last 60 Days)")
    rsi_series = df['RSI'].tail(60)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=rsi_series.index, y=rsi_series.values, fill='tozeroy', line=dict(color='#185FA5')))
    fig3.add_hline(y=70, line=dict(color='red',  dash='dash'))
    fig3.add_hline(y=30, line=dict(color='green', dash='dash'))
    fig3.update_layout(height=300, plot_bgcolor='white', paper_bgcolor='white', yaxis=dict(range=[0,100]))
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("📋 Raw Data (last 20 days)"):
    st.dataframe(df[['Open','High','Low','Close','Volume','RSI','MACD','MA20']].tail(20).round(2).iloc[::-1], use_container_width=True)

st.markdown("---")
st.caption("Built by D. Tarun Swamy · MCA, Vignan University, Guntur · Intra-Day Stock Prediction using Random Forest")
