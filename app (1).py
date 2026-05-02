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
st.caption("Random Forest ML · Indian Stocks · D. Tarun Swamy, MCA, Vignan University")
st.markdown("---")

with st.sidebar:
    st.header("Settings")
    stocks = {
        "Infosys": "INFY.NS",
        "TCS": "TCS.NS",
        "Reliance": "RELIANCE.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "Wipro": "WIPRO.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "SBI": "SBIN.NS",
        "HCL Tech": "HCLTECH.NS",
    }
    name = st.selectbox("Select Stock", list(stocks.keys()))
    ticker = stocks[name]
    custom = st.text_input("Custom ticker (e.g. TATAMOTORS.NS)")
    if custom:
        ticker = custom.upper()
    period = st.selectbox("Period", ["6mo", "1y", "2y"], index=1)
    n = st.slider("Trees", 50, 200, 100, 50)
    st.markdown("---")
    st.info("MCA Final Project\nIntra-Day Stock Prediction\nD. Tarun Swamy\nVignan University")

@st.cache_data(ttl=3600)
def run(ticker, period, n):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['MA5'] = df['Close'].rolling(5).mean()
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
    df['BB'] = (df['Close'] - bb_mid) / (bb_std + 1e-9)
    df['Ret'] = df['Close'].pct_change()
    df['HL'] = (df['High'] - df['Low']) / (df['Close'] + 1e-9)
    df['Vol'] = df['Volume'] / (df['Volume'].rolling(10).mean() + 1e-9)
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)
    feats = ['MA5','MA20','MA50','RSI','MACD','BB','Ret','HL','Vol']
    X = df[feats].values
    y = df['Target'].values
    split = int(len(X) * 0.8)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X[:split])
    Xte = sc.transform(X[split:])
    m = RandomForestClassifier(n_estimators=n, max_depth=6, random_state=42)
    m.fit(Xtr, y[:split])
    acc = accuracy_score(y[split:], m.predict(Xte))
    pred = int(m.predict(sc.transform(X[-1:]))[0])
    prob = m.predict_proba(sc.transform(X[-1:]))[0]
    fi = pd.DataFrame({'Feature': feats, 'Importance': m.feature_importances_}).sort_values('Importance', ascending=False)
    return dict(df=df, acc=acc, pred=pred, prob=prob, fi=fi)

with st.spinner("Fetching data and training..."):
    res = run(ticker, period, n)

if res is None:
    st.error("Could not load data. Check ticker symbol.")
    st.stop()

df = res['df']
price = float(df['Close'].iloc[-1])
prev = float(df['Close'].iloc[-2])
chg = (price - prev) / prev * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Price", f"₹{price:,.2f}", f"{chg:+.2f}%")
c2.metric("Accuracy", f"{res['acc']*100:.1f}%")
c3.metric("RSI", f"{float(df['RSI'].iloc[-1]):.1f}")
c4.metric("Samples", str(len(df)))

st.markdown("---")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Tomorrow's Prediction")
    if res['pred'] == 1:
        st.success("## 📈 BUY / UP")
    else:
        st.error("## 📉 SELL / DOWN")
    st.metric("Confidence", f"{res['prob'][res['pred']]*100:.1f}%")
    st.write(f"P(UP) = **{res['prob'][1]*100:.1f}%**")
    st.write(f"P(DOWN) = **{res['prob'][0]*100:.1f}%**")
    st.caption("Educational only. Not financial advice.")
    st.markdown("---")
    rsi = float(df['RSI'].iloc[-1])
    macd = float(df['MACD'].iloc[-1])
    st.write("**RSI:**", "🔴 Overbought" if rsi > 70 else ("🟢 Oversold" if rsi < 30 else "🟡 Neutral"))
    st.write("**MACD:**", "🟢 Bullish" if macd > 0 else "🔴 Bearish")

with col2:
    st.subheader("Price Chart (Last 60 Days)")
    r = df.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=r.index,
        open=r['Open'], high=r['High'],
        low=r['Low'], close=r['Close'],
        increasing_line_color='green',
        decreasing_line_color='red',
        name='Price'
    ))
    fig.add_trace(go.Scatter(x=r.index, y=r['MA20'], line=dict(color='blue', width=1), name='MA20'))
    fig.add_trace(go.Scatter(x=r.index, y=r['MA50'], line=dict(color='orange', width=1), name='MA50'))
    fig.update_layout(height=380, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.subheader("Feature Importance")
    fig2 = go.Figure(go.Bar(
        x=res['fi']['Importance'],
        y=res['fi']['Feature'],
        orientation='h',
        marker_color='steelblue'
    ))
    fig2.update_layout(height=280, yaxis=dict(autorange='reversed'))
    st.plotly_chart(fig2, use_container_width=True)

with col4:
    st.subheader("RSI Chart")
    rsi_s = df['RSI'].tail(60)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s.values, fill='tozeroy', line=dict(color='steelblue')))
    fig3.add_hline(y=70, line=dict(color='red', dash='dash'))
    fig3.add_hline(y=30, line=dict(color='green', dash='dash'))
    fig3.update_layout(height=280, yaxis=dict(range=[0, 100]))
    st.plotly_chart(fig3, use_container_width=True)

with st.expander("Raw Data"):
    st.dataframe(df[['Open','High','Low','Close','Volume','RSI','MACD']].tail(20).round(2).iloc[::-1], use_container_width=True)

st.caption("D. Tarun Swamy · MCA · Vignan University · Random Forest Stock Predictor")
