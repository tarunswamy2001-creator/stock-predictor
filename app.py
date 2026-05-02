import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intra-Day Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #185FA5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #185FA5;
    }
    .author-tag {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.3rem;
    }
    .predict-box {
        background: linear-gradient(135deg, #E6F1FB, #f0f7ff);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #c8dff5;
        text-align: center;
    }
    .signal-up { color: #1D9E75; font-size: 2rem; font-weight: 700; }
    .signal-down { color: #D85A30; font-size: 2rem; font-weight: 700; }
    .signal-hold { color: #EF9F27; font-size: 2rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">📈 Intra-Day Stock Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Random Forest ML model · Indian & Global Stocks · Built by D. Tarun Swamy</p>', unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Popular Indian stocks
    popular_stocks = {
        "Reliance Industries": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "Infosys": "INFY.NS",
        "HDFC Bank": "HDFCBANK.NS",
        "ICICI Bank": "ICICIBANK.NS",
        "Wipro": "WIPRO.NS",
        "HCL Technologies": "HCLTECH.NS",
        "Bajaj Finance": "BAJFINANCE.NS",
        "Maruti Suzuki": "MARUTI.NS",
        "SBI": "SBIN.NS",
        "Asian Paints": "ASIANPAINT.NS",
        "Titan Company": "TITAN.NS",
    }

    stock_choice = st.selectbox(
        "Select Indian Stock",
        options=list(popular_stocks.keys()),
        index=2
    )
    ticker_symbol = popular_stocks[stock_choice]

    custom_ticker = st.text_input("Or enter custom ticker (e.g. TATAMOTORS.NS)", "")
    if custom_ticker.strip():
        ticker_symbol = custom_ticker.strip().upper()

    period = st.selectbox(
        "Training data period",
        ["3mo", "6mo", "1y", "2y"],
        index=2,
        help="More data = better model accuracy"
    )

    n_estimators = st.slider("Random Forest trees", 50, 300, 100, 50)
    prediction_days = st.slider("Days to show in chart", 30, 120, 60)

    st.markdown("---")
    st.markdown("**About this project**")
    st.markdown("""
    This app was built as part of my MCA final project on **Intra-Day Stock Price Prediction using Machine Learning**.

    **Model:** Random Forest Classifier  
    **Features:** RSI, MACD, Bollinger Bands, Moving Averages, Volume  
    **Target:** Next-day price direction (UP / DOWN)

    — D. Tarun Swamy, MCA  
    Vignan University, Guntur
    """)

# ── Feature Engineering ───────────────────────────────────────────────────────
def compute_features(df):
    df = df.copy()

    # Moving Averages
    df['MA_5']  = df['Close'].rolling(5).mean()
    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()

    # EMA
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / (bb_mid + 1e-10)
    df['BB_pos']   = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)

    # Price features
    df['Daily_Return'] = df['Close'].pct_change()
    df['High_Low_Pct'] = (df['High'] - df['Low']) / (df['Close'] + 1e-10)
    df['Open_Close_Pct'] = (df['Close'] - df['Open']) / (df['Open'] + 1e-10)

    # Volume
    df['Volume_MA'] = df['Volume'].rolling(10).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-10)

    # MA crossovers
    df['MA5_MA20_cross'] = df['MA_5'] - df['MA_20']
    df['MA10_MA50_cross'] = df['MA_10'] - df['MA_50']

    # Target: 1 = price goes UP next day, 0 = goes DOWN
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    return df

# ── Load & Train ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_and_train(ticker, period, n_estimators):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty or len(df) < 60:
        return None, None, None, None, None, None

    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = compute_features(df)
    df.dropna(inplace=True)

    feature_cols = [
        'MA_5','MA_10','MA_20','MA_50',
        'EMA_12','EMA_26',
        'MACD','MACD_Signal','MACD_Hist',
        'RSI',
        'BB_width','BB_pos',
        'Daily_Return','High_Low_Pct','Open_Close_Pct',
        'Volume_Ratio',
        'MA5_MA20_cross','MA10_MA50_cross'
    ]

    X = df[feature_cols].values
    y = df['Target'].values

    # Train/test split (80/20, time-based)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)

    # Next-day prediction
    latest = scaler.transform(X[-1].reshape(1, -1))
    next_pred = model.predict(latest)[0]
    next_prob = model.predict_proba(latest)[0]

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    return df, model, acc, next_pred, next_prob, feature_importance

# ── Main UI ───────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data & training model for {ticker_symbol}..."):
    result = load_and_train(ticker_symbol, period, n_estimators)

if result[0] is None:
    st.error(f"Could not fetch data for **{ticker_symbol}**. Please check the ticker symbol and try again.")
    st.stop()

df, model, acc, next_pred, next_prob, feat_imp = result

# ── Top metrics ───────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

current_price = float(df['Close'].iloc[-1])
prev_price    = float(df['Close'].iloc[-2])
price_change  = current_price - prev_price
price_pct     = (price_change / prev_price) * 100

col1.metric("Current Price", f"₹{current_price:,.2f}", f"{price_pct:+.2f}%")
col2.metric("Model Accuracy", f"{acc*100:.1f}%", "on test data")
col3.metric("RSI", f"{float(df['RSI'].iloc[-1]):.1f}", "14-day")
col4.metric("Training Samples", f"{int(len(df)*0.8)}", f"of {len(df)} days")
col5.metric("Features Used", "18", "engineered indicators")

st.markdown("---")

# ── Prediction Banner ─────────────────────────────────────────────────────────
pred_col, chart_col = st.columns([1, 2])

with pred_col:
    st.markdown("### 🔮 Tomorrow's Prediction")
    signal = "📈 BUY / UP" if next_pred == 1 else "📉 SELL / DOWN"
    signal_class = "signal-up" if next_pred == 1 else "signal-down"
    confidence = next_prob[next_pred] * 100

    st.markdown(f"""
    <div class="predict-box">
        <div class="{signal_class}">{signal}</div>
        <p style="font-size:1.1rem;margin-top:.5rem;color:#444">Confidence: <b>{confidence:.1f}%</b></p>
        <p style="font-size:.85rem;color:#888;margin-top:.3rem">P(UP) = {next_prob[1]*100:.1f}% &nbsp;|&nbsp; P(DOWN) = {next_prob[0]*100:.1f}%</p>
        <hr style="margin:.8rem 0;opacity:.3">
        <p style="font-size:.8rem;color:#aaa">⚠️ For educational purposes only.<br>Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

    # Current indicators
    st.markdown("### 📊 Current Indicators")
    rsi_val  = float(df['RSI'].iloc[-1])
    macd_val = float(df['MACD'].iloc[-1])
    bb_pos   = float(df['BB_pos'].iloc[-1])

    rsi_status  = "Overbought 🔴" if rsi_val > 70 else ("Oversold 🟢" if rsi_val < 30 else "Neutral 🟡")
    macd_status = "Bullish 🟢" if macd_val > 0 else "Bearish 🔴"
    bb_status   = "Near Upper 🔴" if bb_pos > 0.8 else ("Near Lower 🟢" if bb_pos < 0.2 else "Middle 🟡")

    st.markdown(f"**RSI ({rsi_val:.1f}):** {rsi_status}")
    st.markdown(f"**MACD ({macd_val:.3f}):** {macd_status}")
    st.markdown(f"**Bollinger Position:** {bb_status}")

with chart_col:
    st.markdown("### 📉 Price Chart with Signals")
    recent = df.tail(prediction_days).copy()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=recent.index,
        open=recent['Open'], high=recent['High'],
        low=recent['Low'],   close=recent['Close'],
        name='OHLC',
        increasing_line_color='#1D9E75',
        decreasing_line_color='#D85A30'
    ))

    # MAs
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MA_20'],
        line=dict(color='#185FA5', width=1.5), name='MA 20'))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['MA_50'],
        line=dict(color='#EF9F27', width=1.5), name='MA 50'))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_upper'],
        line=dict(color='rgba(150,150,150,0.4)', width=1, dash='dot'), name='BB Upper'))
    fig.add_trace(go.Scatter(x=recent.index, y=recent['BB_lower'],
        line=dict(color='rgba(150,150,150,0.4)', width=1, dash='dot'),
        fill='tonexty', fillcolor='rgba(180,180,180,0.07)', name='BB Lower'))

    fig.update_layout(
        height=380, xaxis_rangeslider_visible=False,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', y=1.08),
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Arial")
    )
    fig.update_xaxis(showgrid=True, gridcolor='#f0f0f0')
    fig.update_yaxis(showgrid=True, gridcolor='#f0f0f0', title="Price (₹)")

    st.plotly_chart(fig, use_container_width=True)

# ── Feature Importance + RSI ──────────────────────────────────────────────────
st.markdown("---")
fi_col, rsi_col = st.columns(2)

with fi_col:
    st.markdown("### 🧠 Feature Importance (Top 10)")
    top10 = feat_imp.head(10)
    fig2 = px.bar(top10, x='Importance', y='Feature', orientation='h',
                  color='Importance', color_continuous_scale='Blues')
    fig2.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10),
                       plot_bgcolor='white', paper_bgcolor='white',
                       coloraxis_showscale=False, yaxis=dict(autorange='reversed'))
    fig2.update_xaxis(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig2, use_container_width=True)

with rsi_col:
    st.markdown("### 📈 RSI (Last 60 Days)")
    rsi_data = df['RSI'].tail(60)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=rsi_data.index, y=rsi_data.values,
        fill='tozeroy', line=dict(color='#185FA5', width=2), name='RSI'))
    fig3.add_hline(y=70, line=dict(color='#D85A30', dash='dash', width=1.5), annotation_text='Overbought 70')
    fig3.add_hline(y=30, line=dict(color='#1D9E75', dash='dash', width=1.5), annotation_text='Oversold 30')
    fig3.update_layout(height=320, margin=dict(l=10,r=10,t=10,b=10),
                       plot_bgcolor='white', paper_bgcolor='white',
                       yaxis=dict(range=[0,100], title='RSI'))
    fig3.update_xaxis(showgrid=True, gridcolor='#f0f0f0')
    st.plotly_chart(fig3, use_container_width=True)

# ── Raw Data ──────────────────────────────────────────────────────────────────
with st.expander("📋 View Raw Data (last 30 days)"):
    display_cols = ['Open','High','Low','Close','Volume','RSI','MACD','MA_20','MA_50','BB_pos']
    st.dataframe(
        df[display_cols].tail(30).round(3).iloc[::-1],
        use_container_width=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#aaa;font-size:.8rem;'>Built by D. Tarun Swamy · MCA, Vignan University · Intra-Day Stock Prediction using Random Forest ML</p>",
    unsafe_allow_html=True
)
