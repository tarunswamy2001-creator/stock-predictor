# 📈 Intra-Day Stock Price Predictor

A Machine Learning web app that predicts next-day stock price direction (UP/DOWN) for Indian & global stocks using a **Random Forest Classifier**.

Built by **D. Tarun Swamy** — MCA, Vignan University, Guntur (2026)

## 🔗 Live Demo
> _[Add your Streamlit Cloud URL here after deployment]_

---

## 🧠 How It Works

1. Fetches real-time stock data using **Yahoo Finance API** (`yfinance`)
2. Engineers **18 technical indicators** as features:
   - Moving Averages (MA 5, 10, 20, 50)
   - RSI (Relative Strength Index)
   - MACD & Signal Line
   - Bollinger Bands (width & position)
   - Volume ratio, daily returns, price ranges
3. Trains a **Random Forest Classifier** on 80% of historical data
4. Predicts **next-day direction** (UP=1 / DOWN=0) with confidence score

## 📊 Features
- 🏦 12 popular Indian stocks (NSE) pre-loaded
- 🔮 Next-day prediction with confidence %
- 📉 Interactive candlestick chart with MA & Bollinger Bands
- 📊 RSI chart with overbought/oversold zones
- 🧠 Feature importance visualization
- 📋 Raw data table (last 30 days)

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python | Core language |
| Scikit-learn | Random Forest model |
| yfinance | Stock data (NSE/BSE) |
| Streamlit | Web app UI |
| Plotly | Interactive charts |
| Pandas / NumPy | Data processing |

## 🚀 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/stock-predictor
cd stock-predictor
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Disclaimer
This app is built for **educational purposes only** as part of an academic ML project. It does not constitute financial advice.

---
*MCA Final Project · Vignan University · 2025–2026*
