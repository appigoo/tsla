import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
import ta  # 技術指標庫 (pip install ta)

# Streamlit 頁面設置
st.set_page_config(page_title="TSLA 5分鐘交易策略監控", layout="wide")
st.title("TSLA 5分鐘K線實時監控與交易策略")
st.write("基於EMA、RSI、VWAP、布林帶、OBV的日內交易策略（資金：1萬美元，中度風險偏好）")

# 模擬或實時數據獲取
@st.cache_data(ttl=300)  # 快取 5 分鐘
def fetch_data(symbol="TSLA", interval="5m", period="1d"):
    try:
        # 使用 yfinance 獲取 TSLA 5分鐘數據
        df = yf.download(symbol, interval=interval, period=period, prepost=False)
        if df.empty:
            raise ValueError("無數據返回")
        return df
    except Exception as e:
        st.error(f"數據獲取失敗: {e}")
        # 模擬數據作為備用
        dates = pd.date_range(end=datetime.now(), periods=100, freq="5min")
        np.random.seed(42)
        prices = np.random.normal(319.04, 3, 100)  # 模擬 TSLA 價格（均值319.04）
        volumes = np.random.randint(10000, 100000, 100)
        df = pd.DataFrame({
            "Open": prices,
            "High": prices + np.random.uniform(0, 1, 100),
            "Low": prices - np.random.uniform(0, 1, 100),
            "Close": prices,
            "Volume": volumes
        }, index=dates)
        return df

# 計算技術指標
def calculate_indicators(df):
    # EMA
    df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    
    # VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Price_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_PV'] = df['Price_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_PV'] / df['Cumulative_Volume']
    
    # 布林帶
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    # OBV
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # 成交量均線（確認成交量趨勢）
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    
    return df

# 生成交易策略
def generate_trading_signal(df, capital=10000, risk_per_trade=0.015):
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    signal = "無交易信號"
    strategy = None
    entry_price = stop_loss = take_profit = risk = reward = rr_ratio = shares = None
    
    # 資金與風險管理
    position_size = capital * 0.2  # 每筆交易使用20%資金（2000美元）
    risk_per_share = latest['Close'] * risk_per_trade  # 每股風險（約1.5%）
    
    # 策略1：VWAP 錨定趨勢策略
    if (latest['Close'] > latest['VWAP'] and 
        prev['EMA5'] <= prev['EMA20'] and 
        latest['EMA5'] > latest['EMA20'] and 
        latest['RSI'] < 70 and 
        latest['OBV'] > prev['OBV'] and 
        latest['Volume'] > latest['Volume_MA5']):
        signal = "做多"
        strategy = "VWAP 錨定趨勢策略"
        entry_price = latest['Close']
        stop_loss = max(latest['VWAP'], latest['Close'] - 2)  # 止損設在 VWAP 或下方2美元
        take_profit = latest['Close'] + 3.5  # 止盈設為3.5美元
        shares = int(position_size / entry_price)
        risk = shares * (entry_price - stop_loss)
        reward = shares * (take_profit - entry_price)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (latest['Close'] < latest['VWAP'] and 
          prev['EMA5'] >= prev['EMA20'] and 
          latest['EMA5'] < latest['EMA20'] and 
          latest['RSI'] > 30 and 
          latest['OBV'] < prev['OBV'] and 
          latest['Volume'] > latest['Volume_MA5']):
        signal = "做空"
        strategy = "VWAP 錨定趨勢策略"
        entry_price = latest['Close']
        stop_loss = min(latest['VWAP'], latest['Close'] + 2)  # 止損設在 VWAP 或上方2美元
        take_profit = latest['Close'] - 3.5  # 止盈設為3.5美元
        shares = int(position_size / entry_price)
        risk = shares * (stop_loss - entry_price)
        reward = shares * (entry_price - take_profit)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    # 策略2：布林帶突破策略
    elif (latest['Close'] > latest['BB_Upper'] and 
          latest['RSI'] < 70 and 
          latest['Volume'] > latest['Volume_MA5'] and 
          latest['OBV'] > prev['OBV']):
        signal = "做多"
        strategy = "布林帶突破策略"
        entry_price = latest['Close']
        stop_loss = latest['BB_Mid']  # 止損設在布林帶中軌
        take_profit = latest['Close'] + (latest['Close'] - latest['BB_Mid'])  # 止盈設為突破距離
        shares = int(position_size / entry_price)
        risk = shares * (entry_price - stop_loss)
        reward = shares * (take_profit - entry_price)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (latest['Close'] < latest['BB_Lower'] and 
          latest['RSI'] > 30 and 
          latest['Volume'] > latest['Volume_MA5'] and 
          latest['OBV'] < prev['OBV']):
        signal = "做空"
        strategy = "布林帶突破策略"
        entry_price = latest['Close']
        stop_loss = latest['BB_Mid']  # 止損設在布林帶中軌
        take_profit = latest['Close'] - (latest['BB_Mid'] - latest['Close'])  # 止盈設為突破距離
        shares = int(position_size / entry_price)
        risk = shares * (stop_loss - entry_price)
        reward = shares * (entry_price - take_profit)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    # 策略3：RSI 反轉策略
    elif (latest['RSI'] > 70 and 
          latest['Close'] > latest['BB_Upper'] and 
          latest['OBV'] < prev['OBV']):
        signal = "做空"
        strategy = "RSI 反轉策略"
        entry_price = latest['Close']
        stop_loss = latest['Close'] + 1.5  # 止損設在高點上方1.5美元
        take_profit = latest['BB_Mid']  # 止盈設為布林帶中軌
        shares = int(position_size / entry_price)
        risk = shares * (stop_loss - entry_price)
        reward = shares * (entry_price - take_profit)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (latest['RSI'] < 30 and 
          latest['Close'] < latest['BB_Lower'] and 
          latest['OBV'] > prev['OBV']):
        signal = "做多"
        strategy = "RSI 反轉策略"
        entry_price = latest['Close']
        stop_loss = latest['Close'] - 1.5  # 止損設在低點下方1.5美元
        take_profit = latest['BB_Mid']  # 止盈設為布林帶中軌
        shares = int(position_size / entry_price)
        risk = shares * (entry_price - stop_loss)
        reward = shares * (take_profit - entry_price)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    return {
        "signal": signal,
        "strategy": strategy,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk": risk,
        "reward": reward,
        "rr_ratio": rr_ratio,
        "shares": shares
    }

# 繪製 K 線圖與指標
def plot_data(df):
    fig = go.Figure()
    
    # K 線圖
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="K線"
    ))
    
    # EMA
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20", line=dict(color='orange')))
    
    # VWAP
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name="VWAP", line=dict(color='purple', dash='dash')))
    
    # 布林帶
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="布林上軌", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="布林下軌", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="布林中軌", line=dict(color='gray', dash='dot')))
    
    # 成交量
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量", yaxis="y2", opacity=0.3))
    
    # OBV
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV", yaxis="y3", line=dict(color='green')))
    
    fig.update_layout(
        title="TSLA 5分鐘K線與技術指標",
        xaxis_title="時間",
        yaxis_title="價格 (美元)",
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="OBV", anchor="free", overlaying="y", side="right", position=0.95, showgrid=False),
        height=800
    )
    return fig

# 主程式：實時更新
def main():
    st.sidebar.header("設置")
    symbol = st.sidebar.text_input("股票代碼", "TSLA")
    refresh_interval = st.sidebar.slider("更新間隔（秒）", 60, 600, 300)  # 預設 5 分鐘
    
    placeholder = st.empty()  # 用於動態更新
    
    while True:
        with placeholder.container():
            # 獲取數據
            df = fetch_data(symbol)
            df = calculate_indicators(df)
            
            # 顯示最新價格與指標
            latest = df.iloc[-1]
            st.subheader(f"當前價格 (TSLA): {latest['Close']:.2f} 美元")
            st.write(f"時間: {df.index[-1]}")
            col1, col2, col3 = st.columns(3)
            col1.metric("EMA5", f"{latest['EMA5']:.2f}")
            col1.metric("EMA20", f"{latest['EMA20']:.2f}")
            col2.metric("VWAP", f"{latest['VWAP']:.2f}")
            col2.metric("RSI", f"{latest['RSI']:.2f}")
            col3.metric("布林上軌", f"{latest['BB_Upper']:.2f}")
            col3.metric("布林下軌", f"{latest['BB_Lower']:.2f}")
            col3.metric("OBV", f"{latest['OBV']:.0f}")
            
            # 生成交易信號
            signal = generate_trading_signal(df)
            st.subheader("交易策略建議")
            if signal['signal'] != "無交易信號":
                st.write(f"**策略**: {signal['strategy']}")
                st.write(f"**信號**: {signal['signal']}")
                st.write(f"**入場價**: {signal['entry_price']:.2f} 美元")
                st.write(f"**止損價**: {signal['stop_loss']:.2f} 美元")
                st.write(f"**止盈價**: {signal['take_profit']:.2f} 美元")
                st.write(f"**持倉股數**: {signal['shares']}")
                st.write(f"**風險**: {abs(signal['risk']):.2f} 美元")
                st.write(f"**潛在盈利**: {signal['reward']:.2f} 美元")
                st.write(f"**風險回報比**: {signal['rr_ratio']:.2f}")
            else:
                st.write("當前無明確交易信號，請等待機會。")
            
            # 顯示圖表
            st.plotly_chart(plot_data(df), use_container_width=True)
            
            # 下次更新時間
            st.write(f"下次更新於 {datetime.now() + timedelta(seconds=refresh_interval)}")
        
        # 等待下一次更新
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
