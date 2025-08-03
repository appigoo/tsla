import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime, timedelta
import time
import ta
import smtplib
from email.mime.text import MIMEText

# Streamlit 頁面設置
st.set_page_config(page_title="TSLA 5分鐘交易策略監控與回測", layout="wide")
st.title("TSLA 5分鐘K線實時監控與回測")
st.write("基於EMA、RSI、VWAP、布林帶、OBV的日內交易策略（資金：1萬美元，中度風險偏好）")

# 初始化 session state
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'last_signal' not in st.session_state:
    st.session_state.last_signal = None

# 獲取歷史數據
def fetch_historical_data(symbol="TSLA", interval="5m", period="5d"):
    try:
        df = yf.download(symbol, interval=interval, period=period, prepost=False)
        if df.empty:
            raise ValueError("無數據返回，請檢查股票代碼或網路連線")
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"數據獲取失敗: {e}. 請檢查網路或 yfinance API 限制")
        return None

# 計算技術指標
def calculate_indicators(df):
    try:
        close = df['Close'].to_numpy().flatten()
        volume = df['Volume'].to_numpy().flatten()
        
        df['EMA5'] = ta.trend.ema_indicator(pd.Series(close), window=5)
        df['EMA20'] = ta.trend.ema_indicator(pd.Series(close), window=20)
        df['RSI'] = ta.momentum.rsi(pd.Series(close), window=14)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        price_volume = typical_price * df['Volume']
        cumulative_pv = price_volume.cumsum()
        cumulative_volume = df['Volume'].cumsum()
        df['VWAP'] = cumulative_pv / cumulative_volume
        bollinger = ta.volatility.BollingerBands(pd.Series(close), window=20, window_dev=2)
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['OBV'] = ta.volume.on_balance_volume(pd.Series(close), pd.Series(volume))
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        
        return df
    except Exception as e:
        st.error(f"指標計算失敗: {e}")
        return None

# 發送 email 警報
def send_email_alert(signal, email_config):
    if not email_config['enabled']:
        return
    try:
        msg = MIMEText(
            f"策略: {signal['strategy']}\n"
            f"信號: {signal['signal']}\n"
            f"入場價: {signal['entry_price']:.2f} 美元\n"
            f"止損價: {signal['stop_loss']:.2f} 美元\n"
            f"止盈價: {signal['take_profit']:.2f} 美元\n"
            f"持倉股數: {signal['shares']}\n"
            f"風險: {abs(signal['risk']):.2f} 美元\n"
            f"潛在盈利: {signal['reward']:.2f} 美元\n"
            f"風險回報比: {signal['rr_ratio']:.2f}\n"
            f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        msg['Subject'] = f"TSLA 交易信號: {signal['signal']} ({signal['strategy']})"
        msg['From'] = email_config['sender']
        msg['To'] = email_config['receiver']
        
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_config['sender'], email_config['password'])
            server.sendmail(email_config['sender'], email_config['receiver'], msg.as_string())
        st.success("已發送 email 警報")
    except Exception as e:
        st.error(f"Email 發送失敗: {e}")

# 生成實時交易信號
def generate_trading_signal(df_5m, df_15m, capital=10000, risk_per_trade=0.015):
    if len(df_5m) < 2 or len(df_15m) < 1:
        return {"signal": "無交易信號", "strategy": None, "timestamp": datetime.now()}
    
    latest_5m = df_5m.iloc[-1]
    prev_5m = df_5m.iloc[-2]
    latest_15m_idx = df_15m.index[df_15m.index <= df_5m.index[-1]].max()
    latest_15m = df_15m.loc[latest_15m_idx]
    
    signal = "無交易信號"
    strategy = None
    entry_price = stop_loss = take_profit = risk = reward = rr_ratio = shares = None
    
    position_size = capital * 0.2
    trend_15m = "多頭" if latest_15m['EMA5'].item() > latest_15m['EMA20'].item() else "空頭"
    rsi_15m_filter = latest_15m['RSI'].item() < 70 if trend_15m == "多頭" else latest_15m['RSI'].item() > 30
    
    if (trend_15m == "多頭" and rsi_15m_filter and
        latest_5m['Close'] > latest_5m['VWAP'] and 
        prev_5m['EMA5'] <= prev_5m['EMA20'] and 
        latest_5m['EMA5'] > latest_5m['EMA20'] and 
        latest_5m['RSI'] < 70 and 
        latest_5m['OBV'] > prev_5m['OBV'] and 
        latest_5m['Volume'] > latest_5m['Volume_MA5']):
        signal = "做多"
        strategy = "VWAP 錨定趨勢策略"
        entry_price = latest_5m['Close']
        stop_loss = max(latest_5m['VWAP'], latest_5m['Close'] - 2)
        take_profit = latest_5m['Close'] + 3.5
        shares = int(position_size / entry_price)
        risk = shares * (entry_price - stop_loss)
        reward = shares * (take_profit - entry_price)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (trend_15m == "空頭" and rsi_15m_filter and
          latest_5m['Close'] < latest_5m['VWAP'] and 
          prev_5m['EMA5'] >= prev_5m['EMA20'] and 
          latest_5m['EMA5'] < latest_5m['EMA20'] and 
          latest_5m['RSI'] > 30 and 
          latest_5m['OBV'] < prev_5m['OBV'] and 
          latest_5m['Volume'] > latest_5m['Volume_MA5']):
        signal = "做空"
        strategy = "VWAP 錨定趨勢策略"
        entry_price = latest_5m['Close']
        stop_loss = min(latest_5m['VWAP'], latest_5m['Close'] + 2)
        take_profit = latest_5m['Close'] - 3.5
        shares = int(position_size / entry_price)
        risk = shares * (stop_loss - entry_price)
        reward = shares * (entry_price - take_profit)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (trend_15m == "多頭" and rsi_15m_filter and
          latest_5m['Close'] > latest_5m['BB_Upper'] and 
          latest_5m['RSI'] < 70 and 
          latest_5m['Volume'] > latest_5m['Volume_MA5'] and 
          latest_5m['OBV'] > prev_5m['OBV']):
        signal = "做多"
        strategy = "布林帶突破策略"
        entry_price = latest_5m['Close']
        stop_loss = latest_5m['BB_Mid']
        take_profit = latest_5m['Close'] + (latest_5m['Close'] - latest_5m['BB_Mid'])
        shares = int(position_size / entry_price)
        risk = shares * (entry_price - stop_loss)
        reward = shares * (take_profit - entry_price)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (trend_15m == "空頭" and rsi_15m_filter and
          latest_5m['Close'] < latest_5m['BB_Lower'] and 
          latest_5m['RSI'] > 30 and 
          latest_5m['Volume'] > latest_5m['Volume_MA5'] and 
          latest_5m['OBV'] < prev_5m['OBV']):
        signal = "做空"
        strategy = "布林帶突破策略"
        entry_price = latest_5m['Close']
        stop_loss = latest_5m['BB_Mid']
        take_profit = latest_5m['Close'] - (latest_5m['BB_Mid'] - latest_5m['Close'])
        shares = int(position_size / entry_price)
        risk = shares * (stop_loss - entry_price)
        reward = shares * (entry_price - take_profit)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (latest_5m['RSI'] > 70 and 
          latest_5m['Close'] > latest_5m['BB_Upper'] and 
          latest_5m['OBV'] < prev_5m['OBV'] and 
          latest_15m['RSI'].item() > 60):
        signal = "做空"
        strategy = "RSI 反轉策略"
        entry_price = latest_5m['Close']
        stop_loss = latest_5m['Close'] + 1.5
        take_profit = latest_5m['BB_Mid']
        shares = int(position_size / entry_price)
        risk = shares * (stop_loss - entry_price)
        reward = shares * (entry_price - take_profit)
        rr_ratio = reward / abs(risk) if risk != 0 else float('inf')
    
    elif (latest_5m['RSI'] < 30 and 
          latest_5m['Close'] < latest_5m['BB_Lower'] and 
          latest_5m['OBV'] > prev_5m['OBV'] and 
          latest_15m['RSI'].item() < 40):
        signal = "做多"
        strategy = "RSI 反轉策略"
        entry_price = latest_5m['Close']
        stop_loss = latest_5m['Close'] - 1.5
        take_profit = latest_5m['BB_Mid']
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
        "shares": shares,
        "timestamp": datetime.now()
    }

# 回測策略
def backtest_strategy(df_5m, df_15m, capital=10000, risk_per_trade=0.015, commission=5):
    if len(df_5m) < 21 or len(df_15m) < 21:
        st.error("數據不足以計算指標（需要至少21根K線）")
        return [], [capital]
    
    trades = []
    equity = [capital]
    position_size = capital * 0.2
    active_trade = False
    
    for i in range(21, len(df_5m) - 1):
        latest_5m = df_5m.iloc[i]
        prev_5m = df_5m.iloc[i - 1]
        latest_15m_idx = df_15m.index[df_15m.index <= df_5m.index[i]].max()
        latest_15m = df_15m.loc[latest_15m_idx]
        
        if active_trade:
            continue
        
        trend_15m = "多頭" if latest_15m['EMA5'].item() > latest_15m['EMA20'].item() else "空頭"
        rsi_15m_filter = latest_15m['RSI'].item() < 70 if trend_15m == "多頭" else latest_15m['RSI'].item() > 30
        
        signal = "無交易信號"
        strategy = None
        entry_price = stop_loss = take_profit = shares = None
        
        if (trend_15m == "多頭" and rsi_15m_filter and
            latest_5m['Close'] > latest_5m['VWAP'] and 
            prev_5m['EMA5'] <= prev_5m['EMA20'] and 
            latest_5m['EMA5'] > latest_5m['EMA20'] and 
            latest_5m['RSI'] < 70 and 
            latest_5m['OBV'] > prev_5m['OBV'] and 
            latest_5m['Volume'] > latest_5m['Volume_MA5']):
            signal = "做多"
            strategy = "VWAP 錨定趨勢策略"
            entry_price = latest_5m['Close']
            stop_loss = max(latest_5m['VWAP'], latest_5m['Close'] - 2)
            take_profit = latest_5m['Close'] + 3.5
            shares = int(position_size / entry_price)
        
        elif (trend_15m == "空頭" and rsi_15m_filter and
              latest_5m['Close'] < latest_5m['VWAP'] and 
              prev_5m['EMA5'] >= prev_5m['EMA20'] and 
              latest_5m['EMA5'] < latest_5m['EMA20'] and 
              latest_5m['RSI'] > 30 and 
              latest_5m['OBV'] < prev_5m['OBV'] and 
              latest_5m['Volume'] > latest_5m['Volume_MA5']):
            signal = "做空"
            strategy = "VWAP 錨定趨勢策略"
            entry_price = latest_5m['Close']
            stop_loss = min(latest_5m['VWAP'], latest_5m['Close'] + 2)
            take_profit = latest_5m['Close'] - 3.5
            shares = int(position_size / entry_price)
        
        elif (trend_15m == "多頭" and rsi_15m_filter and
              latest_5m['Close'] > latest_5m['BB_Upper'] and 
              latest_5m['RSI'] < 70 and 
              latest_5m['Volume'] > latest_5m['Volume_MA5'] and 
              latest_5m['OBV'] > prev_5m['OBV']):
            signal = "做多"
            strategy = "布林帶突破策略"
            entry_price = latest_5m['Close']
            stop_loss = latest_5m['BB_Mid']
            take_profit = latest_5m['Close'] + (latest_5m['Close'] - latest_5m['BB_Mid'])
            shares = int(position_size / entry_price)
        
        elif (trend_15m == "空頭" and rsi_15m_filter and
              latest_5m['Close'] < latest_5m['BB_Lower'] and 
              latest_5m['RSI'] > 30 and 
              latest_5m['Volume'] > latest_5m['Volume_MA5'] and 
              latest_5m['OBV'] < prev_5m['OBV']):
            signal = "做空"
            strategy = "布林帶突破策略"
            entry_price = latest_5m['Close']
            stop_loss = latest_5m['BB_Mid']
            take_profit = latest_5m['Close'] - (latest_5m['BB_Mid'] - latest_5m['Close'])
            shares = int(position_size / entry_price)
        
        elif (latest_5m['RSI'] > 70 and 
              latest_5m['Close'] > latest_5m['BB_Upper'] and 
              latest_5m['OBV'] < prev_5m['OBV'] and 
              latest_15m['RSI'].item() > 60):
            signal = "做空"
            strategy = "RSI 反轉策略"
            entry_price = latest_5m['Close']
            stop_loss = latest_5m['Close'] + 1.5
            take_profit = latest_5m['BB_Mid']
            shares = int(position_size / entry_price)
        
        elif (latest_5m['RSI'] < 30 and 
              latest_5m['Close'] < latest_5m['BB_Lower'] and 
              latest_5m['OBV'] > prev_5m['OBV'] and 
              latest_15m['RSI'].item() < 40):
            signal = "做多"
            strategy = "RSI 反轉策略"
            entry_price = latest_5m['Close']
            stop_loss = latest_5m['Close'] - 1.5
            take_profit = latest_5m['BB_Mid']
            shares = int(position_size / entry_price)
        
        if signal != "無交易信號":
            active_trade = True
            profit, exit_reason = simulate_trade(signal, df_5m, i, shares, entry_price, stop_loss, take_profit, commission)
            risk = abs((entry_price - stop_loss) * shares) if signal == "做多" else abs((stop_loss - entry_price) * shares)
            reward = profit + commission if profit > 0 else 0
            rr_ratio = reward / risk if risk > 0 and profit > 0 else 0
            trades.append({
                "時間": df_5m.index[i],
                "策略": strategy,
                "信號": signal,
                "入場價": entry_price,
                "止損": stop_loss,
                "止盈": take_profit,
                "股數": shares,
                "盈虧": profit,
                "風險": risk,
                "實現回報": reward,
                "風險回報比": rr_ratio,
                "退出原因": exit_reason
            })
            capital += profit
            equity.append(capital)
            active_trade = False
    
    return trades, equity

# 模擬交易
def simulate_trade(signal, df, i, shares, entry_price, stop_loss, take_profit, commission=5):
    for j in range(i + 1, len(df)):
        high = df['High'].iloc[j]
        low = df['Low'].iloc[j]
        if signal == "做多":
            if low <= stop_loss:
                return (stop_loss - entry_price) * shares - commission, "止損"
            if high >= take_profit:
                return (take_profit - entry_price) * shares - commission, "止盈"
        elif signal == "做空":
            if high >= stop_loss:
                return (entry_price - stop_loss) * shares - commission, "止損"
            if low <= take_profit:
                return (entry_price - take_profit) * shares - commission, "止盈"
    close_price = df['Close'].iloc[-1]
    if signal == "做多":
        return (close_price - entry_price) * shares - commission, "持有結束"
    else:
        return (entry_price - close_price) * shares - commission, "持有結束"

# 分析回測結果
def analyze_results(trades, equity):
    if not trades:
        return {
            "勝率": 0,
            "平均風險回報比": 0,
            "交易次數": 0,
            "淨利潤": 0,
            "最大回撤": 0
        }
    
    df_trades = pd.DataFrame(trades)
    win_trades = df_trades[df_trades['盈虧'] > 0]
    win_rate = len(win_trades) / len(df_trades) if len(df_trades) > 0 else 0
    avg_rr_ratio = df_trades['風險回報比'].mean()
    total_profit = df_trades['盈虧'].sum()
    equity_series = pd.Series(equity)
    drawdowns = (equity_series.cummax() - equity_series) / equity_series.cummax()
    max_drawdown = drawdowns.max()
    
    return {
        "勝率": win_rate,
        "平均風險回報比": avg_rr_ratio,
        "交易次數": len(df_trades),
        "淨利潤": total_profit,
        "最大回撤": max_drawdown
    }

# 繪製 K 線圖
def plot_data(df, interval="5分鐘"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="K線"
    ))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA5'], name="EMA5", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20", line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name="VWAP", line=dict(color='purple', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="布林上軌", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="布林下軌", line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="布林中軌", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量", yaxis="y2", opacity=0.3))
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV", yaxis="y3", line=dict(color='green')))
    fig.update_layout(
        title=f"TSLA {interval} K線與技術指標",
        xaxis_title="時間",
        yaxis_title="價格 (美元)",
        yaxis2=dict(title="成交量", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="OBV", anchor="free", overlaying="y", side="right", position=0.95, showgrid=False),
        height=800
    )
    return fig

# 主程式
def main():
    st.sidebar.header("設置")
    symbol = st.sidebar.text_input("股票代碼", "TSLA")
    refresh_interval = st.sidebar.slider("更新間隔（秒）", 60, 600, 300)
    email_enabled = st.sidebar.checkbox("啟用 Email 警報", False)
    email_sender = st.sidebar.text_input("發送者 Email（Gmail）", "")
    email_password = st.sidebar.text_input("應用程式密碼", "", type="password")
    email_receiver = st.sidebar.text_input("接收者 Email", "")
    email_config = {
        "enabled": email_enabled,
        "sender": email_sender,
        "password": email_password,
        "receiver": email_receiver
    }
    
    # 回測結果
    st.header("回測結果（過去 5 天）")
    df_5m_backtest = fetch_historical_data(symbol, interval="5m", period="5d")
    df_15m_backtest = fetch_historical_data(symbol, interval="15m", period="5d")
    
    if df_5m_backtest is not None and not df_5m_backtest.empty and df_15m_backtest is not None and not df_15m_backtest.empty:
        df_5m_backtest = calculate_indicators(df_5m_backtest)
        df_15m_backtest = calculate_indicators(df_15m_backtest)
        if df_5m_backtest is not None and df_15m_backtest is not None:
            trades, equity = backtest_strategy(df_5m_backtest, df_15m_backtest)
            results = analyze_results(trades, equity)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("勝率", f"{results['勝率']:.2%}")
            col2.metric("平均風險回報比", f"{results['平均風險回報比']:.2f}")
            col3.metric("交易次數", results['交易次數'])
            col1.metric("淨利潤", f"{results['淨利潤']:.2f} 美元")
            col2.metric("最大回撤", f"{results['最大回撤']:.2%}")
            
            if trades:
                st.subheader("回測交易詳情")
                df_trades = pd.DataFrame(trades)
                st.dataframe(df_trades[['時間', '策略', '信號', '入場價', '止損', '止盈', '盈虧', '風險回報比', '退出原因']])
        else:
            st.error("回測指標計算失敗，請檢查數據")
    else:
        st.error("無法獲取回測數據，請檢查網路或稍後重試")
    
    # 實時監控
    placeholder = st.empty()
    while True:
        with placeholder.container():
            st.header("實時監控")
            df_5m = fetch_historical_data(symbol, interval="5m", period="1d")
            df_15m = fetch_historical_data(symbol, interval="15m", period="1d")
            
            if df_5m is None or df_5m.empty or df_15m is None or df_15m.empty:
                st.error("實時數據獲取失敗，跳過本次更新")
                time.sleep(refresh_interval)
                st.experimental_rerun()
                continue
            
            df_5m = calculate_indicators(df_5m)
            df_15m = calculate_indicators(df_15m)
            if df_5m is None or df_15m is None:
                st.error("指標計算失敗，跳過本次更新")
                time.sleep(refresh_interval)
                st.experimental_rerun()
                continue
            
            st.subheader("5分鐘K線數據")
            latest_5m = df_5m.iloc[-1]
            st.write(f"當前價格 (TSLA): {latest_5m['Close']:.2f} 美元 | 時間: {df_5m.index[-1]}")
            col1, col2, col3 = st.columns(3)
            col1.metric("EMA5", f"{latest_5m['EMA5']:.2f}")
            col1.metric("EMA20", f"{latest_5m['EMA20']:.2f}")
            col2.metric("VWAP", f"{latest_5m['VWAP']:.2f}")
            col2.metric("RSI", f"{latest_5m['RSI']:.2f}")
            col3.metric("布林上軌", f"{latest_5m['BB_Upper']:.2f}")
            col3.metric("布林下軌", f"{latest_5m['BB_Lower']:.2f}")
            col3.metric("OBV", f"{latest_5m['OBV']:.0f}")
            st.plotly_chart(plot_data(df_5m, "5分鐘"), use_container_width=True)
            
            st.subheader("15分鐘K線數據（趨勢確認）")
            latest_15m = df_15m.iloc[-1]
            st.write(f"趨勢: {'多頭' if latest_15m['EMA5'].item() > latest_15m['EMA20'].item() else '空頭'} | 時間: {df_15m.index[-1]}")
            col1, col2, col3 = st.columns(3)
            col1.metric("EMA5", f"{latest_15m['EMA5']:.2f}")
            col1.metric("EMA20", f"{latest_15m['EMA20']:.2f}")
            col2.metric("RSI", f"{latest_15m['RSI']:.2f}")
            col3.metric("布林上軌", f"{latest_15m['BB_Upper']:.2f}")
            col3.metric("布林下軌", f"{latest_15m['BB_Lower']:.2f}")
            st.plotly_chart(plot_data(df_15m, "15分鐘"), use_container_width=True)
            
            signal = generate_trading_signal(df_5m, df_15m)
            st.subheader("實時交易策略建議")
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
                
                if (st.session_state.last_signal is None or 
                    st.session_state.last_signal['signal'] != signal['signal'] or 
                    st.session_state.last_signal['strategy'] != signal['strategy']):
                    send_email_alert(signal, email_config)
                    st.session_state.last_signal = signal
                
                st.session_state.signal_history.append({
                    "時間": signal['timestamp'],
                    "策略": signal['strategy'],
                    "信號": signal['signal'],
                    "入場價": signal['entry_price'],
                    "止損": signal['stop_loss'],
                    "止盈": signal['take_profit'],
                    "風險回報比": signal['rr_ratio']
                })
                if len(st.session_state.signal_history) > 5:
                    st.session_state.signal_history.pop(0)
            else:
                st.write("當前無明確交易信號，請等待機會。")
            
            st.subheader("最近交易信號（最多5筆）")
            if st.session_state.signal_history:
                history_df = pd.DataFrame(st.session_state.signal_history)
                st.dataframe(history_df[['時間', '策略', '信號', '入場價', '止損', '止盈', '風險回報比']])
            
            st.write(f"下次更新於 {datetime.now() + timedelta(seconds=refresh_interval)}")
        
        time.sleep(refresh_interval)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
