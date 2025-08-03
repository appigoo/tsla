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
# 新增一個用於控制實時監控循環的 session state
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# 獲取歷史數據
def fetch_historical_data(symbol="TSLA", interval="5m", period="5d"):
    try:
        df = yf.download(symbol, interval=interval, period=period, prepost=False)
        if df.empty:
            st.warning("無數據返回，請檢查股票代碼或網路連線。")
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"數據獲取失敗: {e}. 請檢查網路或 yfinance API 限制。")
        return None

# 計算技術指標
def calculate_indicators(df):
    if df is None or len(df) < 21:  # 至少需要21根K線來計算指標
        return None
    try:
        close = df['Close']
        volume = df['Volume']
        
        df['EMA5'] = ta.trend.ema_indicator(close, window=5)
        df['EMA20'] = ta.trend.ema_indicator(close, window=20)
        df['RSI'] = ta.momentum.rsi(close, window=14)
        
        # 修正VWAP計算邏輯，確保每個時間點的VWAP都是基於該時間點之前的數據
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        price_volume = typical_price * df['Volume']
        df['VWAP'] = price_volume.cumsum() / df['Volume'].cumsum()
        
        bollinger = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        
        df['OBV'] = ta.volume.on_balance_volume(close, volume)
        df['Volume_MA5'] = volume.rolling(window=5).mean()
        
        # 移除前20行有 NaN 值的數據
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"指標計算失敗: {e}")
        return None

# 發送 email 警報
def send_email_alert(signal, email_config):
    if not email_config['enabled'] or not all(email_config.values()):
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
        st.error(f"Email 發送失敗: {e}. 請檢查 Gmail 應用程式密碼或收件人設定。")

# 生成實時交易信號
def generate_trading_signal(df_5m, df_15m, capital=10000):
    if df_5m.empty or df_15m.empty or len(df_5m) < 2 or len(df_15m) < 1:
        return {"signal": "無交易信號", "strategy": None, "timestamp": datetime.now()}

    latest_5m = df_5m.iloc[-1]
    prev_5m = df_5m.iloc[-2]
    
    # 找到與最新5分鐘K線時間點最接近的15分鐘K線
    latest_15m_idx = df_15m.index[df_15m.index <= df_5m.index[-1]].max()
    latest_15m = df_15m.loc[latest_15m_idx]

    signal = "無交易信號"
    strategy = None
    entry_price = stop_loss = take_profit = risk = reward = rr_ratio = shares = None
    
    # 風險管理
    risk_per_trade = 0.015
    risk_amount = capital * risk_per_trade
    
    trend_15m = "多頭" if latest_15m['EMA5'] > latest_15m['EMA20'] else "空頭"
    rsi_15m_filter = latest_15m['RSI'] < 70 if trend_15m == "多頭" else latest_15m['RSI'] > 30

    # 策略 1: VWAP 錨定趨勢策略
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
        stop_loss_price = max(latest_5m['VWAP'], entry_price - 2)
        take_profit = entry_price + 3.5
        shares = int(risk_amount / (entry_price - stop_loss_price)) if (entry_price - stop_loss_price) > 0 else 0
        risk = shares * (entry_price - stop_loss_price)
        reward = shares * (take_profit - entry_price)

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
        stop_loss_price = min(latest_5m['VWAP'], entry_price + 2)
        take_profit = entry_price - 3.5
        shares = int(risk_amount / (stop_loss_price - entry_price)) if (stop_loss_price - entry_price) > 0 else 0
        risk = shares * (stop_loss_price - entry_price)
        reward = shares * (entry_price - take_profit)
    
    # 策略 2: 布林帶突破策略
    elif (trend_15m == "多頭" and rsi_15m_filter and
          latest_5m['Close'] > latest_5m['BB_Upper'] and
          latest_5m['RSI'] < 70 and
          latest_5m['Volume'] > latest_5m['Volume_MA5'] and
          latest_5m['OBV'] > prev_5m['OBV']):
        signal = "做多"
        strategy = "布林帶突破策略"
        entry_price = latest_5m['Close']
        stop_loss_price = latest_5m['BB_Mid']
        take_profit = entry_price + (entry_price - latest_5m['BB_Mid'])
        shares = int(risk_amount / (entry_price - stop_loss_price)) if (entry_price - stop_loss_price) > 0 else 0
        risk = shares * (entry_price - stop_loss_price)
        reward = shares * (take_profit - entry_price)

    elif (trend_15m == "空頭" and rsi_15m_filter and
          latest_5m['Close'] < latest_5m['BB_Lower'] and
          latest_5m['RSI'] > 30 and
          latest_5m['Volume'] > latest_5m['Volume_MA5'] and
          latest_5m['OBV'] < prev_5m['OBV']):
        signal = "做空"
        strategy = "布林帶突破策略"
        entry_price = latest_5m['Close']
        stop_loss_price = latest_5m['BB_Mid']
        take_profit = entry_price - (latest_5m['BB_Mid'] - entry_price)
        shares = int(risk_amount / (stop_loss_price - entry_price)) if (stop_loss_price - entry_price) > 0 else 0
        risk = shares * (stop_loss_price - entry_price)
        reward = shares * (entry_price - take_profit)

    # 策略 3: RSI 反轉策略
    elif (latest_5m['RSI'] > 70 and
          latest_5m['Close'] > latest_5m['BB_Upper'] and
          latest_5m['OBV'] < prev_5m['OBV'] and
          latest_15m['RSI'] > 60):
        signal = "做空"
        strategy = "RSI 反轉策略"
        entry_price = latest_5m['Close']
        stop_loss_price = entry_price + 1.5
        take_profit = latest_5m['BB_Mid']
        shares = int(risk_amount / (stop_loss_price - entry_price)) if (stop_loss_price - entry_price) > 0 else 0
        risk = shares * (stop_loss_price - entry_price)
        reward = shares * (entry_price - take_profit)
        
    elif (latest_5m['RSI'] < 30 and
          latest_5m['Close'] < latest_5m['BB_Lower'] and
          latest_5m['OBV'] > prev_5m['OBV'] and
          latest_15m['RSI'] < 40):
        signal = "做多"
        strategy = "RSI 反轉策略"
        entry_price = latest_5m['Close']
        stop_loss_price = entry_price - 1.5
        take_profit = latest_5m['BB_Mid']
        shares = int(risk_amount / (entry_price - stop_loss_price)) if (entry_price - stop_loss_price) > 0 else 0
        risk = shares * (entry_price - stop_loss_price)
        reward = shares * (take_profit - entry_price)

    # 確保不會產生無限風險或無效交易
    if shares is not None and shares <= 0:
        signal = "無交易信號"
    
    rr_ratio = reward / abs(risk) if risk and abs(risk) != 0 else float('inf') if reward else 0

    return {
        "signal": signal,
        "strategy": strategy,
        "entry_price": entry_price,
        "stop_loss": stop_loss_price,
        "take_profit": take_profit,
        "risk": risk,
        "reward": reward,
        "rr_ratio": rr_ratio,
        "shares": shares,
        "timestamp": datetime.now()
    }

# 回測策略
def backtest_strategy(df_5m, df_15m, capital=10000, commission=5):
    if df_5m is None or df_15m is None or len(df_5m) < 21 or len(df_15m) < 21:
        st.error("數據不足以計算指標（需要至少21根K線）")
        return [], [capital]

    trades = []
    equity = [capital]
    active_trade = None  # 使用字典儲存活動交易的資訊
    
    for i in range(21, len(df_5m)):
        latest_5m = df_5m.iloc[i]

        if active_trade:
            # 檢查是否觸發止損或止盈
            if active_trade['signal'] == "做多":
                if latest_5m['Low'] <= active_trade['stop_loss']:
                    profit = (active_trade['stop_loss'] - active_trade['entry_price']) * active_trade['shares'] - commission
                    exit_reason = "止損"
                elif latest_5m['High'] >= active_trade['take_profit']:
                    profit = (active_trade['take_profit'] - active_trade['entry_price']) * active_trade['shares'] - commission
                    exit_reason = "止盈"
                else:
                    continue  # 繼續持有
            elif active_trade['signal'] == "做空":
                if latest_5m['High'] >= active_trade['stop_loss']:
                    profit = (active_trade['entry_price'] - active_trade['stop_loss']) * active_trade['shares'] - commission
                    exit_reason = "止損"
                elif latest_5m['Low'] <= active_trade['take_profit']:
                    profit = (active_trade['entry_price'] - active_trade['take_profit']) * active_trade['shares'] - commission
                    exit_reason = "止盈"
                else:
                    continue # 繼續持有

            capital += profit
            equity.append(capital)
            
            # 記錄交易結果
            active_trade['盈虧'] = profit
            active_trade['退出原因'] = exit_reason
            trades.append(active_trade)
            active_trade = None # 結束交易
            
        else: # 沒有活動交易時，生成新信號
            prev_5m = df_5m.iloc[i - 1]
            latest_15m_idx = df_15m.index[df_15m.index <= df_5m.index[i]].max()
            if pd.isna(latest_15m_idx): continue
            latest_15m = df_15m.loc[latest_15m_idx]

            signal_data = generate_trading_signal(df_5m.iloc[:i+1], df_15m.loc[:latest_15m_idx], capital=equity[-1])
            
            if signal_data['signal'] != "無交易信號":
                # 開始一筆新交易
                active_trade = {
                    "時間": latest_5m.name,
                    "策略": signal_data['strategy'],
                    "信號": signal_data['signal'],
                    "入場價": signal_data['entry_price'],
                    "止損": signal_data['stop_loss'],
                    "止盈": signal_data['take_profit'],
                    "股數": signal_data['shares'],
                    "風險": signal_data['risk'],
                    "實現回報": signal_data['reward'],
                    "風險回報比": signal_data['rr_ratio']
                }

    # 如果回測結束時仍有持倉，則以最後一根K線的收盤價平倉
    if active_trade:
        final_price = df_5m.iloc[-1]['Close']
        if active_trade['signal'] == "做多":
            profit = (final_price - active_trade['entry_price']) * active_trade['shares'] - commission
        else: # 做空
            profit = (active_trade['entry_price'] - final_price) * active_trade['shares'] - commission
        
        capital += profit
        equity.append(capital)
        active_trade['盈虧'] = profit
        active_trade['退出原因'] = "回測結束"
        trades.append(active_trade)

    return trades, equity

# 分析回測結果
def analyze_results(trades, equity):
    if not trades:
        return {
            "勝率": 0,
            "平均風險回報比": 0,
            "交易次數": 0,
            "淨利潤": 0,
            "最大回撤": 0,
            "回測資產": equity[-1] if equity else 0
        }
    
    df_trades = pd.DataFrame(trades)
    win_trades = df_trades[df_trades['盈虧'] > 0]
    win_rate = len(win_trades) / len(df_trades)
    
    # 避免除以零
    risks = df_trades['風險']
    rewards = df_trades['實現回報']
    rr_ratios = rewards / risks
    rr_ratios.replace([np.inf, -np.inf], np.nan, inplace=True)
    avg_rr_ratio = rr_ratios.mean() if not rr_ratios.isnull().all() else 0
    
    total_profit = df_trades['盈虧'].sum()
    equity_series = pd.Series(equity)
    cumulative_max = equity_series.cummax()
    drawdowns = (cumulative_max - equity_series) / cumulative_max
    max_drawdown = drawdowns.max()
    
    return {
        "勝率": win_rate,
        "平均風險回報比": avg_rr_ratio,
        "交易次數": len(df_trades),
        "淨利潤": total_profit,
        "最大回撤": max_drawdown,
        "回測資產": equity[-1]
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
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="布林上軌", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="布林下軌", line=dict(color='gray', dash='dot')))
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], name="布林中軌", line=dict(color='gray')))
    
    # 使用 make_subplots 來處理多個 y 軸
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        row_width=[0.2, 0.2, 0.6],  # 調整子圖高度比例
                        row_titles=['價格/指標', '成交量', 'OBV'])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="K線"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA5'], name="EMA5", line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], name="EMA20", line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['VWAP'], name="VWAP", line=dict(color='purple', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name="布林上軌", line=dict(color='gray', dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name="布林下軌", line=dict(color='gray', dash='dot')), row=1, col=1)
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="成交量", opacity=0.3), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['OBV'], name="OBV", line=dict(color='green')), row=3, col=1)

    fig.update_layout(
        title=f"TSLA {interval} K線與技術指標",
        xaxis_title="時間",
        yaxis_title="價格 (美元)",
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# 主程式
def main():
    st.sidebar.header("設定")
    symbol = st.sidebar.text_input("股票代碼", "TSLA")
    refresh_interval = st.sidebar.slider("更新間隔（秒）", 60, 600, 300)
    email_enabled = st.sidebar.checkbox("啟用 Email 警報", False)
    email_sender = st.sidebar.text_input("發送者 Email（Gmail）", "your_email@gmail.com")
    email_password = st.sidebar.text_input("應用程式密碼", "your_app_password", type="password")
    email_receiver = st.sidebar.text_input("接收者 Email", "receiver_email@example.com")
    email_config = {
        "enabled": email_enabled,
        "sender": email_sender,
        "password": email_password,
        "receiver": email_receiver
    }

    st.header("回測結果（過去 5 天）")
    df_5m_backtest = fetch_historical_data(symbol, interval="5m", period="5d")
    df_15m_backtest = fetch_historical_data(symbol, interval="15m", period="5d")
    
    if df_5m_backtest is not None and not df_5m_backtest.empty and df_15m_backtest is not None and not df_15m_backtest.empty:
        df_5m_backtest = calculate_indicators(df_5m_backtest.copy()) # 使用 .copy() 避免 SettingWithCopyWarning
        df_15m_backtest = calculate_indicators(df_15m_backtest.copy())
        if df_5m_backtest is not None and df_15m_backtest is not None:
            trades, equity = backtest_strategy(df_5m_backtest, df_15m_backtest)
            results = analyze_results(trades, equity)
            
            st.subheader("回測績效總覽")
            col1, col2, col3 = st.columns(3)
            col1.metric("初始資產", "10,000.00 美元")
            col2.metric("最終資產", f"{results['回測資產']:.2f} 美元")
            col3.metric("淨利潤", f"{results['淨利潤']:.2f} 美元")
            col1.metric("勝率", f"{results['勝率']:.2%}")
            col2.metric("平均風險回報比", f"{results['平均風險回報比']:.2f}")
            col3.metric("交易次數", results['交易次數'])
            col1.metric("最大回撤", f"{results['最大回撤']:.2%}")

            if trades:
                st.subheader("回測交易詳情")
                df_trades = pd.DataFrame(trades)
                st.dataframe(df_trades[['時間', '策略', '信號', '入場價', '止損', '止盈', '盈虧', '風險回報比', '退出原因']])
        else:
            st.error("回測指標計算失敗，請檢查數據。")
    else:
        st.error("無法獲取回測數據，請檢查網路或稍後重試。")

    st.header("實時監控")
    
    # 使用按鈕控制實時監控循環
    if st.sidebar.button("開始實時監控") and not st.session_state.is_running:
        st.session_state.is_running = True
        st.experimental_rerun()
    if st.sidebar.button("停止實時監控") and st.session_state.is_running:
        st.session_state.is_running = False
        st.experimental_rerun()

    if st.session_state.is_running:
        placeholder = st.empty()
        while True:
            with placeholder.container():
                df_5m = fetch_historical_data(symbol, interval="5m", period="1d")
                df_15m = fetch_historical_data(symbol, interval="15m", period="1d")
                
                if df_5m is None or df_5m.empty or df_15m is None or df_15m.empty:
                    st.error("實時數據獲取失敗，跳過本次更新。")
                    time.sleep(refresh_interval)
                    continue

                df_5m = calculate_indicators(df_5m.copy())
                df_15m = calculate_indicators(df_15m.copy())
                
                if df_5m is None or df_15m is None:
                    st.error("指標計算失敗，跳過本次更新。")
                    time.sleep(refresh_interval)
                    continue
                
                st.subheader("5分鐘K線數據")
                latest_5m = df_5m.iloc[-1]
                st.write(f"當前價格 ({symbol}): {latest_5m['Close']:.2f} 美元 | 時間: {df_5m.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
                
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
                trend_15m = '多頭' if latest_15m['EMA5'] > latest_15m['EMA20'] else '空頭'
                st.write(f"趨勢: {trend_15m} | 時間: {df_15m.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("EMA5", f"{latest_15m['EMA5']:.2f}")
                col1.metric("EMA20", f"{latest_15m['EMA20']:.2f}")
                col2.metric("RSI", f"{latest_15m['RSI']:.2f}")
                col3.metric("布林上軌", f"{latest_15m['BB_Upper']:.2f}")
                col3.metric("布林下軌", f"{latest_15m['BB_Lower']:.2f}")
                
                st.plotly_chart(plot_data(df_15m, "15分鐘"), use_container_width=True)
                
                signal = generate_trading_signal(df_5m, df_15m)
                st.subheader("實時交易策略建議")
                
                if signal['signal'] != "無交易信號" and signal['shares'] > 0:
                    st.success("發現交易信號！")
                    st.write(f"**策略**: {signal['strategy']}")
                    st.write(f"**信號**: {signal['signal']}")
                    st.write(f"**入場價**: {signal['entry_price']:.2f} 美元")
                    st.write(f"**止損價**: {signal['stop_loss']:.2f} 美元")
                    st.write(f"**止盈價**: {signal['take_profit']:.2f} 美元")
                    st.write(f"**持倉股數**: {signal['shares']}")
                    st.write(f"**風險**: {abs(signal['risk']):.2f} 美元")
                    st.write(f"**潛在盈利**: {signal['reward']:.2f} 美元")
                    st.write(f"**風險回報比**: {signal['rr_ratio']:.2f}")
                    
                    # 避免重複發送 Email
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
                    st.info("當前無明確交易信號，請等待機會。")
                    
                st.subheader("最近交易信號（最多5筆）")
                if st.session_state.signal_history:
                    history_df = pd.DataFrame(st.session_state.signal_history)
                    st.dataframe(history_df[['時間', '策略', '信號', '入場價', '止損', '止盈', '風險回報比']])
                else:
                    st.write("尚無交易信號記錄。")
                
                st.write(f"下次更新於 {datetime.now() + timedelta(seconds=refresh_interval)}")
            
            time.sleep(refresh_interval)
            st.experimental_rerun()
    else:
        st.info("請點擊左側「開始實時監控」按鈕以啟動監控。")

if __name__ == "__main__":
    main()
