import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime

# 獲取歷史數據
def fetch_historical_data(symbol="TSLA", interval="5m", period="5d"):
    try:
        df = yf.download(symbol, interval=interval, period=period, prepost=False)
        if df.empty:
            raise ValueError("無數據返回，請檢查股票代碼或網路連線")
        return df
    except Exception as e:
        raise Exception(f"數據獲取失敗: {e}. 請檢查網路或 yfinance API 限制")

# 計算技術指標
def calculate_indicators(df):
    df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Price_Volume'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_PV'] = df['Price_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_PV'] / df['Cumulative_Volume']
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bollinger.bollinger_hband()
    df['BB_Lower'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
    return df

# 模擬交易並計算結果
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
    # 若未觸發，假設持有至回測結束
    close_price = df['Close'].iloc[-1]
    if signal == "做多":
        return (close_price - entry_price) * shares - commission, "持有結束"
    else:
        return (entry_price - close_price) * shares - commission, "持有結束"

# 回測策略
def backtest_strategy(df_5m, df_15m, capital=10000, risk_per_trade=0.015, commission=5):
    trades = []
    equity = [capital]
    position_size = capital * 0.2  # 20% 資金
    active_trade = False
    
    for i in range(21, len(df_5m) - 1):  # 從第21根開始確保指標計算完整
        latest_5m = df_5m.iloc[i]
        prev_5m = df_5m.iloc[i - 1]
        # 找到最近的 15 分鐘數據
        latest_15m_idx = df_15m.index[df_15m.index <= df_5m.index[i]].max()
        latest_15m = df_15m.loc[latest_15m_idx]
        
        if active_trade:
            continue
        
        # 15 分鐘趨勢確認
        trend_15m = "多頭" if latest_15m['EMA5'] > latest_15m['EMA20'] else "空頭"
        rsi_15m_filter = latest_15m['RSI'] < 70 if trend_15m == "多頭" else latest_15m['RSI'] > 30
        
        signal = "無交易信號"
        strategy = None
        entry_price = stop_loss = take_profit = shares = None
        
        # 策略1：VWAP 錨定趨勢策略
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
        
        # 策略2：布林帶突破策略
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
        
        # 策略3：RSI 反轉策略
        elif (latest_5m['RSI'] > 70 and 
              latest_5m['Close'] > latest_5m['BB_Upper'] and 
              latest_5m['OBV'] < prev_5m['OBV'] and 
              latest_15m['RSI'] > 60):
            signal = "做空"
            strategy = "RSI 反轉策略"
            entry_price = latest_5m['Close']
            stop_loss = latest_5m['Close'] + 1.5
            take_profit = latest_5m['BB_Mid']
            shares = int(position_size / entry_price)
        
        elif (latest_5m['RSI'] < 30 and 
              latest_5m['Close'] < latest_5m['BB_Lower'] and 
              latest_5m['OBV'] > prev_5m['OBV'] and 
              latest_15m['RSI'] < 40):
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

# 計算回測結果
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

# 主程式
def main():
    symbol = "TSLA"
    period = "5d"
    
    print(f"回測 {symbol} 過去 {period} 的 5 分鐘和 15 分鐘 K 線數據")
    
    # 獲取數據
    try:
        df_5m = fetch_historical_data(symbol, interval="5m", period=period)
        df_15m = fetch_historical_data(symbol, interval="15m", period=period)
    except Exception as e:
        print(e)
        return
    
    # 計算指標
    df_5m = calculate_indicators(df_5m)
    df_15m = calculate_indicators(df_15m)
    
    # 執行回測
    trades, equity = backtest_strategy(df_5m, df_15m)
    results = analyze_results(trades, equity)
    
    # 輸出結果
    print("\n回測結果:")
    print(f"勝率: {results['勝率']:.2%}")
    print(f"平均風險回報比: {results['平均風險回報比']:.2f}")
    print(f"交易次數: {results['交易次數']}")
    print(f"淨利潤: {results['淨利潤']:.2f} 美元")
    print(f"最大回撤: {results['最大回撤']:.2%}")
    
    # 顯示交易詳情
    if trades:
        print("\n交易詳情:")
        df_trades = pd.DataFrame(trades)
        print(df_trades[['時間', '策略', '信號', '入場價', '止損', '止盈', '盈虧', '風險回報比', '退出原因']].to_string(index=False))

if __name__ == "__main__":
    main()
