# autotrade.py

from dotenv import load_dotenv
load_dotenv()

import os
from datetime import datetime
import pandas as pd
from alpaca_trade_api import REST, TimeFrame

from utils.memory import MemoryManager
from .telegram_alerts import send_telegram_alert
from .strategies import rsi_strategy, ema_strategy, macd_strategy

# === Load environment variables ===
ALPACA_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
TRADE_PERCENT = float(os.getenv("TRADE_PERCENT", 0.05))
TRADE_CAP = float(os.getenv("TRADE_CAP", 40))
STRATEGY = os.getenv("STRATEGY", "RSI").upper()
COOLDOWN = int(os.getenv("TRADE_COOLDOWN", 3600))

# === Alpaca REST client ===
aip = REST(ALPACA_KEY, ALPACA_SECRET, base_url=ALPACA_BASE_URL)
memory = MemoryManager()

# === Strategy definitions ===
STRATEGIES = {
    "RSI": rsi_strategy,
    "EMA": ema_strategy,
    "MACD": macd_strategy,
}

def choose_strategy():
    return STRATEGIES.get(STRATEGY, rsi_strategy)

def position_size(price: float, cash: float) -> int:
    budget = min(cash * TRADE_PERCENT, TRADE_CAP)
    return int(budget // price)

def trade_signal(symbol: str) -> str:
    end = datetime.utcnow()
    start = end - pd.Timedelta(days=10)
    bars = aip.get_bars(symbol, TimeFrame.Hour, start, end).df
    if bars.empty:
        return "hold"
    prices = bars.close
    strategy = choose_strategy()
    return strategy(prices)

def execute_trade(symbol: str) -> None:
    if not memory.should_trade(symbol, COOLDOWN):
        return
    account = aip.get_account()
    cash = float(account.cash)
    last_price = float(aip.get_latest_trade(symbol).price)
    qty = position_size(last_price, cash)
    if qty <= 0:
        return
    action = trade_signal(symbol)
    if action == "buy":
        aip.submit_order(symbol, qty, "buy", "market", "gtc")
        memory.set_cooldown(symbol)
        send_telegram_alert(f"Bought {qty} {symbol} @ {last_price}")
    elif action == "sell":
        aip.submit_order(symbol, qty, "sell", "market", "gtc")
        memory.set_cooldown(symbol)
        send_telegram_alert(f"Sold {qty} {symbol} @ {last_price}")

def run_autotrader(symbols=None):
    symbols = symbols or ["AAPL"]
    for sym in symbols:
        try:
            execute_trade(sym)
        except Exception as exc:
            print(f"Autotrade error for {sym}: {exc}")
