from utils.memory import MemoryManager

memory = MemoryManager()
memory.load()


def record_sell(
    ticker: str, buy_price: float, sell_price: float, quantity: float
) -> None:
    """Record the result of a completed trade and log the outcome."""
    pnl = memory.record_trade(ticker, buy_price, sell_price, quantity)
    total = memory.memory[ticker]["total_profit"]
    msg = f"\U0001f4ca {ticker} trade result: {pnl:+.2f} | Total P/L: {total:.2f}"
    print(msg)
    # Placeholder for Telegram integration


def run_autotrader():
    print("Running auto-trading logic...")
    # Insert Alpaca integration logic here
