
import json
import os

class MemoryManager:
    def __init__(self, path='data/memory.json'):
        self.path = path
        self.memory = {}

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.memory = json.load(f)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def record_trade(self, ticker: str, buy_price: float, sell_price: float, quantity: float) -> float:
        """Update profit/loss for a completed trade and persist it."""
        pnl = (sell_price - buy_price) * quantity
        data = self.memory.setdefault(ticker, {"total_profit": 0.0, "trade_count": 0})
        data["total_profit"] += pnl
        data["trade_count"] += 1
        self.save()
        return pnl
