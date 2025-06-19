
import json
import time
import os

class MemoryManager:
    def __init__(self, path='data/memory.json'):
        self.path = path
        self.memory = {}
        self.load()

    def load(self):
        if os.path.exists(self.path):
            with open(self.path, 'r') as f:
                self.memory = json.load(f)

    def save(self):
        with open(self.path, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def should_trade(self, ticker: str, cooldown: int) -> bool:
        """Return True if the ticker is not in cooldown period."""
        cooldowns = self.memory.setdefault("cooldowns", {})
        last = cooldowns.get(ticker, 0)
        return time.time() - last > cooldown

    def set_cooldown(self, ticker: str) -> None:
        self.memory.setdefault("cooldowns", {})[ticker] = time.time()
        self.save()

    def record_trade(self, ticker: str, buy_price: float, sell_price: float, quantity: float) -> float:
        """Update profit/loss for a completed trade and persist it."""
        pnl = (sell_price - buy_price) * quantity
        data = self.memory.setdefault(ticker, {"total_profit": 0.0, "trade_count": 0})
        data["total_profit"] += pnl
        data["trade_count"] += 1
        if pnl > 0:
            stats = self.memory.setdefault("stats", {"wins": 0, "losses": 0})
            stats["wins"] += 1
        else:
            stats = self.memory.setdefault("stats", {"wins": 0, "losses": 0})
            stats["losses"] += 1
        self.save()
        return pnl
