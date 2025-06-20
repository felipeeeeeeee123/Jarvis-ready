import json
import os
from typing import Dict, List, Any

DATA_PATH = "data/strategy_stats.json"
STRATEGIES = ["RSI", "EMA", "MACD"]
AUTO_MODE = True


def load_stats() -> Dict[str, Any]:
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def pnl_history(strategy: str) -> List[float]:
    stats = load_stats().get(strategy, {})
    return stats.get("history", [])


def toggle_auto() -> None:
    global AUTO_MODE
    AUTO_MODE = not AUTO_MODE


def switch_strategy(current: str) -> str:
    idx = STRATEGIES.index(current) if current in STRATEGIES else 0
    idx = (idx + 1) % len(STRATEGIES)
    return STRATEGIES[idx]
