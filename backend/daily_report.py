import json
from pathlib import Path
from utils.memory import MemoryManager


def generate_report(path: str = "data/memory.json") -> str:
    if not Path(path).exists():
        return "No trade data."
    mem = MemoryManager(path)
    stats = mem.memory.get("stats", {"wins": 0, "losses": 0})
    report_lines = ["Daily Report:"]
    for ticker, info in mem.memory.items():
        if ticker in ("stats", "cooldowns"):
            continue
        report_lines.append(f"{ticker}: P/L {info['total_profit']:.2f} from {info['trade_count']} trades")
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    total = wins + losses
    if total:
        win_rate = wins / total * 100
        report_lines.append(f"Win rate: {win_rate:.2f}% ({wins}W/{losses}L)")
    return "\n".join(report_lines)


if __name__ == "__main__":
    print(generate_report())
