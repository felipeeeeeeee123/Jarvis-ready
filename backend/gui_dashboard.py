import tkinter as tk
from utils.memory import MemoryManager


def build_gui():
    mem = MemoryManager()
    root = tk.Tk()
    root.title("JARVIS Dashboard")
    text = tk.Text(root, width=60, height=20)
    text.pack()
    for ticker, info in mem.memory.items():
        if ticker in ("stats", "cooldowns"):
            continue
        text.insert(tk.END, f"{ticker}: {info['total_profit']:.2f} P/L\n")
    stats = mem.memory.get("stats", {})
    if stats:
        wins = stats.get('wins',0)
        losses = stats.get('losses',0)
        total = wins+losses
        if total:
            win_rate = wins/total*100
            text.insert(tk.END, f"Win rate: {win_rate:.2f}%\n")
    root.mainloop()


if __name__ == "__main__":
    build_gui()
