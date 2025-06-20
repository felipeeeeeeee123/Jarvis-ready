import json
import os
import time
import tkinter as tk
from tkinter import ttk


DATA_PATH = "data/strategy_stats.json"
MEMORY_PATH = "data/memory.json"


class JarvisGUI(tk.Tk):
    """Simple dashboard showing recent memories and strategy stats."""

    def __init__(self) -> None:
        super().__init__()
        self.title("JARVIS Command Center")
        self.geometry("800x600")
        self.configure(bg="#121212")

        title = tk.Label(
            self,
            text="JARVIS CORE DASHBOARD",
            font=("Helvetica", 20),
            fg="white",
            bg="#121212",
        )
        title.pack(pady=10)

        self.memory_frame = self._create_section("Top Memories")
        self.strategy_frame = self._create_section("Strategy Stats")

        self.refresh_btn = ttk.Button(self, text="\N{CLOCKWISE OPEN CIRCLE ARROW} Refresh", command=self.load_data)
        self.refresh_btn.pack(pady=10)

        self.load_data()

    def _create_section(self, title: str) -> tk.LabelFrame:
        frame = tk.LabelFrame(
            self,
            text=title,
            bg="#1e1e1e",
            fg="white",
            font=("Helvetica", 14),
            bd=2,
        )
        frame.pack(fill="both", expand=True, padx=10, pady=10)
        frame.listbox = tk.Listbox(frame, font=("Consolas", 12), height=10, bg="#222", fg="lime")
        frame.listbox.pack(fill="both", expand=True, padx=10, pady=10)
        return frame

    def _format_timestamp(self, ts: float | str) -> str:
        try:
            ts_f = float(ts)
            return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_f))
        except Exception:
            return str(ts)

    def load_data(self) -> None:
        """Refresh listboxes with data from disk."""
        self.memory_frame.listbox.delete(0, tk.END)
        self.strategy_frame.listbox.delete(0, tk.END)

        # Load memories
        if os.path.exists(MEMORY_PATH):
            try:
                with open(MEMORY_PATH, "r") as f:
                    memories = json.load(f)
            except Exception:
                memories = []
            top_memories: list[dict] = []
            if isinstance(memories, list):
                top_memories = sorted(memories, key=lambda m: m.get("importance", 1), reverse=True)[:5]
            elif isinstance(memories, dict):
                for ticker, info in memories.items():
                    if ticker in {"stats", "cooldowns"}:
                        continue
                    top_memories.append({"timestamp": ticker, "event": f"{info.get('total_profit', 0.0):.2f} P/L"})
                top_memories = top_memories[:5]
            for mem in top_memories:
                ts = self._format_timestamp(mem.get("timestamp"))
                event = mem.get("event", "")
                self.memory_frame.listbox.insert(tk.END, f"{ts} — {event}")

        # Load strategy stats
        if os.path.exists(DATA_PATH):
            try:
                with open(DATA_PATH, "r") as f:
                    stats = json.load(f)
            except Exception:
                stats = {}
            if isinstance(stats, dict):
                for strat, data in stats.items():
                    wins = data.get("wins", 0)
                    losses = data.get("losses", 0)
                    pnl = data.get("pnl", 0.0)
                    self.strategy_frame.listbox.insert(
                        tk.END,
                        f"{strat}: W {wins}, L {losses}, PnL ${pnl:.2f}",
                    )


if __name__ == "__main__":
    app = JarvisGUI()
    app.mainloop()
