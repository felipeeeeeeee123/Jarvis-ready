import json
import os
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText


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

        self.button_frame = tk.Frame(self, bg="#121212")
        self.button_frame.pack(pady=5)

        self.pause_btn = ttk.Button(
            self.button_frame, text="Pause Trading", command=self.pause_trading
        )
        self.switch_btn = ttk.Button(
            self.button_frame, text="Switch Strategy", command=self.switch_strategy
        )
        self.dump_btn = ttk.Button(
            self.button_frame, text="Dump Memory", command=self.dump_memory
        )
        for btn in (self.pause_btn, self.switch_btn, self.dump_btn):
            btn.pack(side="left", padx=5)

        self.refresh_btn = ttk.Button(self, text="\N{CLOCKWISE OPEN CIRCLE ARROW} Refresh", command=self.load_data)
        self.refresh_btn.pack(pady=10)

        self.chat_frame = tk.LabelFrame(
            self,
            text="AI Chatbot",
            bg="#1e1e1e",
            fg="white",
            font=("Helvetica", 14),
            bd=2,
        )
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.chat_log = ScrolledText(
            self.chat_frame,
            font=("Consolas", 12),
            height=8,
            bg="#111",
            fg="lime",
            wrap="word",
            state="disabled",
        )
        self.chat_log.pack(fill="both", expand=True, padx=5, pady=5)

        entry_frame = tk.Frame(self.chat_frame, bg="#1e1e1e")
        entry_frame.pack(fill="x", padx=5, pady=5)
        self.chat_entry = ttk.Entry(entry_frame)
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_entry.bind("<Return>", self.send_chat)
        send_btn = ttk.Button(entry_frame, text="Send", command=self.send_chat)
        send_btn.pack(side="left")

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

    def pause_trading(self) -> None:
        self.paused = not getattr(self, "paused", False)
        state = "paused" if self.paused else "resumed"
        self.pause_btn.config(text="Resume Trading" if self.paused else "Pause Trading")
        messagebox.showinfo("Trading", f"Trading {state}.")

    def switch_strategy(self) -> None:
        messagebox.showinfo("Strategy", "Switching strategy (not implemented).")

    def dump_memory(self) -> None:
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "r") as f:
                data = json.load(f)
            path = "memory_dump.json"
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Memory Dump", f"Memory dumped to {path}.")

    def fake_ai_response(self, msg: str) -> str:
        return f"Simulated reply to: {msg}"

    def _append_chat(self, speaker: str, text: str) -> None:
        self.chat_log.configure(state="normal")
        self.chat_log.insert(tk.END, f"{speaker}: {text}\n")
        self.chat_log.see(tk.END)
        self.chat_log.configure(state="disabled")

    def send_chat(self, event=None) -> None:
        msg = self.chat_entry.get().strip()
        if not msg:
            return
        self.chat_entry.delete(0, tk.END)
        self._append_chat("You", msg)
        resp = self.fake_ai_response(msg)
        self._append_chat("AI", resp)


if __name__ == "__main__":
    app = JarvisGUI()
    app.mainloop()
