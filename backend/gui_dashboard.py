"""Tkinter-based dashboard for interacting with the trading backend."""

import sys
import os
import subprocess
import json
import time
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

# === Inject project root for local imports ===
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# === Required third party packages ===
REQUIRED_PACKAGES = [
    "requests",
    "pandas",
    "matplotlib",
    "plotly",
    "PyMuPDF",
    "beautifulsoup4",
    "ollama",
    "python-dotenv",
]


def ensure_packages_installed() -> None:
    """Install any missing dependencies on the fly."""
    for pkg in REQUIRED_PACKAGES:
        module_name = pkg.replace("-", "_")
        try:
            __import__(module_name)
        except ImportError:
            print(f"\U0001F4E6 Installing missing package: {pkg}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


ensure_packages_installed()

try:
    import requests  # noqa: F401 - imported for side effects
except ImportError as exc:  # pragma: no cover - runtime guard
    raise ImportError(
        "❌ Missing 'requests'. Activate your venv and run: pip install requests"
    ) from exc

from gui.handlers.ai_handler import (  # noqa: E402
    ask_ai,
    record_command,
    interpret_command,
    set_current_strategy,
    last_context,
    apply_feedback,
)
from gui.handlers.memory_handler import export_memory, search_memory, top_memories  # noqa: E402
from gui.handlers.strategy_handler import (  # noqa: E402
    STRATEGIES,
    load_stats,
    pnl_history,
    switch_strategy,
    toggle_auto,
    AUTO_MODE,
)

CONFIG_PATH = "config.json"


class JarvisGUI(tk.Tk):
    """AI Operating System dashboard."""

    def __init__(self) -> None:
        super().__init__()
        self.title("JARVIS Command Center")
        self.geometry("900x700")
        self.configure(bg="#121212")

        self.config_data = self._load_config()
        self.current_strategy = self.config_data.get("default_strategy", STRATEGIES[0])
        set_current_strategy(self.current_strategy)

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
        ttk.Button(entry_frame, text="👍", command=lambda: self.feedback(True)).pack(side="left", padx=(5, 0))
        ttk.Button(entry_frame, text="👎", command=lambda: self.feedback(False)).pack(side="left")

        self.toolbar = tk.Frame(self, bg="#121212")
        self.toolbar.pack(fill="x", pady=5)

        self.pause_btn = ttk.Button(self.toolbar, text="Pause Trading", command=self.pause_trading)
        self.pause_btn.pack(side="left", padx=5)
        self.switch_btn = ttk.Button(self.toolbar, text="Switch Strategy", command=self.switch_strategy_cmd)
        self.switch_btn.pack(side="left", padx=5)
        self.dump_btn = ttk.Button(self.toolbar, text="Dump Memory", command=self.dump_memory)
        self.dump_btn.pack(side="left", padx=5)
        self.refresh_btn = ttk.Button(self.toolbar, text="\N{CLOCKWISE OPEN CIRCLE ARROW} Refresh", command=self.load_data)
        self.refresh_btn.pack(side="left", padx=5)

        # Memory search and export
        search_bar = tk.Frame(self.memory_frame, bg="#1e1e1e")
        search_bar.pack(fill="x", padx=5, pady=(0, 5))
        self.search_entry = ttk.Entry(search_bar)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(search_bar, text="Search", command=self.search_memories).pack(side="left")
        ttk.Button(search_bar, text="Export", command=self.export_mem).pack(side="left")

        # Strategy controls
        controls = tk.Frame(self.strategy_frame, bg="#1e1e1e")
        controls.pack(fill="x", padx=5, pady=(0, 5))
        self.strategy_var = tk.StringVar(value=self.current_strategy)
        self.strategy_dd = ttk.Combobox(controls, values=STRATEGIES, textvariable=self.strategy_var, state="readonly")
        self.strategy_dd.pack(side="left", padx=(0,5))
        self.strategy_dd.bind("<<ComboboxSelected>>", lambda e: self.load_data())
        self.auto_var = tk.BooleanVar(value=self.config_data.get("auto_mode", True))
        auto_btn = ttk.Checkbutton(controls, text="Auto Mode", variable=self.auto_var, command=self.toggle_auto_mode)
        auto_btn.pack(side="left")
        ttk.Button(controls, text="Show Graph", command=self.show_graph).pack(side="left", padx=5)

        self.console = ScrolledText(self, height=5, bg="#000", fg="white", state="disabled")
        self.console.pack(fill="both", padx=10, pady=5)

        self.load_data()

        self.bind_all("<Control-r>", lambda e: self.load_data())
        self.bind_all("<Control-m>", lambda e: self.dump_memory())

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

        for i, mem in enumerate(top_memories(5)):
            ts = self._format_timestamp(mem.get("timestamp"))
            event = mem.get("event", "")
            tag = " \N{FIRE}" if i < 3 else ""
            self.memory_frame.listbox.insert(tk.END, f"{ts} — {event}{tag}")

        stats = load_stats().get(self.strategy_var.get(), {})
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        pnl = stats.get("pnl", 0.0)
        self.strategy_frame.listbox.insert(
            tk.END,
            f"{self.strategy_var.get()}: W {wins}, L {losses}, PnL ${pnl:.2f}",
        )

    def pause_trading(self) -> None:
        self.paused = not getattr(self, "paused", False)
        state = "paused" if self.paused else "resumed"
        self.pause_btn.config(text="Resume Trading" if self.paused else "Pause Trading")
        self._log(f"Trading {state}.")

    def switch_strategy_cmd(self) -> None:
        old = self.strategy_var.get()
        new = switch_strategy(old)
        self.strategy_var.set(new)
        self.current_strategy = new
        set_current_strategy(new)
        self._log(f"Strategy switched to {new}.")
        self.load_data()

    def dump_memory(self) -> None:
        path = export_memory()
        self._log(f"Memory exported to {path}")

    def _log(self, text: str) -> None:
        self.console.configure(state="normal")
        self.console.insert(tk.END, text + "\n")
        self.console.see(tk.END)
        self.console.configure(state="disabled")

    def _append_chat(self, speaker: str, text: str) -> None:
        self.chat_log.configure(state="normal")
        self.chat_log.insert(tk.END, f"{speaker}: {text}\n")
        self.chat_log.see(tk.END)
        self.chat_log.configure(state="disabled")

    def _typewriter(self, text: str, idx: int = 0) -> None:
        if idx < len(text):
            self.chat_log.configure(state="normal")
            self.chat_log.insert(tk.END, text[idx])
            self.chat_log.see(tk.END)
            self.chat_log.configure(state="disabled")
            self.after(25, self._typewriter, text, idx + 1)
        else:
            self.chat_log.configure(state="normal")
            self.chat_log.insert(tk.END, "\n")
            self.chat_log.configure(state="disabled")

    def send_chat(self, event=None) -> None:
        msg = self.chat_entry.get().strip()
        if not msg:
            return
        self.chat_entry.delete(0, tk.END)
        record_command(msg)
        intent = interpret_command(msg)
        if intent["action"] == "switch_strategy":
            new = intent.get("strategy") or switch_strategy(self.current_strategy)
            self.strategy_var.set(new)
            self.current_strategy = new
            set_current_strategy(new)
            self._log(f"Strategy switched to {new}.")
            self.load_data()
            return
        if intent["action"] == "pause_trading":
            self.pause_trading()
            return
        if intent["action"] == "show_history":
            self.show_graph()
            return

        self._append_chat("You", msg)
        self.chat_log.configure(state="normal")
        self.chat_log.insert(tk.END, "AI: ")
        self.chat_log.configure(state="disabled")
        resp = ask_ai(msg)
        self.last_memories = last_context().get("top_memories", []) if last_context() else []
        self._typewriter(resp)

    def feedback(self, positive: bool) -> None:
        if not hasattr(self, "last_memories"):
            return
        apply_feedback(self.last_memories, positive)
        adj = "up" if positive else "down"
        self._log(f"Feedback recorded: thumbs-{adj}.")

    def search_memories(self) -> None:
        query = self.search_entry.get().strip()
        self.memory_frame.listbox.delete(0, tk.END)
        for mem in search_memory(query):
            ts = self._format_timestamp(mem.get("timestamp"))
            event = mem.get("event", "")
            self.memory_frame.listbox.insert(tk.END, f"{ts} — {event}")

    def export_mem(self) -> None:
        path = export_memory()
        self._log(f"Memory exported to {path}")

    def toggle_auto_mode(self) -> None:
        toggle_auto()
        state = "on" if AUTO_MODE else "off"
        self._log(f"Auto mode {state}")

    def show_graph(self) -> None:
        import matplotlib.pyplot as plt

        data = pnl_history(self.strategy_var.get())
        if not data:
            messagebox.showinfo("Graph", "No history available")
            return
        plt.figure(figsize=(4, 3))
        plt.plot(data)
        plt.title(f"{self.strategy_var.get()} PnL")
        plt.xlabel("Trade")
        plt.ylabel("PnL")
        plt.tight_layout()
        plt.show()

    def _load_config(self) -> dict:
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}


if __name__ == "__main__":
    app = JarvisGUI()
    app.mainloop()
