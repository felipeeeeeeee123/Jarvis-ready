# Auto-generated combined file
import csv
import hashlib
import json
import random
import time
from pathlib import Path
from backend.features.ai_brain import AIBrain
from backend.features.evaluator import Evaluator
from backend.features.qa_memory import QAMemory
from backend.features.trending import TrendingTopics
from backend.features.web_search import web_search
import os
from flask import Flask
from threading import Thread
import sys
from utils.memory import MemoryManager
import requests
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
from alpaca_trade_api import REST, TimeFrame
from .telegram_alerts import send_telegram_alert
from .strategies import rsi_strategy, ema_strategy, macd_strategy
import curses
import threading
from .qa_memory import QAMemory
from .web_search import web_search
from typing import Callable, List, Tuple
import re
from uuid import uuid4
from difflib import SequenceMatcher
import fitz
import sympy as sp
import matplotlib
from .ai_brain import AIBrain
from .evaluator import Evaluator
from typing import List
import feedparser
from bs4 import BeautifulSoup
from features.ai_brain import AIBrain
from features.web_search import web_search
from features.autotrade import run_autotrader
from features.self_reflect import SelfReflection
from features.self_audit import SelfAudit
from features.dashboard import TerminalDashboard
import subprocess
from flask import Flask, request, jsonify
import streamlit as st
import base64
from typing import Any, List, Dict
from typing import Dict, List, Any

# === FILE: autotrain.py ===
PAUSE_FILE = Path("autotrain.pause")

TRAINING_PATH = Path("data/training_data.csv")
BACKUP_PATH = Path("data/memory_backup.json")
SEED_TOPICS = [
    "history",
    "politics",
    "technology",
    "war",
    "famous people",
    "trends",
    "science",
    "finance",
    "sports",
    "culture",
]
QUESTION_TEMPLATES = [
    "Explain the latest developments in {}.",
    "Why is {} important today?",
    "Summarize recent news about {}.",
    "How does {} impact society?",
    "What is the future of {}?",
]


class SyntheticTrainer:
    def __init__(self):
        self.brain = AIBrain()
        self.trending = TrendingTopics()
        self.memory = QAMemory()
        self.evaluator = Evaluator()
        self.seen = set()
        self.buffer = []
        self.load_memory()

    def load_memory(self):
        if TRAINING_PATH.exists():
            with open(TRAINING_PATH, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = hashlib.sha1(row["question"].encode()).hexdigest()
                    self.seen.add(h)
        if BACKUP_PATH.exists():
            try:
                with open(BACKUP_PATH) as f:
                    data = json.load(f)
                    self.seen.update(data.get("hashes", []))
            except Exception:
                pass

    def save_progress(self):
        TRAINING_PATH.parent.mkdir(exist_ok=True)
        file_exists = TRAINING_PATH.exists()
        with open(TRAINING_PATH, "a", newline="") as f:
            fieldnames = [
                "question",
                "answer",
                "source",
                "token_count",
                "confidence",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.buffer)
        with open(BACKUP_PATH, "w") as f:
            json.dump({"hashes": list(self.seen)}, f, indent=2)
        self.buffer = []
        self.memory.prune()

    def generate_question(self) -> str:
        template = random.choice(QUESTION_TEMPLATES)
        topic = self.trending.random_topic()
        q_prompt = f"Write a short question about: {topic}"
        question = self.brain.ask(q_prompt)
        if len(question.split()) > 20:
            question = template.format(topic)
        return question

    def ask_question(self, question: str) -> tuple[str, str]:
        try:
            context = web_search(question)
            source = "DuckDuckGo"
        except Exception:
            context = ""
            source = "Ollama"
        prompt = f"{question}\n\nContext:\n{context}"
        answer = self.brain.ask(prompt)
        return answer, source

    def run(self):
        counter = len(self.seen)
        while True:
            if PAUSE_FILE.exists():
                time.sleep(1)
                continue
            question = self.generate_question()
            qhash = hashlib.sha1(question.encode()).hexdigest()
            if qhash in self.seen:
                continue
            try:
                answer, source = self.ask_question(question)
            except Exception as exc:
                print(f"Error fetching answer: {exc}")
                time.sleep(2)
                continue
            tokens = len(answer.split())
            if tokens < 10 or answer.startswith("[Error"):
                continue
            confidence = min(1.0, tokens / 100)
            self.buffer.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "token_count": tokens,
                    "confidence": f"{confidence:.2f}",
                }
            )
            score = self.evaluator.score(question, answer, source)
            self.memory.add(question, answer, source, score)
            self.evaluator.update_leaderboard(question, score)
            self.seen.add(qhash)
            counter += 1
            first_line = answer.splitlines()[0]
            print(f"[# {counter}] Q: {question}\n-> A: {first_line}\n\u2705 Learned from {source} ({tokens} tokens)")
            if len(self.buffer) >= 100:
                self.save_progress()
            time.sleep(1)


if __name__ == "__main__":
    trainer = SyntheticTrainer()
    trainer.run()
# === FILE: combine_files.py ===
IGNORED_DIRS = {'venv', '.venv', '__pycache__'}
SEARCH_DIRS = ['.']


def iter_py_files():
    py_files = []
    for base in SEARCH_DIRS:
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for name in files:
                if name.endswith('.py'):
                    path = os.path.relpath(os.path.join(root, name), '.')
                    py_files.append(path)
    root_files = sorted([f for f in py_files if '/' not in f])
    backend_files = sorted([f for f in py_files if f.startswith('backend/')])
    gui_files = sorted([f for f in py_files if f.startswith('gui/')])
    return root_files + backend_files + gui_files


def collect_imports_and_code(files):
    imports = []
    seen = set()
    sections = []
    for path in files:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        body_start = 0
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                if stripped not in seen:
                    imports.append(line.rstrip())
                    seen.add(stripped)
                body_start = idx + 1
            elif stripped == '' or stripped.startswith('#'):
                body_start = idx + 1
            else:
                break
        body = ''.join(lines[body_start:])
        sections.append((path, body))
    return imports, sections


def main():
    files = iter_py_files()
    imports, sections = collect_imports_and_code(files)
    with open('combined_jarvis.py', 'w', encoding='utf-8') as out:
        out.write('# Auto-generated combined file\n')
        for imp in imports:
            out.write(f'{imp}\n')
        out.write('\n')
        for path, body in sections:
            out.write(f'# === FILE: {path} ===\n')
            out.write(body)
            if not body.endswith('\n'):
                out.write('\n')


if __name__ == '__main__':
    main()
# === FILE: export_ollama_data.py ===
def collect() -> list[dict]:
    """Gather memory and audit logs into fine-tune messages."""
    records: list[dict] = []

    mem_path = Path("data/memory.json")
    if mem_path.exists():
        try:
            data = json.loads(mem_path.read_text())
            if isinstance(data, list):
                for item in data:
                    event = item.get("event")
                    if event:
                        records.append({
                            "messages": [
                                {"role": "system", "content": f"Memory: {event}"}
                            ]
                        })
        except Exception:
            pass

    audit_dir = Path("logs/self_audit")
    if audit_dir.exists():
        for log in sorted(audit_dir.glob("*.json")):
            try:
                items = json.loads(log.read_text())
                if isinstance(items, list):
                    for entry in items:
                        prompt = entry.get("prompt")
                        resp = entry.get("response")
                        if prompt and resp:
                            records.append({
                                "messages": [
                                    {"role": "user", "content": prompt},
                                    {"role": "assistant", "content": resp},
                                ]
                            })
            except Exception:
                continue

    stats_path = Path("data/strategy_stats.json")
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
            for name, val in stats.items():
                summary = (
                    f"{name} strategy: wins {val.get('wins', 0)}, "
                    f"losses {val.get('losses', 0)}, "
                    f"pnl {val.get('pnl', 0.0)}"
                )
                records.append({
                    "messages": [
                        {"role": "system", "content": summary}
                    ]
                })
        except Exception:
            pass

    qa_path = Path("data/qa_memory.json")
    if qa_path.exists():
        try:
            qas = json.loads(qa_path.read_text())
            if isinstance(qas, list):
                for qa in qas:
                    q = qa.get("question")
                    a = qa.get("answer")
                    if q and a:
                        records.append({
                            "messages": [
                                {"role": "user", "content": q},
                                {"role": "assistant", "content": a},
                            ]
                        })
        except Exception:
            pass

    return records


def main() -> None:
    records = collect()
    out = Path("training_data.jsonl")
    out.write_text("\n".join(json.dumps(r) for r in records))
    print(f"Wrote {len(records)} records to {out}")


if __name__ == "__main__":
    main()
# === FILE: keep_alive.py ===
app = Flask('')

@app.route('/')
def home():
    return "I'm alive"

def run():
    app.run(host='0.0.0.0', port=8080)


def keep_alive():
    t = Thread(target=run)
    t.start()
# === FILE: sum_files.py ===
EXTS = {'.py', '.md', '.json', '.txt', '.sh', '.csv'}


def iter_files(base: str):
    for root, _, files in os.walk(base):
        for name in files:
            if any(name.endswith(ext) for ext in EXTS):
                yield os.path.join(root, name)


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else '.'
    total_lines = 0
    total_bytes = 0
    for path in iter_files(base):
        try:
            with open(path, 'rb') as f:
                data = f.read()
        except Exception:
            continue
        total_bytes += len(data)
        total_lines += data.count(b'\n') + 1
    print(f'Total lines: {total_lines}')
    print(f'Total bytes: {total_bytes}')


if __name__ == '__main__':
    main()
# === FILE: backend/daily_report.py ===
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
# === FILE: backend/features/ai_brain.py ===
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.memory import MemoryManager
from .qa_memory import QAMemory
from .evaluator import Evaluator
from .web_search import web_search
from .engineering_expert import EngineeringExpert

openai = None


class AIBrain:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.memory = MemoryManager()
        self.qa_memory = QAMemory()
        self.evaluator = Evaluator()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            import openai as openai_lib

            openai_lib.api_key = self.api_key
            global openai
            openai = openai_lib
        self.client = None

    def solve_pdf(self, path: str) -> dict:
        expert = EngineeringExpert()
        return expert.solve_pdf_worksheet(path)

    def ask(self, prompt: str) -> str:
        self.memory.memory["last_prompt"] = prompt
        if prompt.lower().startswith("solve worksheet"):
            path = prompt.split(" ", 2)[-1]
            try:
                results = self.solve_pdf(path)
                answer = "\n\n".join(results.values())
                self.memory.memory["last_answer"] = answer
                self.memory.save()
                return answer
            except Exception as exc:
                return f"[Error solving worksheet: {exc}]"
        if EngineeringExpert.is_engineering_question(prompt):
            expert = EngineeringExpert()
            answer = expert.answer(prompt)
            self.memory.memory["last_answer"] = answer
            self.memory.save()
            score = self.evaluator.score(prompt, answer, "Ollama")
            self.qa_memory.add(prompt, answer, "Ollama", score)
            self.evaluator.update_leaderboard(prompt, score)
            self.qa_memory.prune()
            return answer
        try:
            if openai:
                response = openai.ChatCompletion.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}]
                )
                answer = (
                    response.choices[0].message["content"].strip()
                    if hasattr(response.choices[0], "message")
                    else response.choices[0].text.strip()
                )
            else:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "mistral", "prompt": prompt, "stream": False},
                )
                answer = response.json().get(
                    "response", "[No response from local model]"
                )
        except Exception as e:
            answer = f"[Error generating response: {e}]"

        self.memory.memory["last_answer"] = answer
        self.memory.save()
        score = self.evaluator.score(prompt, answer, "Ollama")
        # auto-correct low confidence answers using web search context
        if score < 0.5:
            try:
                context = web_search(prompt)
                improved_prompt = f"{prompt}\n\nContext:\n{context}"
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "mistral",
                        "prompt": improved_prompt,
                        "stream": False,
                    },
                )
                new_answer = resp.json().get("response", answer)
                new_score = self.evaluator.score(prompt, new_answer, "Ollama")
                if new_score > score:
                    answer = new_answer
                    score = new_score
            except Exception:
                pass

        self.qa_memory.add(prompt, answer, "Ollama", score)
        self.evaluator.update_leaderboard(prompt, score)
        self.qa_memory.prune()
        return answer
# === FILE: backend/features/autotrade.py ===
load_dotenv()

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
# === FILE: backend/features/dashboard.py ===
class TerminalDashboard(threading.Thread):
    def __init__(self, refresh: int = 2, audit=None):
        super().__init__(daemon=True)
        self.refresh = refresh
        self.stop_event = threading.Event()
        self.paused = False
        self.paused_since: float | None = None
        self.paused_total = 0.0
        self.log_msg = ""
        self.flash_until = 0.0
        self.memory = QAMemory()
        self.start_time = time.time()
        self.audit = audit
        self.success = 0
        self.fail = 0
        self.interactions: list[tuple[str, str]] = []
        self.last_interaction = time.time()

    def log_interaction(self, question: str, answer: str) -> None:
        self.interactions.append((question, answer))
        if len(self.interactions) > 5:
            self.interactions.pop(0)
        self.last_interaction = time.time()

    def _toggle_pause(self):
        pause_file = Path("autotrain.pause")
        if not self.paused:
            pause_file.write_text("1")
            self.paused_since = time.time()
            self.paused = True
            self.log_msg = "Paused autotrain"
        else:
            pause_file.unlink(missing_ok=True)
            if self.paused_since:
                self.paused_total += time.time() - self.paused_since
            self.paused_since = None
            self.paused = False
            self.log_msg = "Resumed autotrain"

    def _save_snapshot(self):
        path = Path("logs/memory_backup.json")
        path.parent.mkdir(exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.memory.data, f, indent=2)
        self.flash_until = time.time() + 2
        self.log_msg = "Snapshot saved"

    def _manual_search(self):
        if self.interactions:
            query = self.interactions[-1][0]
            web_search(query)
            self.log_msg = f"Search triggered for: {query[:30]}"

    def _clear_activity(self):
        self.interactions = []
        self.log_msg = "Activity cleared"

    def _color(self, score: float) -> int:
        if score >= 0.75:
            return curses.color_pair(1)
        if score >= 0.5:
            return curses.color_pair(2)
        return curses.color_pair(3)

    def _topic(self, question: str) -> str:
        words = question.lower().split()
        for token in reversed(words):
            token = token.strip(".,?!")
            if token:
                return token
        return ""

    def run(self):
        try:
            curses.wrapper(self.loop)
        except Exception:
            pass

    def loop(self, stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        stdscr.nodelay(True)
        while not self.stop_event.is_set():
            stdscr.erase()
            self.memory.load()
            total = len(self.memory.data)
            tokens = sum(e.get("tokens", 0) for e in self.memory.data)
            avg_conf = (
                sum(e.get("confidence", 0) for e in self.memory.data) / total
                if total
                else 0
            )
            now = time.time()
            review = [
                e
                for e in self.memory.data
                if now - e.get("timestamp", 0) > 7 * 86400
                or e.get("confidence", 1) < 0.5
            ]
            unique_q = {e["question"] for e in self.memory.data}
            dup_rate = 1 - (len(unique_q) / total) if total else 0
            learning_rate = total / ((time.time() - self.start_time) / 60 + 1e-6)
            active = self.memory.data[-1]["source"] if self.memory.data else "N/A"

            mode = "Paused" if self.paused else "Active"
            if self.paused:
                elapsed = int(time.time() - (self.paused_since or time.time()))
                mode += f" ({elapsed}s)"

            stdscr.addstr(0, 0, "=== JARVIS Console ===", curses.A_BOLD)
            stdscr.addstr(1, 0, f"Status: {mode}")

            row = 3
            stdscr.addstr(row, 0, "Learning", curses.A_UNDERLINE)
            row += 1
            stdscr.addstr(row, 2, f"Success: {self.success}", curses.color_pair(1))
            row += 1
            stdscr.addstr(row, 2, f"Failures: {self.fail}", curses.color_pair(3))

            row += 2
            stdscr.addstr(row, 0, "Memory", curses.A_UNDERLINE)
            row += 1
            stdscr.addstr(row, 2, f"Total stored: {total}")
            row += 1
            stdscr.addstr(row, 2, f"Avg confidence: {avg_conf:.2f}", self._color(avg_conf))
            row += 1
            stdscr.addstr(row, 2, f"Learning rate: {learning_rate:.2f}/min")
            row += 1
            stdscr.addstr(row, 2, f"Token usage: {tokens}")
            row += 1
            stdscr.addstr(row, 2, f"Duplicate rate: {dup_rate:.2f}")
            row += 1
            color_pruned = curses.color_pair(3) if self.memory.pruned_total else curses.color_pair(1)
            stdscr.addstr(row, 2, f"Pruned: {self.memory.pruned_total}", color_pruned)
            row += 1
            color_review = curses.color_pair(2) if len(review) else curses.color_pair(1)
            stdscr.addstr(row, 2, f"Needs review: {len(review)}", color_review)
            row += 1
            stdscr.addstr(row, 2, f"Active source: {active}")

            topic_counts = {}
            for e in review:
                t = self._topic(e.get("question", ""))
                if t:
                    topic_counts[t] = topic_counts.get(t, 0) + 1
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            topics_str = ", ".join(t for t, _ in top_topics) if top_topics else "None"
            row += 1
            stdscr.addstr(row, 2, f"Top outdated topics: {topics_str}")

            row += 2
            stdscr.addstr(row, 0, "Audit", curses.A_UNDERLINE)
            status = "N/A"
            if self.audit:
                status = f"{self.audit.checked}/{self.audit.check_total} checked, {self.audit.updated_last} updated"
            row += 1
            stdscr.addstr(row, 2, status)

            row += 2
            stdscr.addstr(row, 0, "Recent", curses.A_UNDERLINE)
            for i, (q, a) in enumerate(reversed(self.interactions[-3:])):
                row += 1
                stdscr.addstr(row, 2, f"Q: {q}")
                row += 1
                stdscr.addstr(row, 4, f"A: {a.splitlines()[0][:50]}")

            row += 1
            if time.time() - self.last_interaction > self.refresh * 2:
                stdscr.addstr(row, 0, "Awaiting questions...", curses.A_DIM)
            else:
                stdscr.addstr(row, 0, " " * 20)

            max_y, max_x = stdscr.getmaxyx()
            controls = "[p] pause/resume  [m] snapshot  [r] search  [c] clear log  [q] quit"
            stdscr.addstr(max_y - 3, 0, controls[: max_x - 1])
            attr = curses.A_REVERSE if self.flash_until > time.time() else curses.A_DIM
            stdscr.addstr(max_y - 2, 0, self.log_msg[: max_x - 1], attr)
            stdscr.refresh()

            ch = stdscr.getch()
            if ch != -1:
                try:
                    ch = chr(ch)
                except ValueError:
                    ch = ""
                if ch == "p":
                    self._toggle_pause()
                elif ch == "m":
                    self._save_snapshot()
                elif ch == "r":
                    self._manual_search()
                elif ch == "c":
                    self._clear_activity()
                elif ch == "q":
                    self.log_msg = "Dashboard closed"
                    self.stop_event.set()
                    break
            time.sleep(self.refresh)

    def stop(self):
        self.stop_event.set()
# === FILE: backend/features/engineering_expert.py ===
try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    bpy = None

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    go = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .web_search import web_search
from .qa_memory import QAMemory
from .evaluator import Evaluator
from utils.memory import MemoryManager

BLUEPRINT_DIR = "blueprints"
os.makedirs(BLUEPRINT_DIR, exist_ok=True)
SIMULATION_DIR = "simulations"
os.makedirs(SIMULATION_DIR, exist_ok=True)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


class EngineeringExpert:
    """Domain expert for answering engineering questions."""

    FIELD_KEYWORDS = {
        "mechanical": [
            "mechanical",
            "thermodynamics",
            "kinematics",
            "materials",
            "machine",
            "gear",
            "fluid",
        ],
        "civil": [
            "civil",
            "structure",
            "bridge",
            "beam",
            "soil",
            "transport",
            "foundation",
        ],
        "electrical": [
            "electrical",
            "circuit",
            "control",
            "signal",
            "power",
            "electronics",
        ],
        "computer": [
            "computer",
            "algorithm",
            "architecture",
            "embedded",
            "software",
            "hardware",
        ],
        "chemical": [
            "chemical",
            "process",
            "reaction",
            "kinetics",
            "thermodynamics",
            "distillation",
        ],
        "aerospace": [
            "aerospace",
            "rocket",
            "aircraft",
            "flight",
            "aerodynamics",
            "satellite",
        ],
        "nuclear": [
            "nuclear",
            "reactor",
            "radiation",
            "neutron",
            "fission",
        ],
        "industrial": [
            "industrial",
            "manufacturing",
            "operations",
            "logistics",
            "optimization",
        ],
        "biomedical": [
            "biomedical",
            "medical",
            "prosthetic",
            "biomaterial",
            "tissue",
        ],
    }

    def __init__(self):
        self.evaluator = Evaluator()
        self.engineering_memory = QAMemory(path="data/engineering_memory.json")
        self.memory = MemoryManager(path="data/engineering_insights.json")
        self.textbook_memory = MemoryManager(path="data/engineering_textbooks.json")
        self.formula_index = MemoryManager(path="data/formula_index.json")
        self.sim_index = MemoryManager(path="data/simulation_index.json")

    def _index_formula(self, formula: str, tags: List[str], steps: str) -> None:
        entry = {
            "formula": formula,
            "tags": tags,
            "steps": steps,
            "timestamp": time.time(),
        }
        formulas = self.formula_index.memory.setdefault("formulas", [])
        formulas.append(entry)
        self.formula_index.save()

    def _find_similar_formula(self, formula: str) -> dict | None:
        """Return the most similar indexed formula entry if similarity > 0.8."""
        best = None
        best_score = 0.0
        for entry in self.formula_index.memory.get("formulas", []):
            existing = entry.get("formula", "")
            score = SequenceMatcher(None, existing, formula).ratio()
            if score > best_score:
                best = entry
                best_score = score
        if best_score > 0.8:
            return best
        return None

    @staticmethod
    def _is_math_problem(query: str) -> bool:
        q = query.lower()
        triggers = ["integrate", "derivative", "solve", "=", "d/d"]
        return any(t in q for t in triggers)

    def _solve_symbolically(self, query: str) -> str:
        try:
            q = query.lower()
            if "integrate" in q:
                expr_str = q.split("integrate", 1)[1]
                formula = f"∫ {expr_str}"
                cached = self._find_similar_formula(formula)
                if cached:
                    return cached.get("steps", cached["formula"])
                expr = sp.sympify(expr_str)
                result = sp.integrate(expr)
                simplified = sp.simplify(result)
                steps = f"Integrate {expr_str}\n\\boxed{{{simplified}}}"
                self._index_formula(formula, expr_str.split(), steps)
                return steps
            if "derivative" in q or "d/d" in q:
                expr_str = (
                    q.split("derivative of", 1)[1]
                    if "derivative of" in q
                    else q.split("d/dx", 1)[1]
                )
                x = sp.symbols("x")
                formula = f"d({expr_str})/dx"
                cached = self._find_similar_formula(formula)
                if cached:
                    return cached.get("steps", cached["formula"])
                expr = sp.sympify(expr_str)
                result = sp.diff(expr, x)
                simplified = sp.simplify(result)
                steps = f"Differentiate {expr_str} w.r.t x\n\\boxed{{{simplified}}}"
                self._index_formula(formula, expr_str.split(), steps)
                return steps
            if "solve" in q:
                m = re.search(r"solve (.+?)=([^ ]+) for ([a-zA-Z])", q)
                if m:
                    left, right, var = m.groups()
                    symbol = sp.symbols(var)
                    formula = f"{left}={right}"
                    cached = self._find_similar_formula(formula)
                    if cached:
                        return cached.get("steps", cached["formula"])
                    equation = sp.Eq(sp.sympify(left), sp.sympify(right))
                    solution = sp.solve(equation, symbol)
                    steps = f"Solve {left} = {right} for {var}\n\\boxed{{{solution}}}"
                    self._index_formula(formula, [var], steps)
                    return steps
            if "=" in q:
                left, right = q.split("=", 1)
                vars_ = list(
                    sp.sympify(left).free_symbols | sp.sympify(right).free_symbols
                )
                if vars_:
                    formula = f"{left}={right}"
                    cached = self._find_similar_formula(formula)
                    if cached:
                        return cached.get("steps", cached["formula"])
                    solution = sp.solve(
                        sp.Eq(sp.sympify(left), sp.sympify(right)), vars_
                    )
                    steps = f"Solve {left} = {right} for {', '.join(map(str, vars_))}\n\\boxed{{{solution}}}"
                    self._index_formula(formula, [str(v) for v in vars_], steps)
                    return steps
        except Exception as exc:
            return f"[Error solving symbolically: {exc}]"
        return "[Unable to solve symbolically]"

    @staticmethod
    def _is_blueprint_request(query: str) -> bool:
        q = query.lower()
        return any(w in q for w in ["draw", "blueprint", "diagram"])

    @staticmethod
    def _is_simulation_request(query: str) -> bool:
        q = query.lower()
        keywords = [
            "simulate",
            "simulation",
            "stress",
            "current",
            "voltage",
            "thermal",
            "heat",
        ]
        return any(k in q for k in keywords)

    def _generate_blueprint(self, query: str) -> str:
        q = query.lower()
        if "truss" in q:
            joints = 3
            width = joints
            height = 1
            jm = re.search(r"(\d+)[- ]*beam", q)
            if jm:
                joints = int(jm.group(1)) + 1
            m = re.search(r"(\d+)\s*joints?", q)
            if m:
                joints = int(m.group(1))
            dm = re.search(r"(\d+(?:\.\d+)?)\s*[xby]\s*(\d+(?:\.\d+)?)", q)
            if dm:
                width = float(dm.group(1))
                height = float(dm.group(2))
            return self._draw_truss(joints, width, height)
        if "beam" in q:
            spans = 1
            length = spans
            m = re.search(r"(\d+)\s*spans?", q)
            if m:
                spans = int(m.group(1))
            dm = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters)?", q)
            if dm:
                length = float(dm.group(1))
            return self._draw_beam(spans, length)
        if "circuit" in q:
            comps = 2
            m = re.search(r"(\d+)\s*(?:resistors|components)", q)
            if m:
                comps = int(m.group(1))
            return self._draw_circuit(comps)
        if "pcb" in q:
            comps = 2
            m = re.search(r"(\d+)\s*components", q)
            if m:
                comps = int(m.group(1))
            return self._draw_pcb(comps)
        return "[Blueprint request not understood]"

    def _draw_truss(self, joints: int, width: float, height: float) -> str:
        fig, ax = plt.subplots()
        x = np.linspace(0, width, joints)
        ax.plot(x, [0] * joints, "ko-")
        for i in range(joints - 2):
            ax.plot([x[i], x[i + 1]], [0, height], "k-")
            ax.plot([x[i + 1], x[i + 2]], [0, height], "k-")
        ax.axis("equal")
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = f"truss with {joints} joints"
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    def _draw_beam(self, spans: int, length: float) -> str:
        fig, ax = plt.subplots()
        x = np.linspace(0, length, spans + 1)
        ax.plot([0, length], [0, 0], "k-", lw=2)
        for pos in x:
            ax.plot([pos, pos], [0, -0.2], "k-")
        ax.axis("equal")
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = f"beam with {spans} spans"
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    def _draw_circuit(self, components: int) -> str:
        fig, ax = plt.subplots()
        x = np.linspace(0, components, components + 1)
        for i in range(components):
            ax.plot([x[i], x[i + 1]], [0, 0], "k-")
            ax.text((x[i] + x[i + 1]) / 2, 0.1, f"R{i + 1}", ha="center")
        ax.plot([0, 0], [0, 1], "k-")
        ax.plot([components, components], [0, 1], "k-")
        ax.plot([0, components], [1, 1], "k-")
        ax.axis("equal")
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = (
            f"circuit with {components} components"
        )
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    def _draw_pcb(self, components: int) -> str:
        fig, ax = plt.subplots()
        for i in range(components):
            rect = plt.Rectangle(
                (i, i % 2), 0.8, 0.4, edgecolor="black", facecolor="lightgray"
            )
            ax.add_patch(rect)
        ax.set_xlim(0, components)
        ax.set_ylim(0, 2)
        ax.axis("off")
        path = os.path.join(BLUEPRINT_DIR, f"blueprint_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.memory.memory["last_blueprint"] = path
        self.memory.memory["last_blueprint_prompt"] = (
            f"pcb with {components} components"
        )
        self.memory.save()
        return f"Blueprint saved to {path}\n![blueprint]({path})"

    @classmethod
    def is_engineering_question(cls, query: str) -> bool:
        q = query.lower()
        if "engineer" in q:
            return True
        for words in cls.FIELD_KEYWORDS.values():
            for kw in words:
                if kw in q:
                    return True
        return False

    def answer(self, query: str) -> str:
        if self._is_simulation_request(query):
            answer = self.simulate(query)
        elif self._is_blueprint_request(query):
            answer = self._generate_blueprint(query)
        elif self._is_math_problem(query):
            answer = self._solve_symbolically(query)
        else:
            field = self._detect_field(query)
            method: Callable[[str], str] = getattr(
                self, f"_answer_{field}", self._generic_answer
            )
            answer = method(query)
        score = self.evaluator.score(query, answer, "Ollama")
        self.engineering_memory.add(query, answer, "Ollama", score)
        self.memory.memory[query] = {"answer": answer, "score": score}
        self.memory.save()
        return answer

    def _detect_field(self, query: str) -> str:
        q = query.lower()
        best_field = ""
        best_matches = 0
        for field, kws in self.FIELD_KEYWORDS.items():
            matches = sum(1 for kw in kws if kw in q)
            if matches > best_matches:
                best_matches = matches
                best_field = field
        return best_field

    def _format_worksheet_answer(self, question: str, solution: str) -> str:
        """Return tutor-style formatted solution string."""
        final_line = solution.splitlines()[-1]
        boxed = f"**{final_line}**"
        return f"**Question:** {question}\n{solution}\n\n{boxed}"

    def ingest_pdf(self, path: str, discipline: str) -> None:
        """Parse a PDF textbook and store content per discipline."""
        doc = fitz.open(path)
        chapters = []
        current = {"title": "Introduction", "formulas": [], "content": ""}
        for page in doc:
            text = page.get_text()
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("chapter"):
                    if current["content"] or current["formulas"]:
                        chapters.append(current)
                    current = {"title": line, "formulas": [], "content": ""}
                elif "=" in line and any(c.isalpha() for c in line):
                    current["formulas"].append(line)
                else:
                    if current["content"]:
                        current["content"] += " " + line
                    else:
                        current["content"] = line
        chapters.append(current)
        self.textbook_memory.memory.setdefault(discipline, []).extend(chapters)
        self.textbook_memory.save()

    def solve_pdf_worksheet(self, path: str) -> dict:
        """Solve each numbered problem in a PDF worksheet."""
        doc = fitz.open(path)
        results = {}
        pattern = re.compile(r"^\d+\.\s+")
        for page in doc:
            lines = page.get_text().splitlines()
            current = ""
            for line in lines:
                if pattern.match(line.strip()):
                    if current:
                        sol = self.answer(current)
                        results[current] = self._format_worksheet_answer(current, sol)
                    current = pattern.sub("", line.strip())
                else:
                    current += " " + line.strip()
            if current:
                sol = self.answer(current)
                results[current] = self._format_worksheet_answer(current, sol)
                current = ""
        return results

    def _textbook_context(self, discipline: str, query: str) -> str:
        """Retrieve relevant textbook snippets for the query."""
        data = self.textbook_memory.memory.get(discipline, [])
        q = query.lower()
        snippets = []
        for chap in data:
            if q in chap.get("title", "").lower():
                snippets.append(chap.get("content", ""))
                snippets.extend(chap.get("formulas", []))
                continue
            if q in chap.get("content", "").lower():
                snippets.append(chap.get("content", ""))
            for formula in chap.get("formulas", []):
                if q in formula.lower():
                    snippets.append(formula)
        return "\n".join(snippets[:5])

    def _generic_answer(self, query: str, field: str = "engineering") -> str:
        discipline = field.split()[0]
        context = self._textbook_context(discipline, query)
        if not context:
            context = web_search(f"{query} {field}")
        if (
            context
            and "Web search error" not in context
            and "No results" not in context
        ):
            prompt = f"{query}\n\nContext:\n{context}"
        else:
            prompt = query
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "mistral", "prompt": prompt, "stream": False},
                timeout=10,
            )
            return resp.json().get("response", "")
        except Exception as exc:
            return f"[Error generating response: {exc}]"

    def _answer_mechanical(self, query: str) -> str:
        return self._generic_answer(query, "mechanical engineering")

    def _answer_civil(self, query: str) -> str:
        return self._generic_answer(query, "civil engineering")

    def _answer_electrical(self, query: str) -> str:
        return self._generic_answer(query, "electrical engineering")

    def _answer_computer(self, query: str) -> str:
        return self._generic_answer(query, "computer engineering")

    def _answer_chemical(self, query: str) -> str:
        return self._generic_answer(query, "chemical engineering")

    def _answer_aerospace(self, query: str) -> str:
        return self._generic_answer(query, "aerospace engineering")

    def _answer_nuclear(self, query: str) -> str:
        return self._generic_answer(query, "nuclear engineering")

    def _answer_industrial(self, query: str) -> str:
        return self._generic_answer(query, "industrial engineering")

    def _answer_biomedical(self, query: str) -> str:
        return self._generic_answer(query, "biomedical engineering")

    # --- Simulation Capabilities ---
    def simulate(
        self, prompt: str, params: dict | None = None, tags: list[str] | None = None
    ) -> str:
        """Run a simple physics simulation based on the prompt."""
        if params:
            extras = " " + " ".join(f"{k}={v}" for k, v in params.items())
            prompt += extras
        q = prompt.lower()
        field = self._detect_field(q)
        if "flow" in q or "fluid" in q:
            method = self._simulate_fluid_flow
        else:
            method = getattr(self, f"_simulate_{field}", self._simulate_mechanical)
        result_tuple = method(q)
        model = None
        margin = 0.0
        if len(result_tuple) == 5:
            result, path, model, fail, margin = result_tuple
        elif len(result_tuple) == 4:
            result, path, fail, margin = result_tuple
        else:
            result, path, fail = result_tuple
        self.engineering_memory.add(prompt, result, "Simulation", 1.0, tags or ["simulation"])
        self.memory.memory["last_simulation"] = path
        if model:
            self.memory.memory["last_model"] = model
        self.memory.save()
        entry = {
            "uuid": str(uuid4()),
            "prompt": prompt,
            "field": field,
            "result": result,
            "path": path,
            "model": model,
            "timestamp": time.time(),
            "params": params or {},
            "performance": margin,
            "tags": tags or [],
        }
        sims = self.sim_index.memory.setdefault("simulations", [])
        sims.append(entry)
        self.sim_index.save()
        msg = f"{result}\nSimulation saved to {path}\n![simulation]({path})"
        if model:
            msg += f"\n3D model saved to {model}"
        if fail:
            msg += f"\n**Failure detected. Suggestion:** {self._suggest_fix(field)}"
        msg += f"\nPerformance score: {margin:.2f}"
        return msg

    def optimize(
        self,
        sim_type: str,
        objective: str,
        ranges: dict,
        iterations: int = 20,
    ) -> dict:
        """Simple random search optimization for a simulation."""
        import random

        best: dict | None = None
        best_score = float("inf")
        for _ in range(iterations):
            params = {
                k: random.uniform(v[0], v[1]) if isinstance(v, (list, tuple)) else v
                for k, v in ranges.items()
            }
            result_msg = self.simulate(sim_type, params)
            entry = self.sim_index.memory.get("simulations", [])[-1]
            perf = entry.get("performance", 0)
            score = abs(perf)
            if score < best_score:
                best_score = score
                best = {"params": params, "result": result_msg, "score": perf}
        return best or {}

    def design_assistant(self, description: str) -> str:
        """Heuristic-based design suggestions."""
        desc = description.lower()
        material = "steel" if "steel" in desc else "aluminum"
        span = re.search(r"(\d+(?:\.\d+)?)\s*m", desc)
        load = re.search(r"(\d+(?:\.\d+)?)\s*n", desc)
        length = float(span.group(1)) if span else 1.0
        force = float(load.group(1)) if load else 1000.0
        suggestion = (
            f"Material: {material}\nCross-section height: 0.1 m\n"
            f"Estimated span: {length} m under {force} N"
        )
        return suggestion

    def analysis_chain(self, description: str) -> str:
        """Run blueprint, simulation and optimization in sequence."""
        blueprint = self._generate_blueprint(description)
        sim_res = self.simulate(description)
        opt = self.optimize("mechanical", "stress", {"length": [1, 5], "force": [500, 2000]})
        return f"{blueprint}\n\n{sim_res}\n\nBest config: {opt}"

    def _suggest_fix(self, field: str) -> str:
        fixes = {
            "mechanical": "Use stronger material or add support",
            "civil": "Use I-beam instead of flat bar",
            "electrical": "Add a heat sink",
            "chemical": "Lower temperature or pressure",
            "aerospace": "Increase wing area",
            "nuclear": "Insert control rods",
            "computer": "Improve cooling",
        }
        return fixes.get(field, "Review design parameters")

    def _create_beam_model(self, length: float, height: float) -> str:
        """Create a simple 3D beam model using Blender and export as GLB."""
        if bpy is None:  # pragma: no cover - optional dependency
            return ""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        mesh = bpy.data.meshes.new("Beam")
        verts = [
            (0, 0, 0),
            (length, 0, 0),
            (length, height, 0),
            (0, height, 0),
            (0, 0, height / 10),
            (length, 0, height / 10),
            (length, height, height / 10),
            (0, height, height / 10),
        ]
        faces = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        mesh.from_pydata(verts, [], faces)
        obj = bpy.data.objects.new("Beam", mesh)
        bpy.context.scene.collection.objects.link(obj)
        glb_path = os.path.join(MODEL_DIR, f"beam_{int(time.time()*1000)}.glb")
        bpy.ops.export_scene.gltf(filepath=glb_path, export_format="GLB")
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return glb_path

    def _simulate_mechanical(self, q: str) -> Tuple[str, str, str, bool, float]:
        length = 1.0
        force = 1000.0
        h = 0.1
        m = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if m:
            length = float(m.group(1))
        fm = re.search(r"(\d+(?:\.\d+)?)\s*n", q)
        if fm:
            force = float(fm.group(1))
        hm = re.search(r"h=(\d+(?:\.\d+)?)", q)
        if hm:
            h = float(hm.group(1))
        x = np.linspace(0, length, 100)
        inertia = max(h ** 4 / 12, 1e-6)
        stress = force * (length - x) * (h / 2) / inertia
        fig, ax = plt.subplots()
        ax.plot(x, stress)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Stress")
        img_path = os.path.join(
            SIMULATION_DIR, f"mechanical_{int(time.time()*1000)}.png"
        )
        fig.savefig(img_path, bbox_inches="tight")
        plt.close(fig)
        model_path = ""
        try:
            model_path = self._create_beam_model(length, h)
        except Exception:
            model_path = ""
        max_stress = float(stress.max())
        fail = max_stress > 2.5e8
        result = f"Max stress: {max_stress:.2f} Pa"
        margin = 2.5e8 - max_stress
        return result, img_path, model_path, fail, margin

    def _simulate_electrical(self, q: str) -> Tuple[str, str, bool, float]:
        voltage = 5.0
        resistors = re.findall(r"(\d+(?:\.\d+)?)\s*ohm", q)
        values = [float(r) for r in resistors] if resistors else [1.0, 1.0]
        vm = re.search(r"(\d+(?:\.\d+)?)\s*v", q)
        if vm:
            voltage = float(vm.group(1))
        total_r = sum(values)
        current = voltage / total_r
        drops = [current * r for r in values]
        fig, ax = plt.subplots()
        ax.bar(range(1, len(drops) + 1), drops)
        ax.set_xlabel("Resistor")
        ax.set_ylabel("Voltage Drop (V)")
        path = os.path.join(SIMULATION_DIR, f"electrical_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = current > 10
        result = f"Circuit current: {current:.2f} A"
        margin = 10 - current
        return result, path, fail, margin

    def _simulate_thermal(self, q: str) -> Tuple[str, str, bool, float]:
        length = 1.0
        t1, t2 = 100.0, 0.0
        lm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if lm:
            length = float(lm.group(1))
        tmatch = re.findall(r"(\d+(?:\.\d+)?)\s*c", q)
        if len(tmatch) >= 2:
            t1, t2 = float(tmatch[0]), float(tmatch[1])
        x = np.linspace(0, length, 50)
        temp = t1 + (t2 - t1) * x / length
        fig, ax = plt.subplots()
        ax.plot(x, temp)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Temperature (C)")
        path = os.path.join(SIMULATION_DIR, f"thermal_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        max_temp = max(t1, t2)
        fail = max_temp > 500
        result = f"Temperature range: {t1}C to {t2}C"
        margin = 500 - max_temp
        return result, path, fail, margin

    def _simulate_civil(self, q: str) -> Tuple[str, str, bool, float]:
        load = 1e4
        span = 5.0
        lm = re.search(r"(\d+(?:\.\d+)?)\s*ton", q)
        if lm:
            load = float(lm.group(1)) * 9.81e3
        sm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if sm:
            span = float(sm.group(1))
        x = np.linspace(0, span, 50)
        E = 2e11
        moment_of_inertia = 1e-4
        y = load * x**2 * (3 * span - x) / (6 * E * moment_of_inertia)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Deflection (m)")
        path = os.path.join(SIMULATION_DIR, f"civil_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        max_def = float(y.max())
        allow = span / 100
        fail = max_def > allow
        result = f"Max deflection: {max_def:.4f} m"
        margin = allow - max_def
        return result, path, fail, margin

    def _simulate_aerospace(self, q: str) -> Tuple[str, str, bool, float]:
        speed = 50.0
        m = re.search(r"(\d+(?:\.\d+)?)\s*m/s", q)
        if m:
            speed = float(m.group(1))
        rho = 1.225
        area = 1.0
        lift = 0.5 * rho * speed**2 * area
        v = np.linspace(0, speed, 50)
        fig, ax = plt.subplots()
        ax.plot(v, 0.5 * rho * v**2 * area)
        ax.set_xlabel("Speed (m/s)")
        ax.set_ylabel("Lift (N)")
        path = os.path.join(SIMULATION_DIR, f"aero_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = lift < 1e3
        result = f"Lift at {speed} m/s: {lift:.1f} N"
        margin = lift - 1e3
        return result, path, fail, margin

    def _simulate_chemical(self, q: str) -> Tuple[str, str, bool, float]:
        temp = 300.0
        pressure = 1.0
        tm = re.search(r"(\d+(?:\.\d+)?)\s*c", q)
        if tm:
            temp = float(tm.group(1))
        pm = re.search(r"(\d+(?:\.\d+)?)\s*atm", q)
        if pm:
            pressure = float(pm.group(1))
        k = np.exp(-(5000) / (8.314 * (temp + 273.15)))
        fig, ax = plt.subplots()
        t = np.linspace(0, 10, 100)
        conc = np.exp(-k * t)
        ax.plot(t, conc)
        ax.set_xlabel("Time")
        ax.set_ylabel("Concentration")
        path = os.path.join(SIMULATION_DIR, f"chemical_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = temp > 400 or pressure > 5
        result = f"Rate constant: {k:.4f}"
        margin = 400 - temp
        return result, path, fail, margin

    def _simulate_nuclear(self, q: str) -> Tuple[str, str, bool, float]:
        flux = 1e12
        m = re.search(r"(\d+(?:\.\d+)?)\s*n/s", q)
        if m:
            flux = float(m.group(1))
        t = np.linspace(0, 10, 100)
        decay = np.exp(-0.1 * t)
        fig, ax = plt.subplots()
        ax.plot(t, decay)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Activity")
        path = os.path.join(SIMULATION_DIR, f"nuclear_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = flux > 1e14
        result = f"Neutron flux: {flux:.2e} n/s"
        margin = 1e14 - flux
        return result, path, fail, margin

    def _simulate_computer(self, q: str) -> Tuple[str, str, bool, float]:
        freq = 1.0
        fm = re.search(r"(\d+(?:\.\d+)?)\s*ghz", q)
        if fm:
            freq = float(fm.group(1))
        ops = freq * 1e9
        t = np.linspace(0, 1, 50)
        throughput = ops * t
        fig, ax = plt.subplots()
        ax.plot(t, throughput)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Operations")
        path = os.path.join(SIMULATION_DIR, f"computer_{int(time.time()*1000)}.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        fail = freq > 5
        result = f"Throughput: {ops:.2e} ops/s"
        margin = 5 - freq
        return result, path, fail, margin

    def _simulate_fluid_flow(self, q: str) -> Tuple[str, str, bool, float]:
        """Generate a simple laminar flow visualization using Plotly."""
        width = 1.0
        height = 1.0
        wm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if wm:
            width = float(wm.group(1))
        hm = re.search(r"(\d+(?:\.\d+)?)\s*m", q)
        if hm:
            height = float(hm.group(1))
        x = np.linspace(0, width, 30)
        y = np.linspace(0, height, 30)
        X, Y = np.meshgrid(x, y)
        vmax = 1.0
        velocity = 4 * vmax * (Y / height) * (1 - Y / height)
        if go is None:  # pragma: no cover - optional dependency
            fig, ax = plt.subplots()
            c = ax.pcolormesh(X, Y, velocity, shading="auto")
            fig.colorbar(c, ax=ax)
            img_path = os.path.join(
                SIMULATION_DIR, f"fluid_{int(time.time()*1000)}.png"
            )
            fig.savefig(img_path, bbox_inches="tight")
            plt.close(fig)
        else:
            fig = go.Figure(data=go.Heatmap(z=velocity, x=x, y=y, colorscale="Viridis"))
            img_path = os.path.join(
                SIMULATION_DIR, f"fluid_{int(time.time()*1000)}.png"
            )
            try:
                fig.write_image(img_path)
            except Exception:
                fig.write_html(img_path.replace(".png", ".html"))
        max_vel = float(velocity.max())
        fail = False
        result = f"Max velocity: {max_vel:.2f} m/s"
        margin = 1.0 - max_vel
        return result, img_path, fail, margin
# === FILE: backend/features/evaluator.py ===
SOURCE_WEIGHT = {
    "DuckDuckGo": 1.0,
    "Bing": 0.8,
    "Ollama": 0.5,
}


class Evaluator:
    def __init__(self, leaderboard: str = "data/leaderboard.csv"):
        self.board_path = Path(leaderboard)
        if not self.board_path.exists():
            self.board_path.write_text("question,score\n")
        self.memory = QAMemory()

    def score(self, question: str, answer: str, source: str) -> float:
        tokens = len(answer.split())
        token_score = min(tokens / 100, 1.0)
        keywords = set(question.lower().split())
        overlap = sum(1 for w in answer.lower().split() if w in keywords)
        keyword_score = overlap / (len(keywords) or 1)
        source_score = SOURCE_WEIGHT.get(source, 0.5)

        # originality compared to existing memory
        max_sim = 0.0
        for entry in self.memory.data:
            ratio = SequenceMatcher(None, entry.get("answer", ""), answer).ratio()
            if ratio > max_sim:
                max_sim = ratio
        originality = 1 - max_sim

        score = (token_score * 0.4) + (keyword_score * 0.3) + (source_score * 0.1) + (originality * 0.2)
        return round(score, 4)

    def update_leaderboard(self, question: str, score: float):
        entries = []
        if self.board_path.exists():
            with open(self.board_path, newline="") as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        entries.append({"question": question, "score": score})
        entries.sort(key=lambda x: float(x["score"]), reverse=True)
        with open(self.board_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "score"])
            writer.writeheader()
            writer.writerows(entries[:100])
# === FILE: backend/features/qa_memory.py ===
class QAMemory:
    def __init__(self, path: str = "data/qa_memory.json"):
        self.path = Path(path)
        self.data = []
        self.pruned_total = 0
        self.load()

    def load(self):
        self.pruned_total = 0
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []
        self.prune()

    def save(self):
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _is_duplicate(self, prompt: str) -> bool:
        for entry in self.data:
            if entry.get("question") == prompt:
                return True
        return False

    def add(
        self,
        question: str,
        answer: str,
        source: str,
        confidence: float,
        tags: list[str] | None = None,
    ) -> None:
        tokens = len(answer.split())
        if tokens < 10:
            return
        if confidence < 0.5 and self._is_duplicate(question):
            return
        entry = {
            "question": question,
            "answer": answer,
            "source": source,
            "tokens": tokens,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        if tags:
            entry["tags"] = tags
        self._replace_outdated(entry)
        if not self._is_duplicate(question):
            self.data.append(entry)
        self.save()
        self.prune()

    def _replace_outdated(self, new_entry: dict):
        year_tokens = [t for t in new_entry["answer"].split() if t.isdigit() and len(t) == 4]
        if not year_tokens:
            return
        latest_year = max(map(int, year_tokens))
        for idx, entry in enumerate(list(self.data)):
            old_years = [t for t in entry["answer"].split() if t.isdigit() and len(t) == 4]
            if old_years and max(map(int, old_years)) < latest_year:
                if SequenceMatcher(None, entry["question"], new_entry["question"]).ratio() > 0.6:
                    self.data[idx] = new_entry

    def prune(self):
        seen = {}
        for entry in sorted(self.data, key=lambda e: e.get("timestamp", 0)):
            if entry.get("tokens", 0) < 10:
                self.pruned_total += 1
                continue
            key = entry["question"]
            if key in seen:
                if entry.get("timestamp", 0) > seen[key].get("timestamp", 0):
                    seen[key] = entry
                    self.pruned_total += 1
            else:
                seen[key] = entry
        self.data = list(seen.values())
        self.save()

    def get_random(self):
        if not self.data:
            return None
        import random
        return random.choice(self.data)
# === FILE: backend/features/self_audit.py ===
class SelfAudit(threading.Thread):
    """Nightly audit that re-checks all stored Q&A."""

    def __init__(self, interval: int = 24 * 3600, review_days: int = 7):
        super().__init__(daemon=True)
        self.interval = interval
        self.review_age = review_days * 86400
        self.stop_event = threading.Event()
        self.brain = AIBrain()
        self.memory = QAMemory()
        self.evaluator = Evaluator()
        self.check_total = 0
        self.checked = 0
        self.updated_last = 0

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.audit()
            self.stop_event.wait(self.interval)

    def audit(self) -> None:
        """Evaluate all memory entries and refresh low-quality answers."""
        self.memory.load()
        now = time.time()
        changed = False
        self.check_total = len(self.memory.data)
        self.checked = 0
        self.updated_last = 0
        for i, entry in enumerate(list(self.memory.data)):
            score = entry.get("confidence")
            if score is None:
                score = self.evaluator.score(entry["question"], entry["answer"], entry["source"])
            age = now - entry.get("timestamp", 0)
            if age < self.review_age and score >= 0.5:
                self.checked += 1
                continue

            try:
                context = web_search(entry["question"])
            except Exception:
                context = ""
            prompt = f"{entry['question']}\n\nContext:\n{context}"
            new_answer = self.brain.ask(prompt)
            new_score = self.evaluator.score(entry["question"], new_answer, "Ollama")
            if new_score > score:
                self.memory.data[i] = {
                    "question": entry["question"],
                    "answer": new_answer,
                    "source": "Ollama",
                    "tokens": len(new_answer.split()),
                    "confidence": new_score,
                    "timestamp": time.time(),
                }
                self.evaluator.update_leaderboard(entry["question"], new_score)
                changed = True
                self.updated_last += 1
            self.checked += 1
        if changed:
            self.memory.save()
# === FILE: backend/features/self_reflect.py ===
class SelfReflection(threading.Thread):
    def __init__(self, interval: int = 300):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.brain = AIBrain()
        self.memory = QAMemory()
        self.evaluator = Evaluator()

    def run(self):
        while not self.stop_event.is_set():
            entry = self.memory.get_random()
            if entry:
                new_answer = self.brain.ask(entry["question"])
                score_old = self.evaluator.score(
                    entry["question"], entry["answer"], entry["source"]
                )
                score_new = self.evaluator.score(
                    entry["question"], new_answer, entry["source"]
                )
                if score_new > score_old:
                    self.memory.add(
                        entry["question"], new_answer, entry["source"], score_new
                    )
                    self.evaluator.update_leaderboard(entry["question"], score_new)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
# === FILE: backend/features/strategies.py ===
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=period, min_periods=period).mean()
    ma_down = down.rolling(window=period, min_periods=period).mean()
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def rsi_strategy(prices: pd.Series) -> str:
    rsi = compute_rsi(prices).iloc[-1]
    if rsi < 30:
        return "buy"
    if rsi > 70:
        return "sell"
    return "hold"


def ema_strategy(prices: pd.Series, short: int = 12, long: int = 26) -> str:
    ema_short = prices.ewm(span=short, adjust=False).mean()
    ema_long = prices.ewm(span=long, adjust=False).mean()
    if ema_short.iloc[-1] > ema_long.iloc[-1] and ema_short.iloc[-2] <= ema_long.iloc[-2]:
        return "buy"
    if ema_short.iloc[-1] < ema_long.iloc[-1] and ema_short.iloc[-2] >= ema_long.iloc[-2]:
        return "sell"
    return "hold"


def macd_strategy(prices: pd.Series) -> str:
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
        return "buy"
    if macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
        return "sell"
    return "hold"
# === FILE: backend/features/telegram_alerts.py ===
def send_telegram_alert(message: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=5)
    except Exception:
        pass
# === FILE: backend/features/trending.py ===
RSS_FEEDS = [
    "https://feeds.bbci.co.uk/news/rss.xml",
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]


class TrendingTopics:
    def __init__(self, cache_file: str = "data/trending.json"):
        self.cache = Path(cache_file)
        self.topics: List[str] = []
        self.last_fetch = 0
        self.load()

    def load(self):
        if self.cache.exists():
            try:
                import json
                data = json.loads(self.cache.read_text())
                self.topics = data.get("topics", [])
                self.last_fetch = data.get("timestamp", 0)
            except Exception:
                pass

    def save(self):
        self.cache.parent.mkdir(exist_ok=True)
        import json
        self.cache.write_text(
            json.dumps({"topics": self.topics, "timestamp": self.last_fetch}, indent=2)
        )

    def fetch(self):
        if time.time() - self.last_fetch < 24 * 3600 and self.topics:
            return self.topics
        topics = []
        for url in RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    topics.append(entry.title)
            except Exception:
                continue
        if topics:
            self.topics = topics[:20]
            self.last_fetch = time.time()
            self.save()
        return self.topics

    def random_topic(self) -> str:
        topics = self.fetch()
        if not topics:
            return random.choice([
                "technology",
                "science",
                "finance",
                "sports",
                "culture",
            ])
        return random.choice(topics)
# === FILE: backend/features/web_search.py ===
def web_search(query):
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        results = soup.find_all("a", class_="result__a", limit=3)
        return "\n".join([r.get_text() for r in results]) or "No results found."
    except Exception as e:
        return f"[Web search error: {e}]"
# === FILE: backend/gui_dashboard.py ===
"""Tkinter-based dashboard for interacting with the trading backend."""

import sys
import os
import subprocess
import json
import time
import threading
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
    import requests  # noqa: F401
except ImportError as exc:
    raise ImportError("❌ Missing 'requests'. Activate your venv and run: pip install requests") from exc

from gui.handlers.ai_handler import (
    ask_ai,
    record_command,
    interpret_command,
    set_current_strategy,
    last_context,
    apply_feedback,
)
from gui.handlers.memory_handler import export_memory, search_memory, top_memories
from gui.handlers.strategy_handler import (
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

        title = tk.Label(self, text="JARVIS CORE DASHBOARD", font=("Helvetica", 20), fg="white", bg="#121212")
        title.pack(pady=10)

        self.memory_frame = self._create_section("Top Memories")
        self.strategy_frame = self._create_section("Strategy Stats")

        self.chat_frame = tk.LabelFrame(self, text="AI Chatbot", bg="#1e1e1e", fg="white", font=("Helvetica", 14), bd=2)
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.chat_log = ScrolledText(self.chat_frame, font=("Helvetica", 12), height=8, bg="#111", fg="white", wrap="word", state="disabled")
        self.chat_log.tag_config("user", foreground="#00FF7F")
        self.chat_log.tag_config("ai", foreground="#1E90FF")
        self.chat_log.tag_config("thinking", foreground="#888888", font=("Helvetica", 12, "italic"))
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

        search_bar = tk.Frame(self.memory_frame, bg="#1e1e1e")
        search_bar.pack(fill="x", padx=5, pady=(0, 5))
        self.search_entry = ttk.Entry(search_bar)
        self.search_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(search_bar, text="Search", command=self.search_memories).pack(side="left")
        ttk.Button(search_bar, text="Export", command=self.export_mem).pack(side="left")

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
        frame = tk.LabelFrame(self, text=title, bg="#1e1e1e", fg="white", font=("Helvetica", 14), bd=2)
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
        self.strategy_frame.listbox.insert(tk.END, f"{self.strategy_var.get()}: W {wins}, L {losses}, PnL ${pnl:.2f}")

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

    def _append_chat(self, speaker: str, text: str, tag: str = "", newline: bool = True) -> None:
        self.chat_log.configure(state="normal")
        tag_name = tag or ("user" if speaker == "You" else "ai")
        ending = "\n" if newline else ""
        self.chat_log.insert(tk.END, f"{speaker}: {text}{ending}", tag_name)
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
        self._append_chat("AI", "Thinking...", tag="thinking")

        def worker() -> None:
            resp = ask_ai(msg)
            self.last_memories = last_context().get("top_memories", []) if last_context() else []
            self.after(0, lambda: self._display_response(resp))

        threading.Thread(target=worker, daemon=True).start()

    def _display_response(self, text: str) -> None:
        self.chat_log.configure(state="normal")
        if self.chat_log.get("end-2l", "end-1c").strip() == "AI: Thinking...":
            start = "end-2l linestart"
            end = "end-1c"
            self.chat_log.delete(start, end)
        self.chat_log.configure(state="disabled")
        self._append_chat("AI", "", newline=False)
        self._typewriter(text)

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
# === FILE: backend/main.py ===
def main():
    brain = AIBrain()
    print("🤖 JARVIS is online. Type 'exit' to quit.")
    online_mode = True  # Turn this False to go fully offline

    # ==== background autotrain setup ====
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    lock_file = base_dir / "autotrain.lock"

    stop_event = threading.Event()

    def start_autotrain() -> tuple[subprocess.Popen, object]:
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text().strip())
                if pid > 0 and Path(f"/proc/{pid}").exists():
                    return None, None
            except Exception:
                pass
            lock_file.unlink(missing_ok=True)

        log_path = log_dir / "autotrain.log"
        log_f = open(log_path, "a")
        proc = subprocess.Popen(
            ["python", "autotrain.py"],
            cwd=str(base_dir),
            stdout=log_f,
            stderr=log_f,
        )
        lock_file.write_text(str(proc.pid))
        return proc, log_f

    def monitor_autotrain(event: threading.Event):
        proc, log_f = start_autotrain()
        while not event.is_set():
            if proc and proc.poll() is not None:
                if log_f:
                    log_f.write(f"AutoTrain exited with {proc.returncode}, restarting...\n")
                    log_f.flush()
                proc, log_f = start_autotrain()
            time.sleep(5)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if log_f:
            log_f.close()
        lock_file.unlink(missing_ok=True)

    monitor_thread = threading.Thread(
        target=monitor_autotrain, args=(stop_event,), daemon=True
    )
    monitor_thread.start()
    reflect_thread = SelfReflection()
    audit_thread = SelfAudit()
    dashboard_thread = TerminalDashboard(audit=audit_thread)
    reflect_thread.start()
    audit_thread.start()
    dashboard_thread.start()

    try:
        while True:
            prompt = input("🧠 You: ").strip()
            if prompt.lower() == "exit":
                print("👋 JARVIS shutting down.")
                break

            if online_mode and prompt.lower().startswith("search:"):
                query = prompt.split("search:", 1)[-1].strip()
                response = web_search(query)
            elif prompt.lower().startswith("trade"):
                _, *symbols = prompt.split()
                run_autotrader(symbols or None)
                response = "Trade executed"
            else:
                response = brain.ask(prompt)

            if response.startswith("[Error"):
                dashboard_thread.fail += 1
            else:
                dashboard_thread.success += 1
            dashboard_thread.log_interaction(prompt, response)
            print(f"🤖 JARVIS: {response}")
    except KeyboardInterrupt:
        print("\n👋 JARVIS shutting down.")
    finally:
        stop_event.set()
        reflect_thread.stop()
        audit_thread.stop()
        dashboard_thread.stop()
        monitor_thread.join()
        reflect_thread.join()
        audit_thread.join()
        dashboard_thread.join()
# === FILE: backend/server.py ===
app = Flask(__name__)


@app.route("/trade")
def trade():
    symbol = request.args.get("symbol", "AAPL")
    run_autotrader([symbol])
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
# === FILE: backend/utils/memory.py ===
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
# === FILE: backend/web_dashboard.py ===
def _display_model(path: str) -> None:
    """Display a GLB model interactively using model-viewer."""
    if not os.path.exists(path):
        st.write("Model not found.")
        return

    try:
        data = open(path, "rb").read()
        b64 = base64.b64encode(data).decode()
        html = f"""
        <script type=\"module\" src=\"https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js\"></script>
        <model-viewer src=\"data:model/gltf-binary;base64,{b64}\" camera-controls auto-rotate style=\"width: 100%; height: 300px;\"></model-viewer>
        """
        st.components.v1.html(html, height=320)
    except Exception:
        st.write("Preview unavailable. Download below.")
        with open(path, "rb") as f:
            st.download_button(
                "Download model", data=f.read(), file_name=os.path.basename(path)
            )


def show_dashboard():
    mem = MemoryManager()
    st.title("JARVIS Web Dashboard")
    for ticker, info in mem.memory.items():
        if ticker in ("stats", "cooldowns"):
            continue
        st.write(
            f"**{ticker}** - P/L: {info['total_profit']:.2f} from {info['trade_count']} trades"
        )
    stats = mem.memory.get("stats", {})
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    total = wins + losses
    if total:
        st.write(f"Win rate: {wins/total*100:.2f}%")

    st.header("Solve Worksheet PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        path = "uploaded.pdf"
        with open(path, "wb") as f:
            f.write(uploaded.getvalue())
        brain = AIBrain()
        results = brain.solve_pdf(path)
        for res in results.values():
            st.markdown(res)

    st.header("Blueprint Simulation")
    blueprint = mem.memory.get("last_blueprint")
    if blueprint:
        st.image(blueprint)
        if st.button("Run Simulation"):
            from features.engineering_expert import EngineeringExpert

            expert = EngineeringExpert()
            result = expert.simulate(blueprint)
            st.markdown(result)
    else:
        st.write("No blueprint available.")

    st.header("Last 3D Model")
    model = mem.memory.get("last_model")
    if model:
        _display_model(model)
    else:
        st.write("No 3D model available.")

    st.header("Interactive Simulation")
    sim_type = st.selectbox(
        "Simulation type",
        ["mechanical", "civil", "electrical", "thermal", "fluid"],
    )
    params = {}
    if sim_type == "mechanical":
        params["length"] = st.slider("Length (m)", 1.0, 10.0, 1.0)
        params["force"] = st.slider("Force (N)", 100.0, 5000.0, 1000.0)
        params["h"] = st.slider("Height (m)", 0.05, 0.5, 0.1)
    elif sim_type == "civil":
        params["span"] = st.slider("Span (m)", 1.0, 20.0, 5.0)
        params["load"] = st.slider("Load (ton)", 1.0, 20.0, 1.0)
    elif sim_type == "electrical":
        params["voltage"] = st.slider("Voltage (V)", 1.0, 20.0, 5.0)
    elif sim_type == "thermal":
        params["length"] = st.slider("Length (m)", 0.5, 5.0, 1.0)
        params["t1"] = st.slider("Temp 1 (C)", 0.0, 500.0, 100.0)
        params["t2"] = st.slider("Temp 2 (C)", 0.0, 500.0, 0.0)
    elif sim_type == "fluid":
        params["width"] = st.slider("Width (m)", 0.5, 5.0, 1.0)
        params["height"] = st.slider("Height (m)", 0.5, 5.0, 1.0)
    if st.button("Run Interactive Simulation"):
        from features.engineering_expert import EngineeringExpert

        expert = EngineeringExpert()
        st.markdown(expert.simulate(sim_type, params))

    st.subheader("Optimization Mode")
    if st.button("Optimize", key="opt_btn"):
        from features.engineering_expert import EngineeringExpert

        expert = EngineeringExpert()
        best = expert.optimize(sim_type, "perf", {k: (0.5, v) if isinstance(v, float) else v for k, v in params.items()})
        st.json(best)

    st.subheader("AI Design Assistant")
    desc = st.text_input("Describe your design goal")
    if st.button("Get Design", key="design") and desc:
        from features.engineering_expert import EngineeringExpert

        expert = EngineeringExpert()
        st.markdown(expert.design_assistant(desc))
        if st.button("Run Full Analysis", key="chain"):
            st.markdown(expert.analysis_chain(desc))

    st.header("Sim Results")
    sim_mem = MemoryManager(path="data/simulation_index.json")
    tag_filter = st.text_input("Filter by tag")
    sims_to_show = sim_mem.memory.get("simulations", [])
    if tag_filter:
        sims_to_show = [s for s in sims_to_show if tag_filter in s.get("tags", [])]
    for sim in reversed(sims_to_show[-5:]):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sim.get("timestamp", 0)))
        with st.expander(f"{sim['prompt']} ({ts})"):
            st.markdown(sim.get("result", ""))
            st.image(sim.get("path", ""))
            model_path = sim.get("model")
            if model_path:
                _display_model(model_path)
            col1, col2 = st.columns(2)
            if col1.button("Rerun", key=f"rerun_{ts}"):
                from features.engineering_expert import EngineeringExpert

                expert = EngineeringExpert()
                st.markdown(expert.simulate(sim["prompt"]))
            if col2.download_button(
                "Export",
                data=open(sim["path"], "rb").read(),
                file_name=os.path.basename(sim["path"]),
            ):
                pass

    st.header("Multi-Sim Comparison")
    all_sims = sim_mem.memory.get("simulations", [])
    sim_map = {s["uuid"]: s for s in all_sims}
    selection = st.multiselect(
        "Select up to 3 simulations",
        list(sim_map.keys()),
        format_func=lambda x: sim_map[x]["prompt"],
        max_selections=3,
    )
    if selection:
        cols = st.columns(len(selection))
        for col, sid in zip(cols, selection):
            sim = sim_map[sid]
            with col:
                st.markdown(f"**{sim['prompt']}**")
                view = st.radio(
                    "View",
                    ["2D Plot", "3D Model"],
                    key=f"view_{sid}",
                )
                if view == "3D Model" and sim.get("model"):
                    _display_model(sim["model"])
                else:
                    st.image(sim.get("path", ""))
                st.write(sim.get("result", ""))
                new_prompt = st.text_input(
                    "Edit prompt",
                    value=sim["prompt"],
                    key=f"edit_{sid}",
                )
                if st.button("Run", key=f"run_{sid}"):
                    from features.engineering_expert import EngineeringExpert

                    expert = EngineeringExpert()
                    st.markdown(expert.simulate(new_prompt))


if __name__ == "__main__":
    show_dashboard()
# === FILE: gui/handlers/ai_handler.py ===
"""AI interaction helpers with contextual prompts and command parsing."""

from __future__ import annotations

import json
import os
import subprocess
import requests
import time
from datetime import date
from typing import List, Dict, Any

from .memory_handler import top_memories, mark_used
from .memory_handler import feedback_memories
from .strategy_handler import load_stats, STRATEGIES
import re


CONFIG_PATH = "config.json"


def _load_model() -> str:
    """Return model name from config.json or default."""
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get("model"):
                    return str(data["model"])
        except Exception:
            pass
    return "jarvisbrain"


OLLAMA_MODEL = _load_model()
ENDPOINT = "http://localhost:11434/api/generate"

# history of user commands
_commands: List[str] = []
_LAST_CONTEXT: Dict[str, object] | None = None

# currently selected trading strategy
CURRENT_STRATEGY = "RSI"


def record_command(cmd: str) -> None:
    """Store user command for context."""
    _commands.append(cmd)
    if len(_commands) > 20:
        del _commands[0]


def set_current_strategy(strategy: str) -> None:
    global CURRENT_STRATEGY
    CURRENT_STRATEGY = strategy


def _strategy_summary() -> str:
    global CURRENT_STRATEGY
    data = load_stats()
    stats = data.get(CURRENT_STRATEGY, {})

    # pick best win-rate strategy if no data for current
    if not stats and isinstance(data, dict):
        best, rate = CURRENT_STRATEGY, -1.0
        for name, info in data.items():
            wins = info.get("wins", 0)
            losses = info.get("losses", 0)
            total = wins + losses
            win_rate = (wins / total) if total else 0.0
            if win_rate > rate:
                best, rate, stats = name, win_rate, info
        CURRENT_STRATEGY = best

    pnl = stats.get("pnl", 0.0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    return f"Current: {CURRENT_STRATEGY} | Wins: {wins} | Losses: {losses} | PnL: ${pnl:.2f}"


def _build_context() -> Dict[str, object]:
    memories = top_memories(3)
    mark_used(memories)
    return {
        "top_memories": memories,
        "strategy": _strategy_summary(),
        "recent_commands": _commands[-3:],
    }


def _context_block(context: Dict[str, object]) -> str:
    mem_lines = []
    for mem in context.get("top_memories", []):
        ts = mem.get("timestamp")
        if ts:
            ts = time.strftime("%Y-%m-%d", time.localtime(float(ts)))
        else:
            ts = "unknown"
        event = mem.get("event", "")
        mem_lines.append(f"- {ts}: {event}")
    mem_text = "\n".join(mem_lines)
    strat_summary = context.get("strategy", "")
    block = f"[MEMORY CONTEXT]\n{mem_text}\n\n[STRATEGY SUMMARY]\n{strat_summary}"
    return block


def _clean_response(text: str) -> str:
    """Remove repeating prefixes like 'Jarvis:' or 'AI:' from model output."""
    lines = []
    for line in text.splitlines():
        line = re.sub(r'^(?:Jarvis|AI|Assistant)\s*:\s*', '', line, flags=re.I)
        if line.strip():
            lines.append(line.strip())
    return ' '.join(lines).strip()


def ask_ai(prompt: str) -> str:
    """Send a prompt to the AI model with contextual information."""
    global _LAST_CONTEXT
    context = _build_context()
    _LAST_CONTEXT = context
    block = _context_block(context)
    full_prompt = f"{block}\n\n{prompt}\nJarvis:"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
    }
    response_text = "Error: AI engine unavailable."
    try:
        resp = requests.post(ENDPOINT, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                response_text = data.get("response", "").strip()
    except Exception:
        pass
    if "Error" in response_text:
        try:
            result = subprocess.run(
                ["ollama", "run", OLLAMA_MODEL],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                response_text = result.stdout.strip()
        except Exception:
            pass

    response_text = _clean_response(response_text)

    _log_interaction(prompt, response_text, context)
    return response_text


def last_context() -> Dict[str, object] | None:
    return _LAST_CONTEXT


def _log_interaction(user_prompt: str, ai_response: str, context: Dict[str, object]) -> None:
    entry = {
        "timestamp": time.time(),
        "prompt": user_prompt,
        "response": ai_response,
        "strategy": context.get("strategy"),
        "memories": context.get("top_memories"),
    }
    day = date.today().isoformat()
    path = f"logs/self_audit/{day}.json"
    os.makedirs("logs/self_audit", exist_ok=True)
    data: List[Dict[str, object]] = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            data = []
    data.append(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def interpret_command(prompt: str) -> Dict[str, str]:
    text = prompt.lower()
    if "switch strategy" in text:
        for s in STRATEGIES:
            if s.lower() in text:
                return {"action": "switch_strategy", "strategy": s}
        return {"action": "switch_strategy"}
    if "pause trading" in text:
        return {"action": "pause_trading"}
    if "show history" in text:
        return {"action": "show_history"}
    return {"action": "none"}


def apply_feedback(memories: List[Dict[str, Any]], positive: bool) -> None:
    feedback_memories(memories, positive)

# === FILE: gui/handlers/memory_handler.py ===
MEMORY_PATH = "data/memory.json"
DECAY_DAYS = 7
DECAY_RATE = 0.1


def load_memory() -> Any:
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_memory(data: Any) -> None:
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _apply_decay(mem: Dict[str, Any]) -> None:
    """Decay importance if memory unused for a period."""
    last = mem.get("last_used", mem.get("timestamp", 0))
    age_days = (time.time() - float(last)) / 86400
    if age_days > DECAY_DAYS:
        decay = (age_days - DECAY_DAYS) * DECAY_RATE
        mem["importance"] = max(mem.get("importance", 1) - decay, 0)


def top_memories(n: int = 5) -> List[Dict[str, Any]]:
    data = load_memory()
    memories: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for mem in data:
            _apply_decay(mem)
        data.sort(key=lambda m: m.get("importance", 1), reverse=True)
        save_memory(data)
        memories = data[:n]
    elif isinstance(data, dict):
        for ticker, info in data.items():
            if ticker in {"stats", "cooldowns"}:
                continue
            memories.append({"timestamp": ticker, "event": f"{info.get('total_profit', 0.0):.2f} P/L"})
        memories = memories[:n]
    return memories


def search_memory(keyword: str = "", start: float | None = None, end: float | None = None) -> List[Dict[str, Any]]:
    data = load_memory()
    if not isinstance(data, list):
        return []
    results: List[Dict[str, Any]] = []
    for mem in data:
        ts = mem.get("timestamp")
        event = mem.get("event", "")
        if keyword and keyword.lower() not in event.lower():
            continue
        if start and ts < start:
            continue
        if end and ts > end:
            continue
        results.append(mem)
    return results


def mark_used(memories: List[Dict[str, Any]]) -> None:
    data = load_memory()
    if not isinstance(data, list):
        return
    changed = False
    for mem in data:
        for m in memories:
            if mem.get("timestamp") == m.get("timestamp"):
                mem["importance"] = mem.get("importance", 1) + 1
                mem["last_used"] = time.time()
                changed = True
    if changed:
        save_memory(data)


def feedback_memories(memories: List[Dict[str, Any]], positive: bool = True) -> None:
    data = load_memory()
    if not isinstance(data, list):
        return
    changed = False
    for mem in data:
        for m in memories:
            if mem.get("timestamp") == m.get("timestamp"):
                delta = 1 if positive else -1
                mem["importance"] = max(mem.get("importance", 1) + delta, 0)
                if not positive:
                    mem["flagged"] = True
                changed = True
    if changed:
        save_memory(data)


def export_memory(path: str = "memory_export.json") -> str:
    data = load_memory()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
# === FILE: gui/handlers/strategy_handler.py ===
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
