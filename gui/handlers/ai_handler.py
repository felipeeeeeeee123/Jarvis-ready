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

