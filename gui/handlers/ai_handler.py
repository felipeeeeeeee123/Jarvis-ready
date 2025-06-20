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
    stats = load_stats().get(CURRENT_STRATEGY, {})
    pnl = stats.get("pnl", 0.0)
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    return f"{CURRENT_STRATEGY}: PnL ${pnl:.2f} (W {wins}/L {losses})"


def _build_context() -> Dict[str, object]:
    memories = top_memories(3)
    mark_used(memories)
    return {
        "top_memories": memories,
        "strategy": _strategy_summary(),
        "recent_commands": _commands[-3:],
    }


def ask_ai(prompt: str) -> str:
    """Send a prompt to the AI model with contextual information."""
    global _LAST_CONTEXT
    context = _build_context()
    _LAST_CONTEXT = context
    background = json.dumps(context, indent=2)
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Context:\n{background}\n\nUser: {prompt}\nAI:",
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
                ["ollama", "run", OLLAMA_MODEL, payload["prompt"]],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.stdout:
                response_text = result.stdout.strip()
        except Exception:
            pass

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

