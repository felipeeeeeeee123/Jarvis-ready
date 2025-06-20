import json
import os
import time
from datetime import datetime
from typing import Any, List, Dict

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
