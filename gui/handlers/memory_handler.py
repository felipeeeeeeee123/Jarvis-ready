import json
import os
from typing import Any, List, Dict

MEMORY_PATH = "data/memory.json"


def load_memory() -> Any:
    if os.path.exists(MEMORY_PATH):
        try:
            with open(MEMORY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def top_memories(n: int = 5) -> List[Dict[str, Any]]:
    data = load_memory()
    memories: List[Dict[str, Any]] = []
    if isinstance(data, list):
        memories = sorted(data, key=lambda m: m.get("importance", 1), reverse=True)[:n]
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


def export_memory(path: str = "memory_export.json") -> str:
    data = load_memory()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
