"""Simplified CLI assistant using local Ollama and a searchable memory."""

from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from pathlib import Path

import requests
from bs4 import BeautifulSoup


MEMORY_PATH = Path("memory.json")


def load_memory() -> list[dict]:
    if MEMORY_PATH.exists():
        try:
            return json.loads(MEMORY_PATH.read_text())
        except Exception:
            return []
    return []


def save_memory(records: list[dict]) -> None:
    MEMORY_PATH.write_text(json.dumps(records, indent=2))


def add_memory(prompt: str, response: str) -> None:
    records = load_memory()
    records.append({
        "prompt": prompt,
        "response": response,
        "timestamp": time.time(),
    })
    save_memory(records)


def search_memory(prompt: str, threshold: float = 0.8) -> str | None:
    best = None
    best_score = threshold
    for entry in load_memory():
        score = SequenceMatcher(None, prompt.lower(), entry["prompt"].lower()).ratio()
        if score > best_score:
            best = entry
            best_score = score
    if best:
        return best["response"]
    return None


def web_search(query: str) -> str:
    url = f"https://duckduckgo.com/html/?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        links = soup.select("a.result__a")
        results = [link.get_text(" ", strip=True) for link in links[:3]]
        return "\n".join(results) if results else "No results found."
    except Exception as exc:
        return f"[Web search error: {exc}]"


def ollama_answer(prompt: str, model: str = "mistral") -> str:
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=30,
        )
        return res.json().get("response", "").strip()
    except Exception as exc:
        return f"[Ollama error: {exc}]"


def answer_question(question: str) -> str:
    cached = search_memory(question)
    if cached:
        return cached

    context = web_search(question)
    if context and not context.startswith("[Web search error") and "No results" not in context:
        prompt = f"{question}\n\n{context}"
        response = ollama_answer(prompt)
    else:
        response = ollama_answer(question)

    add_memory(question, response)
    return response


def main() -> None:
    print("\U0001F9E0 JARVIS AI System Activated")
    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in {"exit", "quit"}:
                break
            reply = answer_question(user_input)
            print(f"\U0001F9E0 JARVIS: {reply}")
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"\u26A0\uFE0F Error: {exc}")


if __name__ == "__main__":
    main()

