"""Simplified CLI assistant using local Ollama and a searchable memory."""

from __future__ import annotations

import json
import time
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import re
from difflib import SequenceMatcher


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


COMMON_MISSPELLINGS = {
    "lenght": "length",
    "metres": "meters",
    "metre": "meter",
    "meteres": "meters",
}


def normalize_prompt(text: str) -> str:
    """Return text with basic typo fixes and fuzzy corrections."""
    for wrong, right in COMMON_MISSPELLINGS.items():
        text = text.replace(wrong, right)
    words = text.split()
    vocab = [
        "beam",
        "model",
        "render",
        "length",
        "meters",
        "calculate",
        "torque",
        "force",
        "moment",
        "inertia",
    ]
    fixed: list[str] = []
    for word in words:
        best = word
        best_score = 0.0
        for v in vocab:
            score = SequenceMatcher(None, word.lower(), v).ratio()
            if score > best_score and score > 0.8:
                best = v
                best_score = score
        fixed.append(best)
    return " ".join(fixed)


def extract_beam_length(prompt: str) -> float | None:
    """Return beam length in meters if found using fuzzy patterns."""
    text = normalize_prompt(prompt.lower())
    match = re.search(r"beam.*?(\d+(?:\.\d+)?)\s*(?:m|meters?)", text)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:m|meters?).*?beam", text)
    if match:
        try:
            return float(match.group(1))
        except Exception:
            pass
    # look for number adjacent to a fuzzy 'beam'
    parts = text.split()
    for i, word in enumerate(parts):
        if SequenceMatcher(None, word, "beam").ratio() > 0.8:
            for idx in (i - 1, i + 1):
                if 0 <= idx < len(parts):
                    m = re.match(r"(\d+(?:\.\d+)?)(?:m)?", parts[idx])
                    if m:
                        try:
                            return float(m.group(1))
                        except Exception:
                            pass
    return None


def answer_question(question: str) -> str:
    """Return an answer using web search and Ollama with fallbacks."""
    question = normalize_prompt(question)
    lower = question.lower()

    # route engineering or 3d modeling prompts to the expert
    if any(
        kw in lower
        for kw in [
            "moment of inertia",
            "beam",
            "load",
            "stress",
            "strain",
            "symbolic",
            "blueprint",
            "calculate",
            "render 3d",
            "torque",
            "force",
            "moment",
        ]
    ):
        if "render" in lower and "3d" in lower:
            try:
                length = extract_beam_length(question)
                if length is None:
                    raise ValueError("length not found")
                model_path = engineering_expert.generate_beam_model(length)
                return f"3D beam model generated: {model_path}"
            except Exception:
                return (
                    "Could not parse beam length. Please say something like 'render 3D model of beam 10m long'."
                )
        else:
            return engineering_expert.solve(question)

    # handle direct date requests instead of the old fixed answer
    date_triggers = (
        "current date",
        "today's date",
        "date today",
        "what is the date",
        "what's the date",
        "what day is it",
    )
    if any(t in lower for t in date_triggers):
        response = datetime.now().strftime("%B %d, %Y")
        add_memory(question, response)
        return response

    cached = search_memory(question)
    if cached:
        return cached

    context = web_search(question)
    if context and not context.startswith("[Web search error") and "No results" not in context:
        prompt = f"{question}\n\n{context}"
        response = ollama_answer(prompt)
    else:
        response = ollama_answer(question)

    if (
        not response
        or response.startswith("[Ollama error")
        or (
            context.startswith("[Web search error") or "No results" in context
        )
    ):
        response = "I'm not sure, but I can look into that further if you'd like."

    add_memory(question, response)
    return response


# Import the engineering expert and expose a default instance for other modules
from backend.features.engineering_expert import EngineeringExpert

# Exposed instance used by GUI and other components
engineering_expert = EngineeringExpert()


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

