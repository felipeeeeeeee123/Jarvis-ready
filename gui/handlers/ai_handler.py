import subprocess
import requests
from typing import Optional


MODEL = "mistral"
ENDPOINT = "http://localhost:11434/api/generate"


def ask_ai(prompt: str) -> str:
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(ENDPOINT, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                return data.get("response", "").strip()
    except Exception:
        pass
    try:
        result = subprocess.run([
            "ollama",
            "run",
            MODEL,
            prompt,
        ], capture_output=True, text=True, timeout=30)
        if result.stdout:
            return result.stdout.strip()
    except Exception:
        pass
    return "Error: AI engine unavailable."
