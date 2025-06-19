"""Core AI interactions for JARVIS."""

from __future__ import annotations

import logging
import os

import requests
from dotenv import load_dotenv

from utils.memory import MemoryManager

logger = logging.getLogger(__name__)

openai = None

class AIBrain:
    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        """Initialize the AI brain."""

        load_dotenv()
        self.model = model
        self.memory = MemoryManager()
        self.api_key = os.getenv("OPENAI_API_KEY")
        self._init_openai()

    def _init_openai(self) -> None:
        """Attempt to configure the OpenAI client."""

        if not self.api_key:
            logger.info("OPENAI_API_KEY not set. Falling back to local model.")
            return
        try:
            import openai as openai_lib

            openai_lib.api_key = self.api_key
            global openai
            openai = openai_lib
            logger.debug("OpenAI client initialized")
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.error("Failed to initialise OpenAI: %s", exc)

    def ask(self, prompt: str) -> str:
        """Generate a response from either OpenAI or the local model."""

        self.memory.memory["last_prompt"] = prompt
        try:
            if openai:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                choice = response.choices[0]
                answer = choice.message["content"].strip() if hasattr(choice, "message") else choice.text.strip()  # type: ignore[attr-defined]
            else:
                logger.debug("Querying local model")
                resp = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": "mistral", "prompt": prompt, "stream": False},
                    timeout=30,
                )
                answer = resp.json().get("response", "[No response from local model]")
        except Exception as exc:  # pragma: no cover - network errors
            logger.error("AI generation failed: %s", exc)
            answer = f"[Error generating response: {exc}]"

        self.memory.memory["last_answer"] = answer
        self.memory.save()
        return answer

