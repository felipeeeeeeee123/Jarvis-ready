
"""Simple persistent memory for JARVIS components."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

class MemoryManager:
    """Lightweight JSON file based storage."""

    def __init__(self, path: str = "data/memory.json") -> None:
        self.path = path
        self.memory: dict[str, str] = {}
        self.load()

    def load(self) -> None:
        """Load memory from disk."""
        if not os.path.exists(self.path):
            logger.debug("Memory file %s does not exist. Starting fresh.", self.path)
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.memory = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Failed to load memory: %s", exc)
            self.memory = {}

    def save(self) -> None:
        """Persist memory to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=4)
        except OSError as exc:
            logger.error("Failed to save memory: %s", exc)
