import threading
import time

from .ai_brain import AIBrain
from .qa_memory import QAMemory
from .evaluator import Evaluator
from .web_search import web_search


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
        for i, entry in enumerate(list(self.memory.data)):
            score = entry.get("confidence")
            if score is None:
                score = self.evaluator.score(entry["question"], entry["answer"], entry["source"])
            age = now - entry.get("timestamp", 0)
            if age < self.review_age and score >= 0.5:
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
        if changed:
            self.memory.save()
