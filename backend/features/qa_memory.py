import json
import time
from pathlib import Path
from difflib import SequenceMatcher


class QAMemory:
    def __init__(self, path: str = "data/qa_memory.json"):
        self.path = Path(path)
        self.data = []
        self.pruned_total = 0
        self.load()

    def load(self):
        self.pruned_total = 0
        if self.path.exists():
            try:
                with open(self.path) as f:
                    self.data = json.load(f)
            except Exception:
                self.data = []
        self.prune()

    def save(self):
        self.path.parent.mkdir(exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def _is_duplicate(self, prompt: str) -> bool:
        for entry in self.data:
            if entry.get("question") == prompt:
                return True
        return False

    def add(
        self,
        question: str,
        answer: str,
        source: str,
        confidence: float,
        tags: list[str] | None = None,
    ) -> None:
        tokens = len(answer.split())
        if tokens < 10:
            return
        if confidence < 0.5 and self._is_duplicate(question):
            return
        entry = {
            "question": question,
            "answer": answer,
            "source": source,
            "tokens": tokens,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        if tags:
            entry["tags"] = tags
        self._replace_outdated(entry)
        if not self._is_duplicate(question):
            self.data.append(entry)
        self.save()
        self.prune()

    def _replace_outdated(self, new_entry: dict):
        year_tokens = [t for t in new_entry["answer"].split() if t.isdigit() and len(t) == 4]
        if not year_tokens:
            return
        latest_year = max(map(int, year_tokens))
        for idx, entry in enumerate(list(self.data)):
            old_years = [t for t in entry["answer"].split() if t.isdigit() and len(t) == 4]
            if old_years and max(map(int, old_years)) < latest_year:
                if SequenceMatcher(None, entry["question"], new_entry["question"]).ratio() > 0.6:
                    self.data[idx] = new_entry

    def prune(self):
        seen = {}
        for entry in sorted(self.data, key=lambda e: e.get("timestamp", 0)):
            if entry.get("tokens", 0) < 10:
                self.pruned_total += 1
                continue
            key = entry["question"]
            if key in seen:
                if entry.get("timestamp", 0) > seen[key].get("timestamp", 0):
                    seen[key] = entry
                    self.pruned_total += 1
            else:
                seen[key] = entry
        self.data = list(seen.values())
        self.save()

    def get_random(self):
        if not self.data:
            return None
        import random
        return random.choice(self.data)
