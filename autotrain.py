import csv
import hashlib
import json
import random
import time
from pathlib import Path

from backend.features.ai_brain import AIBrain
from backend.features.evaluator import Evaluator
from backend.features.qa_memory import QAMemory
from backend.features.trending import TrendingTopics
from backend.features.web_search import web_search

PAUSE_FILE = Path("autotrain.pause")

TRAINING_PATH = Path("data/training_data.csv")
BACKUP_PATH = Path("data/memory_backup.json")
SEED_TOPICS = [
    "history",
    "politics",
    "technology",
    "war",
    "famous people",
    "trends",
    "science",
    "finance",
    "sports",
    "culture",
]
QUESTION_TEMPLATES = [
    "Explain the latest developments in {}.",
    "Why is {} important today?",
    "Summarize recent news about {}.",
    "How does {} impact society?",
    "What is the future of {}?",
]


class SyntheticTrainer:
    def __init__(self):
        self.brain = AIBrain()
        self.trending = TrendingTopics()
        self.memory = QAMemory()
        self.evaluator = Evaluator()
        self.seen = set()
        self.buffer = []
        self.load_memory()

    def load_memory(self):
        if TRAINING_PATH.exists():
            with open(TRAINING_PATH, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    h = hashlib.sha1(row["question"].encode()).hexdigest()
                    self.seen.add(h)
        if BACKUP_PATH.exists():
            try:
                with open(BACKUP_PATH) as f:
                    data = json.load(f)
                    self.seen.update(data.get("hashes", []))
            except Exception:
                pass

    def save_progress(self):
        TRAINING_PATH.parent.mkdir(exist_ok=True)
        file_exists = TRAINING_PATH.exists()
        with open(TRAINING_PATH, "a", newline="") as f:
            fieldnames = [
                "question",
                "answer",
                "source",
                "token_count",
                "confidence",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(self.buffer)
        with open(BACKUP_PATH, "w") as f:
            json.dump({"hashes": list(self.seen)}, f, indent=2)
        self.buffer = []
        self.memory.prune()

    def generate_question(self) -> str:
        template = random.choice(QUESTION_TEMPLATES)
        topic = self.trending.random_topic()
        q_prompt = f"Write a short question about: {topic}"
        question = self.brain.ask(q_prompt)
        if len(question.split()) > 20:
            question = template.format(topic)
        return question

    def ask_question(self, question: str) -> tuple[str, str]:
        try:
            context = web_search(question)
            source = "DuckDuckGo"
        except Exception:
            context = ""
            source = "Ollama"
        prompt = f"{question}\n\nContext:\n{context}"
        answer = self.brain.ask(prompt)
        return answer, source

    def run(self):
        counter = len(self.seen)
        while True:
            if PAUSE_FILE.exists():
                time.sleep(1)
                continue
            question = self.generate_question()
            qhash = hashlib.sha1(question.encode()).hexdigest()
            if qhash in self.seen:
                continue
            try:
                answer, source = self.ask_question(question)
            except Exception as exc:
                print(f"Error fetching answer: {exc}")
                time.sleep(2)
                continue
            tokens = len(answer.split())
            if tokens < 10 or answer.startswith("[Error"):
                continue
            confidence = min(1.0, tokens / 100)
            self.buffer.append(
                {
                    "question": question,
                    "answer": answer,
                    "source": source,
                    "token_count": tokens,
                    "confidence": f"{confidence:.2f}",
                }
            )
            score = self.evaluator.score(question, answer, source)
            self.memory.add(question, answer, source, score)
            self.evaluator.update_leaderboard(question, score)
            self.seen.add(qhash)
            counter += 1
            first_line = answer.splitlines()[0]
            print(f"[# {counter}] Q: {question}\n-> A: {first_line}\n\u2705 Learned from {source} ({tokens} tokens)")
            if len(self.buffer) >= 100:
                self.save_progress()
            time.sleep(1)


if __name__ == "__main__":
    trainer = SyntheticTrainer()
    trainer.run()
