import csv
import math
from pathlib import Path


SOURCE_WEIGHT = {
    "DuckDuckGo": 1.0,
    "Bing": 0.8,
    "Ollama": 0.5,
}


class Evaluator:
    def __init__(self, leaderboard: str = "data/leaderboard.csv"):
        self.board_path = Path(leaderboard)
        if not self.board_path.exists():
            self.board_path.write_text("question,score\n")

    def score(self, question: str, answer: str, source: str) -> float:
        tokens = len(answer.split())
        token_score = min(tokens / 100, 1.0)
        keywords = set(question.lower().split())
        overlap = sum(1 for w in answer.lower().split() if w in keywords)
        keyword_score = overlap / (len(keywords) or 1)
        source_score = SOURCE_WEIGHT.get(source, 0.5)
        return round((token_score * 0.5) + (keyword_score * 0.3) + (source_score * 0.2), 4)

    def update_leaderboard(self, question: str, score: float):
        entries = []
        if self.board_path.exists():
            with open(self.board_path, newline="") as f:
                reader = csv.DictReader(f)
                entries = list(reader)
        entries.append({"question": question, "score": score})
        entries.sort(key=lambda x: float(x["score"]), reverse=True)
        with open(self.board_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "score"])
            writer.writeheader()
            writer.writerows(entries[:100])
