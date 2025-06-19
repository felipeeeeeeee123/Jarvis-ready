import threading
import time

from .ai_brain import AIBrain
from .qa_memory import QAMemory
from .evaluator import Evaluator


class SelfReflection(threading.Thread):
    def __init__(self, interval: int = 300):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.brain = AIBrain()
        self.memory = QAMemory()
        self.evaluator = Evaluator()

    def run(self):
        while not self.stop_event.is_set():
            entry = self.memory.get_random()
            if entry:
                new_answer = self.brain.ask(entry["question"])
                score_old = self.evaluator.score(
                    entry["question"], entry["answer"], entry["source"]
                )
                score_new = self.evaluator.score(
                    entry["question"], new_answer, entry["source"]
                )
                if score_new > score_old:
                    self.memory.add(
                        entry["question"], new_answer, entry["source"], score_new
                    )
                    self.evaluator.update_leaderboard(entry["question"], score_new)
            time.sleep(self.interval)

    def stop(self):
        self.stop_event.set()
