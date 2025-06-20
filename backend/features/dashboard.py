import threading
import time


class TerminalDashboard(threading.Thread):
    """Minimal dashboard stub for Windows compatibility."""

    def __init__(self, refresh: int = 2, audit=None):
        super().__init__(daemon=True)
        self.refresh = refresh
        self.stop_event = threading.Event()
        self.audit = audit
        self.success = 0
        self.fail = 0
        self.interactions: list[tuple[str, str]] = []

    def log_interaction(self, question: str, answer: str) -> None:
        self.interactions.append((question, answer))
        if len(self.interactions) > 5:
            self.interactions.pop(0)

    def run(self) -> None:
        while not self.stop_event.is_set():
            time.sleep(self.refresh)

    def stop(self) -> None:
        self.stop_event.set()
