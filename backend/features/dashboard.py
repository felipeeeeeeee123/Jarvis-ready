import curses
import threading
import time

from .qa_memory import QAMemory


class TerminalDashboard(threading.Thread):
    def __init__(self, refresh: int = 2):
        super().__init__(daemon=True)
        self.refresh = refresh
        self.stop_event = threading.Event()
        self.paused = False
        self.memory = QAMemory()
        self.start_time = time.time()

    def run(self):
        try:
            curses.wrapper(self.loop)
        except Exception:
            pass

    def loop(self, stdscr):
        stdscr.nodelay(True)
        while not self.stop_event.is_set():
            stdscr.clear()
            self.memory.load()
            total = len(self.memory.data)
            tokens = sum(e.get("tokens", 0) for e in self.memory.data)
            avg_conf = (
                sum(e.get("confidence", 0) for e in self.memory.data) / total
                if total
                else 0
            )
            unique_q = {e["question"] for e in self.memory.data}
            dup_rate = 1 - (len(unique_q) / total) if total else 0
            learning_rate = total / ((time.time() - self.start_time) / 60 + 1e-6)
            active = self.memory.data[-1]["source"] if self.memory.data else "N/A"

            stdscr.addstr(0, 0, f"Q&A stored: {total}")
            stdscr.addstr(1, 0, f"Avg confidence: {avg_conf:.2f}")
            stdscr.addstr(2, 0, f"Learning rate: {learning_rate:.2f}/min")
            stdscr.addstr(3, 0, f"Token usage: {tokens}")
            stdscr.addstr(4, 0, f"Duplicate rate: {dup_rate:.2f}")
            stdscr.addstr(5, 0, f"Pruned: {self.memory.pruned_total}")
            stdscr.addstr(6, 0, f"Active source: {active}")
            stdscr.addstr(8, 0, "Commands: [p]ause/[r]esume [c]lear [q]uit")
            stdscr.refresh()
            ch = stdscr.getch()
            if ch != -1:
                ch = chr(ch)
                if ch == "p":
                    self.paused = True
                elif ch == "r":
                    self.paused = False
                elif ch == "c":
                    self.memory.data = []
                    self.memory.save()
                elif ch == "q":
                    self.stop_event.set()
                    break
            time.sleep(self.refresh)

    def stop(self):
        self.stop_event.set()
