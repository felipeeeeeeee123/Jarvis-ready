import curses
import threading
import time

from .qa_memory import QAMemory


class TerminalDashboard(threading.Thread):
    def __init__(self, refresh: int = 2, audit=None):
        super().__init__(daemon=True)
        self.refresh = refresh
        self.stop_event = threading.Event()
        self.paused = False
        self.memory = QAMemory()
        self.start_time = time.time()
        self.audit = audit
        self.success = 0
        self.fail = 0
        self.interactions: list[tuple[str, str]] = []
        self.last_interaction = time.time()

    def log_interaction(self, question: str, answer: str) -> None:
        self.interactions.append((question, answer))
        if len(self.interactions) > 5:
            self.interactions.pop(0)
        self.last_interaction = time.time()

    def _color(self, score: float) -> int:
        if score >= 0.75:
            return curses.color_pair(1)
        if score >= 0.5:
            return curses.color_pair(2)
        return curses.color_pair(3)

    def _topic(self, question: str) -> str:
        words = question.lower().split()
        for token in reversed(words):
            token = token.strip(".,?!")
            if token:
                return token
        return ""

    def run(self):
        try:
            curses.wrapper(self.loop)
        except Exception:
            pass

    def loop(self, stdscr):
        curses.curs_set(0)
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        stdscr.nodelay(True)
        while not self.stop_event.is_set():
            stdscr.erase()
            self.memory.load()
            total = len(self.memory.data)
            tokens = sum(e.get("tokens", 0) for e in self.memory.data)
            avg_conf = (
                sum(e.get("confidence", 0) for e in self.memory.data) / total
                if total
                else 0
            )
            now = time.time()
            review = [
                e
                for e in self.memory.data
                if now - e.get("timestamp", 0) > 7 * 86400
                or e.get("confidence", 1) < 0.5
            ]
            unique_q = {e["question"] for e in self.memory.data}
            dup_rate = 1 - (len(unique_q) / total) if total else 0
            learning_rate = total / ((time.time() - self.start_time) / 60 + 1e-6)
            active = self.memory.data[-1]["source"] if self.memory.data else "N/A"

            stdscr.addstr(0, 0, "=== JARVIS Console ===", curses.A_BOLD)

            row = 2
            stdscr.addstr(row, 0, "Learning", curses.A_UNDERLINE)
            row += 1
            stdscr.addstr(row, 2, f"Success: {self.success}", curses.color_pair(1))
            row += 1
            stdscr.addstr(row, 2, f"Failures: {self.fail}", curses.color_pair(3))

            row += 2
            stdscr.addstr(row, 0, "Memory", curses.A_UNDERLINE)
            row += 1
            stdscr.addstr(row, 2, f"Total stored: {total}")
            row += 1
            stdscr.addstr(row, 2, f"Avg confidence: {avg_conf:.2f}", self._color(avg_conf))
            row += 1
            stdscr.addstr(row, 2, f"Learning rate: {learning_rate:.2f}/min")
            row += 1
            stdscr.addstr(row, 2, f"Token usage: {tokens}")
            row += 1
            stdscr.addstr(row, 2, f"Duplicate rate: {dup_rate:.2f}")
            row += 1
            color_pruned = curses.color_pair(3) if self.memory.pruned_total else curses.color_pair(1)
            stdscr.addstr(row, 2, f"Pruned: {self.memory.pruned_total}", color_pruned)
            row += 1
            color_review = curses.color_pair(2) if len(review) else curses.color_pair(1)
            stdscr.addstr(row, 2, f"Needs review: {len(review)}", color_review)
            row += 1
            stdscr.addstr(row, 2, f"Active source: {active}")

            topic_counts = {}
            for e in review:
                t = self._topic(e.get("question", ""))
                if t:
                    topic_counts[t] = topic_counts.get(t, 0) + 1
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            topics_str = ", ".join(t for t, _ in top_topics) if top_topics else "None"
            row += 1
            stdscr.addstr(row, 2, f"Top outdated topics: {topics_str}")

            row += 2
            stdscr.addstr(row, 0, "Audit", curses.A_UNDERLINE)
            status = "N/A"
            if self.audit:
                status = f"{self.audit.checked}/{self.audit.check_total} checked, {self.audit.updated_last} updated"
            row += 1
            stdscr.addstr(row, 2, status)

            row += 2
            stdscr.addstr(row, 0, "Recent", curses.A_UNDERLINE)
            for i, (q, a) in enumerate(reversed(self.interactions[-3:])):
                row += 1
                stdscr.addstr(row, 2, f"Q: {q}")
                row += 1
                stdscr.addstr(row, 4, f"A: {a.splitlines()[0][:50]}")

            row += 1
            if time.time() - self.last_interaction > self.refresh * 2:
                stdscr.addstr(row, 0, "Awaiting questions...", curses.A_DIM)
            else:
                stdscr.addstr(row, 0, " " * 20)

            stdscr.addstr(row + 2, 0, "Commands: [p]ause/[r]esume [c]lear [q]uit")
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
