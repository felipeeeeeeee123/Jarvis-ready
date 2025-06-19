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
        self.success = 0
        self.fail = 0

    def run(self):
        try:
            curses.wrapper(self.loop)
        except Exception:
            pass

    def loop(self, stdscr):
        stdscr.nodelay(True)
        while not self.stop_event.is_set():
            stdscr.clear()
            stdscr.addstr(0, 0, f"Q&A stored: {len(self.memory.data)}")
            stdscr.addstr(1, 0, f"Success: {self.success}  Fail: {self.fail}")
            stdscr.addstr(3, 0, "Commands: [p]ause/[r]esume [c]lear [q]uit")
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
