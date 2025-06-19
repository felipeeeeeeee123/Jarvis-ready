from features.ai_brain import AIBrain
from features.web_search import web_search
from features.autotrade import run_autotrader

import subprocess
import threading
import time
from pathlib import Path

def main():
    brain = AIBrain()
    print("🤖 JARVIS is online. Type 'exit' to quit.")
    online_mode = True  # Turn this False to go fully offline

    # ==== background autotrain setup ====
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    lock_file = base_dir / "autotrain.lock"

    stop_event = threading.Event()

    def start_autotrain() -> tuple[subprocess.Popen, object]:
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text().strip())
                if pid > 0 and Path(f"/proc/{pid}").exists():
                    return None, None
            except Exception:
                pass
            lock_file.unlink(missing_ok=True)

        log_path = log_dir / "autotrain.log"
        log_f = open(log_path, "a")
        proc = subprocess.Popen(
            ["python", "autotrain.py"],
            cwd=str(base_dir),
            stdout=log_f,
            stderr=log_f,
        )
        lock_file.write_text(str(proc.pid))
        return proc, log_f

    def monitor_autotrain(event: threading.Event):
        proc, log_f = start_autotrain()
        while not event.is_set():
            if proc and proc.poll() is not None:
                if log_f:
                    log_f.write(f"AutoTrain exited with {proc.returncode}, restarting...\n")
                    log_f.flush()
                proc, log_f = start_autotrain()
            time.sleep(5)
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if log_f:
            log_f.close()
        lock_file.unlink(missing_ok=True)

    monitor_thread = threading.Thread(
        target=monitor_autotrain, args=(stop_event,), daemon=True
    )
    monitor_thread.start()

    try:
        while True:
            prompt = input("🧠 You: ").strip()
            if prompt.lower() == "exit":
                print("👋 JARVIS shutting down.")
                break

            if online_mode and prompt.lower().startswith("search:"):
                query = prompt.split("search:", 1)[-1].strip()
                response = web_search(query)
            elif prompt.lower().startswith("trade"):
                _, *symbols = prompt.split()
                run_autotrader(symbols or None)
                response = "Trade executed"
            else:
                response = brain.ask(prompt)

            print(f"🤖 JARVIS: {response}")
    except KeyboardInterrupt:
        print("\n👋 JARVIS shutting down.")
    finally:
        stop_event.set()
        monitor_thread.join()
