
from features.autotrade import run_autotrader
from utils.memory import MemoryManager

def main():
    memory = MemoryManager()
    memory.load()
    print("JARVIS initialized with memory.")
    run_autotrader()

if __name__ == "__main__":
    main()
