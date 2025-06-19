from keep_alive import keep_alive
from backend.features.autotrade import run_autotrader


def main():
    keep_alive()
    run_autotrader()


if __name__ == "__main__":
    main()
