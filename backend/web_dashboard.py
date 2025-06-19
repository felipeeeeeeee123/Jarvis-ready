import streamlit as st
from utils.memory import MemoryManager


def show_dashboard():
    mem = MemoryManager()
    st.title("JARVIS Web Dashboard")
    for ticker, info in mem.memory.items():
        if ticker in ("stats", "cooldowns"):
            continue
        st.write(f"**{ticker}** - P/L: {info['total_profit']:.2f} from {info['trade_count']} trades")
    stats = mem.memory.get("stats", {})
    wins = stats.get("wins", 0)
    losses = stats.get("losses", 0)
    total = wins + losses
    if total:
        st.write(f"Win rate: {wins/total*100:.2f}%")


if __name__ == "__main__":
    show_dashboard()
