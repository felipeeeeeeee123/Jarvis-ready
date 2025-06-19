import streamlit as st
from utils.memory import MemoryManager
from features.ai_brain import AIBrain


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

    st.header("Solve Worksheet PDF")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded:
        path = "uploaded.pdf"
        with open(path, "wb") as f:
            f.write(uploaded.getvalue())
        brain = AIBrain()
        results = brain.solve_pdf(path)
        for res in results.values():
            st.markdown(res)

    st.header("Blueprint Simulation")
    blueprint = mem.memory.get("last_blueprint")
    if blueprint:
        st.image(blueprint)
        if st.button("Run Simulation"):
            from features.engineering_expert import EngineeringExpert

            expert = EngineeringExpert()
            result = expert.simulate(blueprint)
            st.markdown(result)
    else:
        st.write("No blueprint available.")


if __name__ == "__main__":
    show_dashboard()
