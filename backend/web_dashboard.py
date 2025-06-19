import streamlit as st
import os
import time
from utils.memory import MemoryManager
from features.ai_brain import AIBrain


def show_dashboard():
    mem = MemoryManager()
    st.title("JARVIS Web Dashboard")
    for ticker, info in mem.memory.items():
        if ticker in ("stats", "cooldowns"):
            continue
        st.write(
            f"**{ticker}** - P/L: {info['total_profit']:.2f} from {info['trade_count']} trades"
        )
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

    st.header("Sim Results")
    sim_mem = MemoryManager(path="data/simulation_index.json")
    for sim in reversed(sim_mem.memory.get("simulations", [])[-5:]):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sim.get("timestamp", 0)))
        with st.expander(f"{sim['prompt']} ({ts})"):
            st.markdown(sim.get("result", ""))
            st.image(sim.get("path", ""))
            col1, col2 = st.columns(2)
            if col1.button("Rerun", key=f"rerun_{ts}"):
                from features.engineering_expert import EngineeringExpert

                expert = EngineeringExpert()
                st.markdown(expert.simulate(sim["prompt"]))
            if col2.download_button(
                "Export",
                data=open(sim["path"], "rb").read(),
                file_name=os.path.basename(sim["path"]),
            ):
                pass


if __name__ == "__main__":
    show_dashboard()
