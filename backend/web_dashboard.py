import streamlit as st
import os
import time
import base64
from utils.memory import MemoryManager
from features.ai_brain import AIBrain


def _display_model(path: str) -> None:
    """Display a GLB model interactively using model-viewer."""
    if not os.path.exists(path):
        st.write("Model not found.")
        return

    try:
        data = open(path, "rb").read()
        b64 = base64.b64encode(data).decode()
        html = f"""
        <script type=\"module\" src=\"https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js\"></script>
        <model-viewer src=\"data:model/gltf-binary;base64,{b64}\" camera-controls auto-rotate style=\"width: 100%; height: 300px;\"></model-viewer>
        """
        st.components.v1.html(html, height=320)
    except Exception:
        st.write("Preview unavailable. Download below.")
        with open(path, "rb") as f:
            st.download_button(
                "Download model", data=f.read(), file_name=os.path.basename(path)
            )


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

    st.header("Last 3D Model")
    model = mem.memory.get("last_model")
    if model:
        _display_model(model)
    else:
        st.write("No 3D model available.")

    st.header("Sim Results")
    sim_mem = MemoryManager(path="data/simulation_index.json")
    for sim in reversed(sim_mem.memory.get("simulations", [])[-5:]):
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(sim.get("timestamp", 0)))
        with st.expander(f"{sim['prompt']} ({ts})"):
            st.markdown(sim.get("result", ""))
            st.image(sim.get("path", ""))
            model_path = sim.get("model")
            if model_path:
                _display_model(model_path)
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

    st.header("Multi-Sim Comparison")
    all_sims = sim_mem.memory.get("simulations", [])
    sim_map = {s["uuid"]: s for s in all_sims}
    selection = st.multiselect(
        "Select up to 3 simulations",
        list(sim_map.keys()),
        format_func=lambda x: sim_map[x]["prompt"],
        max_selections=3,
    )
    if selection:
        cols = st.columns(len(selection))
        for col, sid in zip(cols, selection):
            sim = sim_map[sid]
            with col:
                st.markdown(f"**{sim['prompt']}**")
                view = st.radio(
                    "View",
                    ["2D Plot", "3D Model"],
                    key=f"view_{sid}",
                )
                if view == "3D Model" and sim.get("model"):
                    _display_model(sim["model"])
                else:
                    st.image(sim.get("path", ""))
                st.write(sim.get("result", ""))
                new_prompt = st.text_input(
                    "Edit prompt",
                    value=sim["prompt"],
                    key=f"edit_{sid}",
                )
                if st.button("Run", key=f"run_{sid}"):
                    from features.engineering_expert import EngineeringExpert

                    expert = EngineeringExpert()
                    st.markdown(expert.simulate(new_prompt))


if __name__ == "__main__":
    show_dashboard()
