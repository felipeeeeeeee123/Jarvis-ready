from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from combined_jarvis import (
    answer_question,
    add_memory,
    load_memory,
    save_memory,
    engineering_expert,
)

MEMORY_PATH = Path("memory.json")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# keywords that trigger the engineering expert
ENGINEERING_KEYWORDS = (
    "solve",
    "calculate",
    "blueprint",
    "physics",
    "mechanics",
    "force",
    "equation",
    "engineer",
)

st.set_page_config(page_title="JARVIS Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stChatMessage {max-width: 700px; margin-left: auto; margin-right: auto;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    for rec in load_memory():
        st.session_state.messages.append({"role": "user", "content": rec.get("prompt", "")})
        st.session_state.messages.append({"role": "assistant", "content": rec.get("response", "")})

# Sidebar actions
with st.sidebar:
    st.markdown("## 🤖 JARVIS")
    if st.button("New Chat", use_container_width=True):
        save_memory([])
        st.session_state.clear()
        st.rerun()
    if st.button("Exit", use_container_width=True):
        st.write("Chat closed. You can now close this tab.")
        st.stop()
    uploaded = st.file_uploader("Upload worksheet (.pdf or .txt)", type=["pdf", "txt"])
    if uploaded is not None:
        file_path = UPLOAD_DIR / uploaded.name
        file_path.write_bytes(uploaded.getvalue())
        expert = st.session_state.get("expert", engineering_expert)
        st.session_state.expert = expert
        with st.spinner("Processing worksheet..."):
            if file_path.suffix.lower() == ".pdf":
                result = expert.process_pdf(str(file_path))
            else:
                result = expert.process_txt(str(file_path))
        st.session_state.messages.append({"role": "user", "content": f"[Uploaded {uploaded.name}]"})
        st.session_state.messages.append({"role": "assistant", "content": result})
        add_memory(f"worksheet:{uploaded.name}", result)
        st.rerun()

# Display chat history
for msg in st.session_state.get("messages", []):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

expert = st.session_state.get("expert", engineering_expert)
st.session_state.expert = expert

# Chat input
if prompt := st.chat_input("Type your message and press Enter"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            lower = prompt.lower()
            if any(k in lower for k in ENGINEERING_KEYWORDS):
                response = expert.answer(prompt)
            else:
                response = answer_question(prompt)

        if response.strip().endswith(".glb"):
            model_path = response.strip().split(":")[-1].strip()
            st.markdown("### 3D Model Viewer:")
            st.components.v1.html(
                f'''
        <model-viewer src="/{model_path}" alt="3D model" auto-rotate camera-controls style="width:100%; height:500px;"></model-viewer>
        <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
        ''',
                height=500,
            )
        else:
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    add_memory(prompt, response)
