from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import sys
import os

# Ensure parent directory is in sys.path to import combined_jarvis.py
sys.path.insert(0, os.path.abspath(Path(__file__).parent.parent))

from combined_jarvis import (
    answer_question,
    add_memory,
    load_memory,
    save_memory,
    EngineeringExpert,
)

MEMORY_PATH = Path("memory.json")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="JARVIS Chatbot", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stChatMessage {max-width: 700px; margin-left: auto; margin-right: auto;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🤖 JARVIS")
    if st.button("New Chat", use_container_width=True):
        save_memory([])
        st.session_state.pop("messages", None)
        st.experimental_rerun()
    if st.button("Exit", use_container_width=True):
        st.write("Chat closed. You can now close this tab.")
        st.stop()
    uploaded = st.file_uploader("Upload worksheet (.pdf or .txt)", type=["pdf", "txt"])
    if uploaded:
        file_path = UPLOAD_DIR / uploaded.name
        file_path.write_bytes(uploaded.getvalue())
        expert = st.session_state.get("expert", EngineeringExpert())
        st.session_state.expert = expert
        if file_path.suffix.lower() == ".pdf":
            with st.spinner("Solving worksheet..."):
                results = expert.solve_pdf_worksheet(str(file_path))
            for q, sol in results.items():
                st.markdown(f"**{q}**")
                st.markdown(sol)
        else:
            content = file_path.read_text()
            with st.spinner("Solving..."):
                answer = expert.answer(content)
            st.markdown(answer)

# --- Load previous messages ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    for record in load_memory():
        st.session_state.messages.append({"role": "user", "content": record.get("prompt", "")})
        st.session_state.messages.append({"role": "assistant", "content": record.get("response", "")})

# --- Display chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

expert = st.session_state.get("expert", EngineeringExpert())
st.session_state.expert = expert

# --- Prompt input ---
if prompt := st.chat_input("Type your message and press Enter"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if EngineeringExpert.is_engineering_question(prompt):
                response = expert.answer(prompt)
            else:
                response = answer_question(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    add_memory(prompt, response)
