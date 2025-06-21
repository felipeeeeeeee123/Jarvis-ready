from pathlib import Path

import streamlit as st

from combined_jarvis import answer_question, load_memory, save_memory


MEMORY_PATH = Path("memory.json")

# Configure basic page settings for a clean layout
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


# --- Prompt input ---
if prompt := st.chat_input("Type your message and press Enter"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = answer_question(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})


