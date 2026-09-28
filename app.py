"""
Streamlit UI for the incident response agent. Wraps the same MemoryAgent
used by chat.py and demo.py — no new logic, just a visual layer.

Deliberately keeps recalled memories visible in a sidebar rather than
hiding them inside the chat bubble, since memory being visible is the
whole point of the demo (25% of judging is "is memory central and
visible", not just "does the agent work").

Usage:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
from agent import MemoryAgent, BANK_ID

load_dotenv()


st.set_page_config(page_title="Incident Response Agent", page_icon="🛠️")


@st.cache_resource
def get_agent():
    return MemoryAgent(bank_id=BANK_ID)


agent = get_agent()

st.title("🛠️ On-Call Incident Response Agent")
st.caption("Persistent memory across incidents, via Hindsight")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role", "content", "recalled"?}
    agent.reset_conversation()     # new browser session = fresh short-term chat

# --- Sidebar: what the agent recalled for the most recent query ---
with st.sidebar:
    st.header("Memory recall")
    st.caption("What Hindsight surfaced for the latest incident")

    last_assistant_turn = next(
        (t for t in reversed(st.session_state.history) if t["role"] == "assistant"),
        None,
    )
    if last_assistant_turn is None:
        st.caption("Describe an incident to see recall in action.")
    elif last_assistant_turn["recalled"]:
        for m in last_assistant_turn["recalled"]:
            st.markdown(f"**Recalled:**\n\n{m}")
            st.divider()
    else:
        st.caption("No matching past incident found for this one.")

    if last_assistant_turn and last_assistant_turn.get("retained"):
        st.header("Just learned")
        st.caption("Saved to memory this turn")
        st.markdown(last_assistant_turn["retained"])

    st.divider()
    if st.button("Reset conversation"):
        st.session_state.history = []
        agent.reset_conversation()
        st.rerun()

# --- Main chat area ---
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_input = st.chat_input("Describe a new incident...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking memory and responding..."):
            log = agent.respond(user_input)
        st.markdown(log.response)

    st.session_state.history.append(
        {"role": "assistant", "content": log.response,
         "recalled": log.recalled, "retained": log.retained}
    )
    st.rerun()