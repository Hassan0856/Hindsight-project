"""
Streamlit UI for the incident response agent. Wraps the same MemoryAgent
used by chat.py and demo.py — no new logic, just a visual layer.

Deliberately keeps recalled memories visible in a sidebar rather than
hiding them inside the chat bubble, since memory being visible is the
whole point of the demo (25% of judging is "is memory central and
visible", not just "does the agent work").

Two things added on top of the base chat UI:
  - a color-coded verdict badge (Strong / Partial / No match) above each
    response, so the classification is scannable at a glance, not just
    readable in the text
  - a try/except around the agent call, so a transient API hiccup during
    a LIVE judge demo shows an error message instead of crashing the app

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
    st.session_state.history = []  # list of {"role", "content", "recalled"?, "retained"?}
    agent.reset_conversation()     # new browser session = fresh short-term chat


def render_verdict_badge(response_text: str):
    """Surfaces the Strong/Partial/No-match classification as a colored
    badge instead of leaving it buried in the response text."""
    low = response_text.lower()
    if "strong match" in low:
        st.success("Strong match", icon="✅")
    elif "partial match" in low:
        st.warning("Partial match — Confidence: Medium", icon="🟡")
    elif "no relevant historical incident" in low or "no relevant match" in low:
        st.info("No relevant historical incident found", icon="ℹ️")
    # else: response didn't state a verdict (e.g. a clarifying question) — no badge


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
    elif last_assistant_turn.get("recalled"):
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
        if turn["role"] == "assistant":
            render_verdict_badge(turn["content"])
        st.markdown(turn["content"])

user_input = st.chat_input("Describe a new incident...")

if user_input:
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    log = None
    with st.chat_message("assistant"):
        try:
            with st.spinner("Checking memory and responding..."):
                log = agent.respond(user_input)
            render_verdict_badge(log.response)
            st.markdown(log.response)
        except Exception as e:
            st.error(
                "Something went wrong talking to the agent (Hindsight or "
                f"Groq may be temporarily unavailable): {e}"
            )

    if log:
        st.session_state.history.append(
            {"role": "assistant", "content": log.response,
             "recalled": log.recalled, "retained": log.retained}
        )
        st.rerun()
    # on failure: don't rerun — the error stays visible, and the user can
    # just retype the message rather than the app silently dying