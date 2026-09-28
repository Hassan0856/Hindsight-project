"""
MemoryAgent: incident-response agent with persistent memory (Hindsight).

Per turn:
  1. RECALL   -> query Hindsight using the current message plus the previous
                 one, so follow-ups like "that fix didn't work" still find context
  2. RESPOND  -> LLM sees short-term chat history + recalled long-term memories
  3. RETAIN   -> a small extraction step saves only clean facts (incident
                 reports, confirmed outcomes). The agent's own suggestions are
                 NOT stored as facts, so they can't come back as fake "past incidents".
"""

import os
import json
from datetime import date
from dataclasses import dataclass
from dotenv import load_dotenv
from hindsight_client import Hindsight
from groq import Groq

load_dotenv()

# Change HINDSIGHT_BANK_ID in .env to start from a fresh, empty memory bank.
BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "oncall-history")

MAX_HISTORY_MESSAGES = 6   # short-term memory: last 3 exchanges
MAX_MEMORIES = 5           # cap recalled memories fed to the LLM


SYSTEM_PROMPT = """You are an on-call incident response assistant. You have
long-term memory of this company's past incidents (in "Relevant memories")
and you can see the current conversation.

Rules:
1. Follow-ups matter. If the engineer says a fix did not work, do NOT repeat
   that fix. Say what it rules out and propose the next most likely cause.
2. Only call something a "Match" if a memory shares the same service OR the
   same specific symptom/error. State the basis in one line: which past
   incident (date, service), what matched, and confidence (High/Medium/Low).
3. If the service or symptom is too vague to match confidently, say so and ask
   ONE clarifying question (e.g. which service, what error) instead of
   guessing. You may still give one low-confidence hypothesis.
4. If nothing in memory matches, say "No similar incident in memory" and give
   at most 4 short, generic triage steps. Do not import specific tools, flags
   or config names from unrelated memories.
5. Memories marked WORKED are proven fixes; DID NOT WORK means avoid them.
6. Keep it tight: under ~150 words, short bullets, commands only when useful.
   No filler, no long checklists.

Never claim to remember anything that is not in the memories or the chat."""


EXTRACT_PROMPT = """You maintain an incident knowledge base. Read the engineer's
latest message (with the previous report and the assistant's previous
suggestion for context) and output ONE JSON object and nothing else:

{"kind": "incident" | "outcome" | "other", "summary": "..."}

- incident: a new incident report or postmortem details. summary = 1-2 factual
  sentences (service, symptom, error, cause and fix if stated).
- outcome: the engineer says whether a suggested fix worked or failed. summary
  = the incident, the specific fix tried, and "WORKED" or "DID NOT WORK", plus
  any detail the engineer gave.
- other: questions or chit-chat with nothing worth remembering. summary = "".

Use only facts the engineer stated. Never treat the assistant's suggestion as
a fact unless the engineer says they applied it."""


@dataclass
class TurnLog:
    user_input: str
    recalled: list
    response: str
    retained: str = ""   # what was saved to memory this turn ("" = nothing)


class MemoryAgent:
    def __init__(self, bank_id: str = BANK_ID):
        self.bank_id = bank_id
        self.hindsight = Hindsight(
            base_url=os.environ["HINDSIGHT_BASE_URL"],
            api_key=os.environ["HINDSIGHT_API_KEY"],
        )
        self.llm = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.history = []            # short-term chat messages
        self.last_user_message = ""  # used to widen the recall query
        # Hindsight creates the bank on the first retain() with a new bank_id.

    def reset_conversation(self):
        """Clears short-term chat only; long-term Hindsight memory is untouched."""
        self.history = []
        self.last_user_message = ""

    def _extract_memory(self, user_input: str) -> str:
        """Turn the engineer's message into a clean fact worth retaining."""
        prev_suggestion = ""
        if self.history and self.history[-1]["role"] == "assistant":
            prev_suggestion = self.history[-1]["content"][:800]
        context = (
            f"Previous report: {self.last_user_message[:500]}\n"
            f"Assistant's previous suggestion: {prev_suggestion}\n\n"
            f"Engineer's latest message: {user_input}"
        )
        try:
            out = self.llm.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": EXTRACT_PROMPT},
                    {"role": "user", "content": context},
                ],
            ).choices[0].message.content.strip()
            out = out.replace("```json", "").replace("```", "").strip()
            data = json.loads(out)
            if data.get("kind") in ("incident", "outcome") and data.get("summary"):
                return data["summary"].strip()
            return ""
        except Exception:
            # Open models sometimes return malformed JSON: fall back to the raw report
            return user_input

    def respond(self, user_input: str) -> TurnLog:
        # 1. RECALL: widen the query with the previous message for follow-ups
        query = user_input
        if self.last_user_message:
            query = f"{self.last_user_message[:500]}\n{user_input}"
        recall_result = self.hindsight.recall(bank_id=self.bank_id, query=query)
        memories = [m.text for m in recall_result.results] if recall_result.results else []
        memories = memories[:MAX_MEMORIES]
        memory_block = "\n".join(f"- {m}" for m in memories) if memories else "(none)"

        # 2. RESPOND: short-term history + long-term memories
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages += self.history[-MAX_HISTORY_MESSAGES:]
        messages.append({
            "role": "user",
            "content": f"Relevant memories:\n{memory_block}\n\nEngineer: {user_input}",
        })
        response = self.llm.chat.completions.create(
            model=self.model, messages=messages
        ).choices[0].message.content

        # 3. RETAIN: only clean facts, never the agent's own advice
        summary = self._extract_memory(user_input)
        retained = ""
        if summary:
            retained = f"[{date.today().isoformat()}] {summary}"
            self.hindsight.retain(bank_id=self.bank_id, content=retained)

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        self.last_user_message = user_input

        return TurnLog(user_input=user_input, recalled=memories,
                       response=response, retained=retained)

    def close(self):
        self.hindsight.close()