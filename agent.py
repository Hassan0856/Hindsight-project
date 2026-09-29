"""
MemoryAgent: incident-response agent with persistent memory (Hindsight).

Per turn:
  1. RECALL   -> query Hindsight using the current message plus the previous
                 one, so follow-ups like "that fix didn't work" still find context
  2. RESPOND  -> LLM sees short-term chat history + recalled long-term memories
  3. RETAIN   -> a small extraction step saves only clean facts (incident
                 reports, confirmed outcomes). The agent's own suggestions are
                 NOT stored as facts, so they can't come back as fake "past incidents".

IMPORTANT: the Hindsight/Groq clients are created FRESH on every respond()
call rather than once in __init__. In Streamlit, a persisted agent object
(cached with @st.cache_resource) can get re-invoked from a different thread
on a later rerun than the one that created it, and a long-lived async HTTP
client bound to the original thread's event loop then fails with
"Event loop is closed". Only plain data (chat history, bank_id) survives
across calls on this class; network clients never do.
"""

import asyncio
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
MAX_MEMORIES = 8           # cap memories passed to the LLM


SYSTEM_PROMPT = """You are an on-call incident response assistant. You have
long-term memory of this company's past incidents (in "Relevant memories")
and you can see the current conversation.

Rules:
1. Follow-ups matter. If the engineer says a fix did not work, do NOT repeat
   that fix. Say what it rules out and propose the next most likely cause.
2. A separate classifier supplies a "Historical-match verdict" and evidence
   in the latest user message. Treat that verdict as authoritative: do not
   upgrade or replace it. Use only the matching response branch below.
3. For each recalled memory, judge it on ALL of these together, not any one
   alone: service identity, symptoms, error patterns, the type of
   infrastructure/resource problem (e.g. DB/connection saturation, memory,
   cache, certs), root cause category, and whether the past resolution
   worked or failed. A different service name never disqualifies a memory
   by itself, and a single shared keyword never qualifies one by itself —
   weigh the overall pattern.
4. Follow the supplied classification exactly:
   - Strong match: the failure pattern is essentially the same (root cause
     category + symptom class align closely), even if the service differs.
     State which past incident (date, service), which dimensions aligned,
     and the fix to try, adapted to the current service's naming.
   - Partial match: shares the underlying pattern (e.g. same resource-
     saturation category) but differs in specifics — different service,
     different ratios/wording, etc. Explain the shared pattern in one line,
     give the past fix as a hypothesis to try, and end that line with
     "Confidence: Medium".
   - No relevant historical incident found: say exactly
     "No relevant historical incident found." Do not add a match, advice,
     or details from unrelated memories.
   If the report itself is too vague to judge (no service, no symptom),
   ask ONE clarifying question instead of guessing.
5. Memories marked WORKED are proven fixes; DID NOT WORK means avoid them.
6. Keep it tight: under ~150 words, short bullets, commands only when useful.
   No filler, no long checklists.

Never claim to remember anything that is not in the memories or the chat."""


CLASSIFY_PROMPT = """You classify incident-history relevance. Compare the current
incident with each recalled memory using concrete dimensions: service,
symptom class, error behavior, affected resource/infrastructure, root cause,
and a recorded resolution or outcome. Ignore generic operational words such
as deployment, configuration, service, restart, and failure unless specific
technical evidence connects the incidents. A service's presence in a memory
is not itself evidence of a shared failure pattern.

Return only one JSON object: {\"verdict\": \"strong\" | \"partial\" | \"none\", \"evidence\": \"brief explanation\"}.

Use strong only when the service is the same AND at least two independent,
meaningful failure dimensions align (for example, same symptom class plus same
resource/root-cause pattern or specific error behavior). Use partial for a
different service only when the underlying resource/root-cause pattern and
symptoms meaningfully align; state this pattern as a hypothesis. A different
service with a merely generic similarity is none. Use none when the incidents
describe different problems, share only generic words, or the evidence is
insufficient. Confirmed WORKED/DID NOT WORK outcomes are relevant evidence for
which resolution to recommend, but do not by themselves establish a match.
When uncertain, choose none. Never invent relationships or details."""


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
        self.model = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        self.history = []            # short-term chat messages
        self.last_user_message = ""  # used to widen the recall query
        # No client objects stored here on purpose — see module docstring.
        # Hindsight creates the bank on the first retain() with a new bank_id.

    def _new_hindsight_client(self) -> Hindsight:
        return Hindsight(
            base_url=os.environ["HINDSIGHT_BASE_URL"],
            api_key=os.environ["HINDSIGHT_API_KEY"],
        )

    def _new_llm_client(self) -> Groq:
        return Groq(api_key=os.environ["GROQ_API_KEY"])

    def reset_conversation(self):
        """Clears short-term chat only; long-term Hindsight memory is untouched."""
        self.history = []
        self.last_user_message = ""

    def _extract_memory(self, llm: Groq, user_input: str) -> str:
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
            out = llm.chat.completions.create(
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

    def _classify_memories(self, llm: Groq, user_input: str,
                           memories: list[str]) -> tuple[str, str]:
        """Choose a conservative verdict before asking the model to write advice."""
        if not memories:
            return "none", "No recalled historical incidents."
        raw = llm.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": (
                    f"Current incident:\n{user_input}\n\nRecalled memories:\n" +
                    "\n".join(f"- {memory}" for memory in memories)
                )},
            ],
        ).choices[0].message.content.strip()
        try:
            raw = raw.replace("```json", "").replace("```", "").strip()
            result = json.loads(raw)
            verdict = result.get("verdict")
            if verdict not in {"strong", "partial", "none"}:
                return "none", "The recalled memories did not yield a valid match classification."
            return verdict, str(result.get("evidence", "")).strip()
        except Exception:
            # Fail closed: malformed classification must not become a strong match.
            return "none", "The recalled memories could not be reliably classified."

    def respond(self, user_input: str) -> TurnLog:
        """Run one turn in a fresh event loop, isolated from Streamlit reruns."""
        return asyncio.run(self._respond_async(user_input))

    async def _respond_async(self, user_input: str) -> TurnLog:
        hindsight = self._new_hindsight_client()
        llm = self._new_llm_client()
        try:
            # 1. RECALL: widen the query with the previous message for follow-ups
            query = user_input
            if self.last_user_message:
                query = f"{self.last_user_message[:500]}\n{user_input}"
            # Use the async SDK methods on the fresh loop owned by this turn.
            recall_result = await hindsight.arecall(bank_id=self.bank_id, query=query)
            memories = [m.text for m in recall_result.results] if recall_result.results else []
            memories = memories[:MAX_MEMORIES]
            memory_block = "\n".join(f"- {m}" for m in memories) if memories else "(none)"
            verdict, evidence = self._classify_memories(llm, query, memories)

            # A none verdict has one exact, deterministic response. The model
            # cannot turn unrelated retrieval results into a forced connection.
            if verdict == "none":
                response = "No relevant historical incident found."
            else:
                # 2. RESPOND: short-term history + classified long-term memories
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages += self.history[-MAX_HISTORY_MESSAGES:]
                messages.append({
                    "role": "user",
                    "content": (
                        f"Historical-match verdict: {verdict}\n"
                        f"Classification evidence: {evidence}\n\n"
                        f"Relevant memories:\n{memory_block}\n\nEngineer: {user_input}"
                    ),
                })
                response = llm.chat.completions.create(
                    model=self.model, messages=messages
                ).choices[0].message.content
                if verdict == "strong" and not response.lstrip().lower().startswith("strong match"):
                    response = f"Strong match — {response}"
                elif verdict == "partial":
                    if not response.lstrip().lower().startswith("partial match"):
                        response = f"Partial match — {response}"
                    if not response.rstrip().lower().endswith("confidence: medium"):
                        response = f"{response.rstrip()}\nConfidence: Medium"

            # 3. RETAIN: only clean facts, never the agent's own advice
            summary = self._extract_memory(llm, user_input)
            retained = ""
            if summary:
                retained = f"[{date.today().isoformat()}] {summary}"
                await hindsight.aretain(bank_id=self.bank_id, content=retained)
        finally:
            await hindsight.aclose()
            llm.close()

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        self.last_user_message = user_input

        return TurnLog(user_input=user_input, recalled=memories,
                       response=response, retained=retained)

    def close(self):
        # No persistent client to close anymore — kept for backward
        # compatibility with chat.py/demo.py/app.py, which all call this.
        pass
