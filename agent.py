"""
MemoryAgent: incident-response agent with persistent memory (Hindsight).

Per turn:
  1. RECALL   -> query Hindsight for concrete failure mechanisms; widen only
                 explicit follow-ups with the preceding incident
  2. RESPOND  -> validate cited dimensions and format a grounded verdict
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
import re
from datetime import date
from dataclasses import dataclass
from dotenv import load_dotenv
from hindsight_client import Hindsight
from groq import Groq

load_dotenv()

# Change HINDSIGHT_BANK_ID in .env to start from a fresh, empty memory bank.
BANK_ID = os.environ.get("HINDSIGHT_BANK_ID", "oncall-history")

MAX_MEMORIES = 8           # cap memories passed to the LLM


CLASSIFY_PROMPT = """You classify incident-history relevance. Compare the current
incident against each memory independently. Service identity is not evidence
of similarity. Ignore generic words such as service, deployment, configuration,
restart, traffic, incident, issue, and failure. Require a shared concrete
failure mechanism, not a broad operational setting.

Return only JSON with this schema:
{\"verdict\":\"strong\",\"memory_index\":1,\"aligned_evidence\":[{\"dimension\":\"mechanism\",\"current_quote\":\"verbatim quote from the current incident\",\"memory_quote\":\"verbatim quote from this memory\"}],\"resolution_quote\":\"verbatim historical resolution or empty string\"}
The verdict must be exactly strong, partial, or none. A dimension must be
exactly mechanism, symptoms, errors, resource, or root_cause.

Every quote must be an exact substring of the supplied current incident or
memory. Do not invent or paraphrase quotes. Strong requires a shared mechanism,
at least two other independent aligned dimensions among symptoms/errors/
resource/root_cause, and a directly relevant fix or attempted-fix outcome
explicitly present in the memory. Partial requires a meaningful shared mechanism plus at least one
other aligned dimension, but lacks strong evidence. It may cross service names.
Use none when the mechanism differs, only generic words overlap, or evidence
is insufficient. A 429/rate-limit incident does not match missing report data;
ZIP/archive corruption does not match rate limiting. Confirmed WORKED/DID NOT
WORK outcomes affect whether to reuse a fix, not whether unrelated incidents
match. When uncertain, choose none."""


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
            if data.get("kind") == "incident":
                return f"Engineer-reported incident (verbatim): {user_input}"
            if data.get("kind") == "outcome":
                return (
                    "Engineer-confirmed outcome.\n"
                    f"Previous engineer-reported incident (verbatim): {self.last_user_message}\n"
                    f"Previous assistant proposal referenced by the engineer: {prev_suggestion}\n"
                    f"Engineer outcome (verbatim): {user_input}"
                )
            return ""
        except Exception:
            # Preserve source text and enough labeled context for terse outcome replies.
            if re.search(r"\b(worked|did not work|didn't work|failed|no change|still failing)\b", user_input, re.I):
                return (
                    "Engineer outcome (verbatim): " + user_input + "\n"
                    f"Previous engineer-reported incident (verbatim): {self.last_user_message}\n"
                    f"Previous assistant proposal (unverified): {prev_suggestion}"
                )
            return f"Engineer statement (verbatim): {user_input}"

    def _classify_memories(self, llm: Groq, user_input: str,
                           memories: list[str]) -> tuple[str, str, str]:
        """Validate a cited, dimension-based classification and select one memory."""
        if not memories:
            return "none", "No recalled historical incidents.", ""
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
                return "none", "The memories did not yield a valid classification.", ""
            if verdict == "none":
                return "none", "No concrete failure mechanism aligned.", ""

            index = result.get("memory_index")
            if not isinstance(index, int) or isinstance(index, bool) or not 1 <= index <= len(memories):
                return "none", "The classification did not identify a recalled memory.", ""
            memory = memories[index - 1]
            pairs = result.get("aligned_evidence")
            if not isinstance(pairs, list):
                return "none", "The classification did not provide grounded evidence.", ""

            generic_words = {
                "service", "deployment", "configuration", "config", "restart",
                "traffic", "spike", "incident", "issue", "failure", "error", "problem",
                "data", "report", "reports", "daily", "http", "api", "rate", "limit",
                "request", "requests", "file", "files",
                "the", "and", "for", "with", "from", "after", "when", "that",
                "this", "was", "were", "has", "had", "into", "during", "reported",
            }
            allowed_dimensions = {"mechanism", "symptoms", "errors", "resource", "root_cause"}
            grounded = []
            for pair in pairs:
                if not isinstance(pair, dict) or pair.get("dimension") not in allowed_dimensions:
                    continue
                current_quote = str(pair.get("current_quote", "")).strip()
                memory_quote = str(pair.get("memory_quote", "")).strip()
                if (not current_quote or not memory_quote
                        or current_quote.casefold() not in user_input.casefold()
                        or memory_quote.casefold() not in memory.casefold()):
                    continue
                current_terms = set(re.findall(r"[a-z0-9]+", current_quote.casefold())) - generic_words
                memory_terms = set(re.findall(r"[a-z0-9]+", memory_quote.casefold())) - generic_words
                if current_terms and memory_terms:
                    # The failure-mechanism claim must share concrete language;
                    # the model cannot bridge unrelated quotes with an assertion.
                    if pair["dimension"] != "mechanism" or current_terms & memory_terms:
                        grounded.append((pair["dimension"], current_quote, memory_quote))

            dimensions = {pair[0] for pair in grounded}
            if "mechanism" not in dimensions:
                return "none", "No concrete shared failure mechanism was evidenced.", ""
            corroborating = dimensions & {"symptoms", "errors", "resource", "root_cause"}
            if verdict == "partial" and len(corroborating) < 1:
                return "none", "The proposed match lacks a second meaningful dimension.", ""
            resolution_quote = str(result.get("resolution_quote", "")).strip()
            has_resolution = bool(
                resolution_quote
                and resolution_quote.casefold() in memory.casefold()
                and re.search(
                    r"\b(fix|appl\w*|set|increas\w*|rais\w*|add\w*|remov\w*|"
                    r"restart\w*|roll\w* back|rollout|renew\w*|disabl\w*|enabl\w*|"
                    r"restor\w*|throttl\w*|resolv\w*|work\w*|fail\w*|upgrade\w*|"
                    r"scal\w*|tun\w*|reduc\w*|revert\w*|rollback)\b",
                    resolution_quote, re.IGNORECASE,
                )
            )
            if verdict == "strong":
                if len(corroborating) < 2 or not has_resolution:
                    return "none", "The proposed strong match lacks corroboration or a recorded resolution.", ""

            evidence = "\n".join(
                f'{dimension}: current "{current_quote}"; memory "{memory_quote}"'
                for dimension, current_quote, memory_quote in grounded
            )
            return verdict, evidence, resolution_quote if has_resolution else ""
        except Exception:
            # Fail closed: malformed classification must not become a strong match.
            return "none", "The recalled memories could not be reliably classified.", ""

    def respond(self, user_input: str) -> TurnLog:
        """Run one turn in a fresh event loop, isolated from Streamlit reruns."""
        return asyncio.run(self._respond_async(user_input))

    async def _respond_async(self, user_input: str) -> TurnLog:
        hindsight = self._new_hindsight_client()
        llm = self._new_llm_client()
        try:
            # 1. RECALL: retain follow-up context, but direct retrieval toward
            # concrete incident mechanisms rather than generic operational words.
            incident_context = user_input
            explicit_new_incident = bool(re.match(r"^\s*(new incident|incident:)", user_input, re.I))
            is_follow_up = bool(re.search(
                r"\b(that|it|this|same fix|worked|failed|did not work|didn't work|"
                r"still|tried|applied|resolved|no change)\b",
                user_input, re.I,
            ))
            if self.last_user_message and is_follow_up and not explicit_new_incident:
                incident_context = (
                    f"Previous incident or outcome: {self.last_user_message[:500]}\n"
                    f"Current report: {user_input}"
                )
            query = (
                "Find prior incidents only when they share a concrete failure mechanism. "
                "Compare symptoms, specific errors, affected resource, root cause, and recorded "
                "resolution/outcome. Generic overlap such as traffic, deployment, configuration, "
                "service, or restart is not relevance. Current incident context:\n"
                f"{incident_context}"
            )
            # Use the async SDK methods on the fresh loop owned by this turn.
            recall_result = await hindsight.arecall(bank_id=self.bank_id, query=query)
            memories = [m.text for m in recall_result.results] if recall_result.results else []
            memories = memories[:MAX_MEMORIES]
            verdict, evidence, resolution_quote = self._classify_memories(
                llm, incident_context, memories
            )

            # A none verdict has one exact, deterministic response. The model
            # cannot turn unrelated retrieval results into a forced connection.
            if verdict == "none":
                response = "No relevant historical incident found."
            else:
                # Build the answer only from verified quotes. This keeps old
                # assistant verdicts out and prevents generated fixes or values.
                label = "Strong match" if verdict == "strong" else "Partial match"
                response = f"{label}\nAligned evidence:\n{evidence}"
                if resolution_quote:
                    response += f"\nKnown from historical memory: {resolution_quote}"
                    if re.search(r"did not work|didn't work|failed", resolution_quote, re.I):
                        response += "\nThe recorded attempt failed; do not repeat it as a proven fix."
                    else:
                        response += (
                            "\nHypothesis for the current incident: consider this recorded action "
                            "only after confirming the same failure mechanism."
                        )
                else:
                    response += "\nNo resolution is recorded in the selected memory."
                if verdict == "partial":
                    response += "\nConfidence: Medium"

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
