"""
Generates a synthetic incident history — the "3-4 weeks of on-call
backstory" that gets seeded into Hindsight before the demo recording.

Grounded in 8 real, well-documented failure modes (DB pool exhaustion,
bad deploy, DNS/cert issues, cache stampede, third-party outage, OOM,
rate limiting, race conditions) so the output sounds like real postmortems
rather than generic AI filler. Each incident is forced into a strict
schema so nothing comes out vague.

Usage:
    python generator.py
    -> writes incidents.json
"""

import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

# The taxonomy — real, common root causes. Keep this list tight (this is
# what makes recall demos work: incidents sharing a root cause days apart).
FAILURE_MODES = [
    "database connection pool exhaustion under traffic spike",
    "bad deploy / config push causing cascading errors",
    "expired TLS certificate causing service-to-service auth failures",
    "cache stampede after a cold restart",
    "third-party dependency (payment processor) outage",
    "memory leak leading to OOM kill",
    "rate limit / quota exceeded during traffic spike",
    "race condition under concurrent load causing data corruption",
]

# Services in your fictional company — reuse these names across incidents
# so recurring-service patterns are visible too, not just recurring causes.
SERVICES = [
    "checkout-api", "auth-service", "payments-worker", "search-api",
    "notifications-service", "payment-api", "inventory-service",
    "recommendation-engine",
]

SCHEMA_PROMPT = """You are generating ONE realistic incident postmortem entry for a
fictional company's on-call history. Ground it in the failure mode given.
Be SPECIFIC — a real error string, a real-sounding metric or log line, a
specific timestamp, and a concrete fix (a command, config change, or
rollback action). Do NOT write vague phrases like "we fixed the issue" or
"there was a problem with the server."

Return ONLY valid JSON, no markdown fences, no commentary, matching this
exact schema:

{{
  "service": "{service}",
  "date": "a specific date, 1-28 days ago, format YYYY-MM-DD",
  "symptom": "what was observed (alerts, user reports, error rates)",
  "error_log": "one realistic log line or error message with specifics",
  "root_cause": "{failure_mode}, explained specifically for this incident",
  "fix": "the concrete action taken to resolve it",
  "resolution_minutes": a realistic integer between 8 and 90
}}

Failure mode for this incident: {failure_mode}
Service: {service}
"""


def generate_incident(failure_mode: str, service: str) -> dict:
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": SCHEMA_PROMPT.format(failure_mode=failure_mode, service=service)}
        ],
        temperature=0.8,
    )
    raw = completion.choices[0].message.content.strip()
    # strip accidental markdown fences if the model adds them anyway
    if raw.startswith("```"):
        raw = raw.strip("`").split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def main():
    incidents = []
    # Every failure mode appears on at least 2 DIFFERENT services, so
    # cross-service pattern matches (the hardest recall case) aren't
    # riding on a single hand-picked example. DB saturation gets 4
    # instances across 3 services since that's the flagship demo pattern.
    plan = [
        # --- DB connection pool / saturation cluster (flagship pattern) ---
        (FAILURE_MODES[0], "checkout-api"),          # oldest
        (FAILURE_MODES[0], "checkout-api"),          # recent recurrence, same service
        (FAILURE_MODES[0], "payment-api"),           # cross-service match
        (FAILURE_MODES[0], "inventory-service"),     # third cluster member

        # --- bad deploy / config push ---
        (FAILURE_MODES[1], "auth-service"),
        (FAILURE_MODES[1], "search-api"),

        # --- expired cert / TLS auth failures ---
        (FAILURE_MODES[2], "payments-worker"),
        (FAILURE_MODES[2], "auth-service"),

        # --- cache stampede ---
        (FAILURE_MODES[3], "search-api"),
        (FAILURE_MODES[3], "recommendation-engine"),

        # --- third-party dependency outage ---
        (FAILURE_MODES[4], "payments-worker"),
        (FAILURE_MODES[4], "notifications-service"),

        # --- memory leak / OOM ---
        (FAILURE_MODES[5], "notifications-service"),
        (FAILURE_MODES[5], "recommendation-engine"),

        # --- rate limit / quota exceeded ---
        (FAILURE_MODES[6], "search-api"),
        (FAILURE_MODES[6], "checkout-api"),

        # --- race condition under concurrent load ---
        (FAILURE_MODES[7], "checkout-api"),
        (FAILURE_MODES[7], "inventory-service"),
    ]

    for failure_mode, service in plan:
        incident = generate_incident(failure_mode, service)
        incidents.append(incident)
        print(f"Generated: {incident['service']} — {incident['root_cause'][:60]}...")

    with open("incidents.json", "w") as f:
        json.dump(incidents, f, indent=2)

    print(f"\nWrote {len(incidents)} incidents to incidents.json")
    print("Coverage: all 8 failure modes, each on 2+ different services")


if __name__ == "__main__":
    main()