"""
Runs a scripted sequence of NEW incidents against MemoryAgent, structured
specifically to show the contrast that makes memory legible in ~60 seconds:
  No match -> Strong match -> Partial match (cross-service) -> live learning

This is what you screen-record. Pair it with video_script.md for narration.

IMPORTANT: run generator.py then seed_incidents.py FIRST on a CLEAN bank
(new HINDSIGHT_BANK_ID), so there's real history behind this and no
leftover duplicates from earlier testing.

Usage:
    python demo.py
"""

import os
from dotenv import load_dotenv
from agent import MemoryAgent, BANK_ID

load_dotenv()

# Each turn is (narration_beat, message) so the terminal output doubles as
# a teleprompter while you record.
SCRIPT = [
    (
        "BEAT 1 — Honesty first. This is a novel problem, nothing like it exists "
        "in memory. Watch it say so instead of forcing a match.",
        "New incident: recommendation-engine is returning identical, "
        "non-personalized rankings for every user since last night — "
        "personalization appears to have stopped working entirely.",
    ),
    (
        "BEAT 2 — Strong match. Same service, same failure, recurring.",
        "New incident: checkout-api is throwing 'FATAL: remaining connection "
        "slots are reserved' again, DB pool near capacity during a flash-sale "
        "traffic spike.",
    ),
    (
        "BEAT 3 — Partial match. Different service, same underlying pattern. "
        "This is the moment that proves it's reasoning about the pattern, "
        "not just matching keywords or service names.",
        "New incident: payment-api — DB utilization at 96%, requests are "
        "slow, payments are failing with 504s.",
    ),
    (
        "BEAT 4 — Confirm the fix. This gets retained live.",
        "That fix worked on payment-api — raising the connection pool and "
        "restarting resolved it in 14 minutes.",
    ),
    (
        "BEAT 5 — Live learning. Same failure returns; it now has both the "
        "seed history AND the confirmation from BEAT 4 to draw on.",
        "New incident: payment-api is seeing connection pool pressure again "
        "during another traffic spike.",
    ),
]


def main():
    agent = MemoryAgent(bank_id=BANK_ID)

    for i, (beat, turn) in enumerate(SCRIPT, start=1):
        print(f"\n{'=' * 60}")
        print(f"TURN {i} — {beat}")
        print(f"{'=' * 60}")
        print(f"USER: {turn}")

        log = agent.respond(turn)

        print(f"\n[RECALLED {len(log.recalled)} MEMORIES]")
        for m in log.recalled:
            print(f"  - {m}")

        print(f"\nAGENT: {log.response}")
        if log.retained:
            print(f"\n[RETAINED] {log.retained}")

    agent.close()


if __name__ == "__main__":
    main()