"""
Runs a scripted sequence of NEW incidents against MemoryAgent and prints
each retain/recall call to the terminal — this is what you screen-record.

IMPORTANT: run generator.py then seed_incidents.py FIRST, so there's
already a few weeks of incident history in Hindsight before this runs.
This script represents "today" — new incidents coming in live, some of
which deliberately echo a seeded past incident (checkout-api DB pool
exhaustion) so the recall moment is visible on camera.

Usage:
    python demo.py
"""

import os
from dotenv import load_dotenv
from agent import MemoryAgent, BANK_ID

load_dotenv()


SCRIPT = [
    # Turn 1: a genuinely NEW incident — no past match expected.
    # Shows the "no memory yet for this one" honest baseline.
    "New incident: search-api is returning 500s intermittently, "
    "about 15% of requests over the last 10 minutes. What's the likely cause?",

    # Turn 2: deliberately echoes the seeded checkout-api DB pool exhaustion
    # incident. This is the "watch it recall" moment for your demo.
    "New incident: checkout-api is throwing 'FATAL: remaining connection "
    "slots are reserved' errors and checkout success rate just dropped to 40%. "
    "What should we check first?",

    # Turn 3: confirm the suggested fix worked -- this gets retained too,
    # closing the loop live (proves ongoing learning, not just seeded lookup).
    "That fix worked — connection pool max was hit again during the "
    "traffic spike, bumping the pool size resolved it in 12 minutes.",

    # Turn 4: same failure class comes back a third time. The agent should
    # now answer with MORE confidence/specificity than turn 2, because it
    # has both the seeded incident AND the turn-3 confirmation to draw on.
    "New incident: checkout-api connection errors again, same 'remaining "
    "connection slots are reserved' message, during another traffic spike.",
]


def main():
    agent = MemoryAgent(bank_id=BANK_ID)

    for i, turn in enumerate(SCRIPT, start=1):
        print(f"\n{'=' * 60}")
        print(f"TURN {i}")
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