"""
Interactive chat with the incident response agent. Use this to poke at it
yourself during development, and for live judge Q&A.

Talks to the SAME bank as seed_incidents.py, demo.py and app.py.

Usage:
    python chat.py
    (type 'exit' or 'quit' to stop)
"""

import os
from dotenv import load_dotenv
from agent import MemoryAgent, BANK_ID

load_dotenv()


def main():
    print(f"Incident Response Agent — bank: {BANK_ID}")
    print("Type an incident, or 'exit' to quit.\n")
    agent = MemoryAgent(bank_id=BANK_ID)

    try:
        while True:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            log = agent.respond(user_input)

            if log.recalled:
                print(f"\n[recalled {len(log.recalled)} past memories]")
                for i, m in enumerate(log.recalled, 1):
                    print(f"  {i}. {m}")
            else:
                print("\n[no matching past memories]")

            print(f"\nAgent: {log.response}\n")
            if log.retained:
                print(f"[retained] {log.retained}\n")
    finally:
        agent.close()


if __name__ == "__main__":
    main()