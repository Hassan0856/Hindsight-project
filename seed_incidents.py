"""
Retains every incident from incidents.json into Hindsight, simulating
weeks of on-call history that already existed before your demo recording
starts. Run this ONCE after generator.py, before you record demo.py.

Usage:
    python seed_incidents.py
"""

import os
import json
from dotenv import load_dotenv
from hindsight_client import Hindsight
from agent import BANK_ID

load_dotenv()


def main():
    client = Hindsight(
        base_url=os.environ["HINDSIGHT_BASE_URL"],
        api_key=os.environ["HINDSIGHT_API_KEY"],
    )

    # No explicit bank-creation step needed — the first retain() call
    # below creates the bank automatically.

    with open("incidents.json") as f:
        incidents = json.load(f)

    for inc in incidents:
        content = (
            f"Incident on {inc['date']} — service: {inc['service']}.\n"
            f"Symptom: {inc['symptom']}\n"
            f"Error: {inc['error_log']}\n"
            f"Root cause: {inc['root_cause']}\n"
            f"Fix applied: {inc['fix']}\n"
            f"Resolved in {inc['resolution_minutes']} minutes."
        )
        client.retain(bank_id=BANK_ID, content=content)
        print(f"Seeded: {inc['date']} — {inc['service']}")

    client.close()
    print(f"\nSeeded {len(incidents)} historical incidents into bank '{BANK_ID}'")


if __name__ == "__main__":
    main()