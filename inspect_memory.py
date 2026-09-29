"""
Lists everything Hindsight currently has for a bank — the seeded
incidents plus anything retained since. Run this FIRST whenever the
agent seems to have "lost its memory" (e.g. every response says "no
relevant historical incident found") — it tells you in one shot whether
the bank actually has data, before you assume the code is broken.

Usage:
    python inspect_memory.py
    python inspect_memory.py --bank some-other-bank
"""

import os
import argparse
from dotenv import load_dotenv
from hindsight_client import Hindsight
from agent import BANK_ID as DEFAULT_BANK_ID

load_dotenv()


def _first_attr(obj, *names, default=None):
    for name in names:
        val = getattr(obj, name, None)
        if val is not None:
            return val
    return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=DEFAULT_BANK_ID)
    args = parser.parse_args()

    client = Hindsight(
        base_url=os.environ["HINDSIGHT_BASE_URL"],
        api_key=os.environ["HINDSIGHT_API_KEY"],
    )

    print(f"Bank: {args.bank}\n{'=' * 60}")

    try:
        result = client.list_memories(bank_id=args.bank)
    except Exception as e:
        print(f"list_memories() failed: {e}")
        client.close()
        return

    items = _first_attr(result, "results", "memories", "items", default=result)
    try:
        items = list(items)
    except TypeError:
        items = [items]

    if not items:
        print("(bank is EMPTY — this is almost certainly why the agent "
              "always says 'no relevant historical incident found'. "
              "Run seed_incidents.py against this bank_id.)")
    else:
        for i, m in enumerate(items, 1):
            text = _first_attr(m, "text", "content", default=str(m))
            mtype = _first_attr(m, "type", default="")
            when = _first_attr(m, "mentioned_at", "created_at", default="")
            header = f"{i}."
            if mtype:
                header += f" [{mtype}]"
            if when:
                header += f" ({when})"
            print(header)
            print(f"   {text}\n")

    print(f"Total: {len(items)} memories")
    client.close()


if __name__ == "__main__":
    main()