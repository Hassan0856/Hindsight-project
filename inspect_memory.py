"""
Lists everything Hindsight currently has for a bank — the seeded
incidents plus anything retained since via chat.py, demo.py, or app.py.

Useful whenever recall behaves unexpectedly: confirms whether a memory
you expect is actually in the bank at all, before assuming the bug is
in recall() or the prompt.

NOTE: this uses client.list_memories(), which is a real documented call,
but its exact return shape wasn't confirmed by the docs I could see, so
this script sniffs a few likely attribute names defensively. If it
errors, paste the traceback and we'll adjust to the real field names.

Usage:
    python inspect_memory.py                    # inspects the default bank
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
    parser.add_argument(
        "--bank", default=DEFAULT_BANK_ID,
        help="Bank id to inspect (defaults to the HINDSIGHT_BANK_ID / agent's current bank)",
    )
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
        print("(the call itself may need different arguments than assumed here)")
        client.close()
        return

    # Field names for the response object aren't confirmed, so try the
    # likely candidates before giving up.
    items = _first_attr(result, "results", "memories", "items", default=result)
    try:
        items = list(items)
    except TypeError:
        items = [items]

    if not items:
        print("(bank is empty, or the response shape didn't match — "
              "if you expected entries here, paste this output back)")
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
    print("\n(Note: list_memories() may be paginated — if you seeded 18+ "
          "incidents and see fewer here, there may be more on a later page.)")

    client.close()


if __name__ == "__main__":
    main()
