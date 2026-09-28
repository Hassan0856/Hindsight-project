# Hindsight Agent Starter

Minimal, working retain/recall loop. Use-case-agnostic — swap `SYSTEM_PROMPT`
and the `SCRIPT` in `demo.py` once you've picked a use case.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# fill in HINDSIGHT_API_KEY (from Hindsight Cloud, use code MEMHACK99 for $50 credit)
# fill in GROQ_API_KEY (free tier at groq.com)
```

## Run

```bash
python demo.py
```

You should see three turns print to the terminal, with turn 3 explicitly
recalling the preference stated in turn 2 — that's the "before/after"
memory proof judges are scoring on (25% of the rubric).

## How this maps to the files you'll produce

- `agent.py` — the actual retain/recall loop. This is your core code
  snippet for the article (Prompt 2 in the content guide).
- `demo.py` — this is what you screen-record for the video.
- Once you pick a use case, `SYSTEM_PROMPT` and `SCRIPT` are the only
  things that need to change to start — everything else is reusable.

## Next steps

- Swap the generic script for a real scenario (sales calls, support
  tickets, incident logs — whatever use case you land on)
- Add a thin UI (Streamlit is fastest) if you want something more visual
  than a terminal for judging
- Decide what "worth retaining" means for your use case — right now every
  turn is retained wholesale; you may want to be more selective (e.g. only
  retain extracted facts, not full raw exchanges)
