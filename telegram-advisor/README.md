# Telegram Mental-Models Advisor

A Telegram bot that runs the [`mental-models`](../skills/mental-models/SKILL.md)
skill as a conversational advisor. You describe a problem; it picks 2–4
models **from different areas**, walks their Thinking Steps against your
actual facts, names the conditions where those models mislead you, and ends
with concrete next steps.

It reads the same OKF bundle the skill does — including your own models in
`.mental-models/` and `~/.claude/mental-models/`, which win a name collision
with a bundled one.

## Two modes

| | Selection | Analysis | Needs |
|---|---|---|---|
| **LLM mode** | Claude reads the index and picks | Claude applies the models to your case | `ANTHROPIC_API_KEY` |
| **Offline mode** | keyword heuristics from SKILL.md Step 2 | you get the models' own Thinking Steps to work through | nothing |

The mode is chosen automatically. Offline mode is a real fallback, not a
stub — but it hands you the framework rather than reasoning for you, because
SKILL.md is explicit that model *selection* is a reasoning task and a keyword
matcher is worse at it.

## Setup

```bash
cd telegram-advisor
cp .env.example .env     # then fill in the two required values
pip install -r requirements.txt   # optional: enables LLM mode
./run.sh
```

`.env` needs at minimum:

```
TELEGRAM_BOT_TOKEN=...      # from @BotFather
TELEGRAM_ALLOWED_IDS=...    # your Telegram user id, comma separated
```

Add `ANTHROPIC_API_KEY=sk-ant-...` for LLM mode.
`.env` is gitignored — keep real tokens out of the repository.

**The allowlist is the only access control.** Anyone who finds the bot can
message it; only ids in `TELEGRAM_ALLOWED_IDS` get answers. Send `/whoami`
to the bot to learn your id, or message `@userinfobot`.

## Commands

| Command | Does |
|---|---|
| *(any message)* | advise on it |
| `/models` | list every loaded model |
| `/model <slug>` | read one model in full |
| `/reset` | clear the conversation history |
| `/status` | current mode, model count, which directories were read |
| `/whoami` | your Telegram user id |
| `/help` | usage |

## How it maps to SKILL.md

| SKILL.md | Here |
|---|---|
| Step 1 — load index, check `.mental-models/` and `~/.claude/mental-models/` | `mental_models.load_bundle()` |
| Step 2 — select 2–4 models from different areas | `llm.Advisor.select()`, or `selector.select_models()` offline |
| Step 3 — retrieve each pick | `Model.full_text()` |
| Step 4 — apply, check *When to Avoid*, show disagreement | `llm.ANALYSE_SYSTEM` |
| Step 5 — report, name the models, 3–5 next steps | reply structure in the same prompt |

## Files

```
bot.py            main loop, commands, access control
llm.py            two-stage Claude reasoning (select -> analyse)
selector.py       offline keyword selection, SKILL.md Step 2 heuristics
mental_models.py  OKF bundle loader (frontmatter + sections)
tg.py             Telegram Bot API client, HTML conversion, chunking
state.py          per-chat conversation history
test_bot.py       offline test suite — no network, no credentials
```

## Tests

```bash
python3 test_bot.py
```

Stubs both the Telegram transport and the Anthropic client, so everything
except the two real HTTP conversations is covered: commands, access control,
chunking at Telegram's 4096-character limit, HTML tag balance, long and
non-text input, state persistence, the two-stage prompt assembly, and the
fallbacks when the selector returns something unparseable.

## Running it for real

The bot is a single long-running process using long polling — no webhook, no
public URL. Anywhere that can run Python 3.10+ works.

As a systemd service:

```ini
[Unit]
Description=Mental-models Telegram advisor
After=network-online.target

[Service]
WorkingDirectory=/opt/claude-skills-mental-models/telegram-advisor
ExecStart=/usr/bin/python3 bot.py
Restart=always
RestartSec=5
User=advisor

[Install]
WantedBy=multi-user.target
```

Only one process may poll a given bot token at a time; a second one makes
Telegram return 409 to both.

## Cost

LLM mode makes two API calls per message: a short selection call at `low`
effort, and the analysis at `high`. The analysis system prompt (the full text
of the selected models) is cached, so follow-up turns in the same
conversation are substantially cheaper than the first.

## Limits

- Not a substitute for professional medical, legal or financial advice. The
  bot reasons about the *structure* of a decision and says when a
  professional is the right call.
- Offline-mode selection is keyword-based and will sometimes pick a model
  that is topically adjacent rather than genuinely useful.
- Conversation history is capped at 8 messages per chat and stored
  unencrypted in `.state/conversations.json`.
