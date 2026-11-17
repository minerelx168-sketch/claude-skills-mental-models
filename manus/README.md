# Running mental-models in Manus

Manus adopted the same open **Agent Skills** standard this repo is written
in, so the skill installs natively — no port, no rewrite, no API key. And
because Manus runs agents on Telegram with the same engine and the same
skills as the web app, you very likely do **not** need a Telegram bot of
your own.

> **If a Telegram advisor is the goal, read this first.**
> Install the skill in Manus, then connect Telegram from the Agents tab in
> your Manus workspace (scan the QR code). You get the advisor in Telegram
> with Manus's full toolset behind it — web search, files, connectors — and
> nothing to host, no key to pay for, no process to keep alive.
> `../telegram-advisor/` only earns its keep if you need the bot under
> **your own** bot account and token rather than Manus's.

## Install

```bash
python3 manus/build.py
# -> manus/dist/mental-models-manus.zip
```

Then in Manus: **Skills → + Add → upload the zip**. The archive root is the
`mental-models/` folder, which is the layout Manus expects.

Invoke it with `/mental-models`, or just describe a problem — the
`description` in the frontmatter is what Manus matches on to load a skill
by itself.

Two other routes Manus accepts, if you prefer them:

- **Folder** — drag `manus/build/mental-models/` in after a build.
- **GitHub** — Manus can import a skill repo, but it expects `SKILL.md` at
  the repository root. This repo keeps it at `skills/mental-models/`, so
  GitHub import will not find it as-is. Say the word and I'll set up a
  branch or a small mirror repo with the built package at the root, which
  also gets you auto-updates on push.

## Check it works

Send `/mental-models` and a real problem. A correct run names the models it
chose and why, walks each one's Thinking Steps against *your* facts rather
than restating the framework, flags any "when to avoid" condition that
applies, and closes with 3–5 concrete next steps. If it answers with
generic advice and no model names, the skill did not load.

## What the build changes, and why

The skill in `../skills/mental-models/` stays the single source of truth.
`build.py` applies a short list of host adaptations, each anchored on exact
text, then validates the result. Nothing in Steps 1–5 changes.

| Adaptation | Why |
|---|---|
| user-model directory table | `~/.claude/mental-models/` is a Claude Code path. In Manus the equivalents are `./.mental-models/`, `~/mental-models/` on the cloud computer, and files attached to the task. |
| "both globs" → "all three" | the table gained a third location; the prose had to follow |
| "if both are empty" → "if all three are empty" | same |
| OKF drop-in path | pointed at the old directory name |
| files-outside-skill list | same |
| `TEMPLATE.md` header paths | same |
| appended `## Working in Manus` | ground facts with Manus's tools before reasoning; ask rather than guess; keep chat answers in chat; reply in the user's language; stay an advisor |

**If SKILL.md changes upstream and an anchor stops matching, the build
fails** rather than shipping a package that still tells Manus to look in
`~/.claude/`. When that happens, update `REWRITES` in `build.py`.

## Validation

`build.py` checks the package the way a host would, and refuses to write a
zip that fails:

- `SKILL.md` at the skill root, with a `name` (this is the `/command`) and a
  non-empty `description` (this is how Manus decides to load it)
- every model has a non-empty OKF `type`
- every model has all of *When to Avoid*, *Thinking Steps*, *Coaching Questions*
- every `[^footnote]` resolves to a `sources[].id`
- `models/index.md` lists every model and links no missing file
- every relative link in the package resolves
- no `~/.claude` path survives anywhere

Run it without writing:

```bash
python3 manus/build.py --check
```

## Notes

- `manus/build/` and `manus/dist/` are build output and are gitignored.
- The package is 24 files, ~45 KB.
- Claude Code users: nothing here affects you. Keep using
  `../skills/mental-models/` directly.
