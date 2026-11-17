#!/usr/bin/env python3
"""
Build a Manus-installable package from the canonical mental-models skill.

Manus accepts Agent Skills in the same SKILL.md format this repo already
uses, so there is no second copy of the skill to maintain: this script
takes `skills/mental-models/` as the single source of truth, applies a
short, explicit list of host adaptations, validates the result, and writes
a zip whose archive root is the skill folder.

Every rewrite is anchored on exact text. If an anchor stops matching --
because SKILL.md changed upstream -- the build FAILS rather than quietly
shipping a package that still tells Manus to look in ~/.claude/.

    python3 manus/build.py
    python3 manus/build.py --check    # validate only, write nothing
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "skills" / "mental-models"
BUILD = REPO / "manus" / "build"
DIST = REPO / "manus" / "dist"
SKILL_NAME = "mental-models"

# --------------------------------------------------------------------------
# Host adaptations -- (description, exact old text, new text)
# --------------------------------------------------------------------------

REWRITES: list[tuple[str, str, str]] = [
    (
        "TEMPLATE.md header paths",
        """  .mental-models/<name>.md          this project's models — commit them with the code
  ~/.claude/mental-models/<name>.md your personal models, available everywhere""",
        """  ./.mental-models/<name>.md  this project's models — commit them with the code
  ~/mental-models/<name>.md   your own models, kept on this cloud computer across tasks""",
    ),
    (
        "user-model directory table",
        """| Path | Holds |
|---|---|
| `.mental-models/` (relative to the working directory) | this project's or team's models, committed with the code |
| `~/.claude/mental-models/` | the user's personal models, available everywhere |""",
        """| Path | Holds |
|---|---|
| `./.mental-models/` (relative to the working directory) | models that came with the project or repo you are working in |
| `~/mental-models/` | the user's own models, kept on this cloud computer across tasks |
| files attached to this task | models the user uploaded just now — check the attachments before deciding the set is empty |""",
    ),
    (
        "both-globs wording (three locations here, not two)",
        "**Always run both globs before selecting. Do not assume they are empty** — one tool call",
        "**Always check all three before selecting. Do not assume they are empty** — one tool call",
    ),
    (
        "both-are-empty wording",
        "If both are empty, carry on with the bundled models.",
        "If all three are empty, carry on with the bundled models.",
    ),
    (
        "OKF drop-in path",
        "`type`. That means any OKF bundle can be dropped into `.mental-models/` and read the same way,",
        "`type`. That means any OKF bundle can be dropped into one of the directories above and read\nthe same way,",
    ),
    (
        "files-outside-skill list",
        """- `.mental-models/*.md` — the working directory's own models
- `~/.claude/mental-models/*.md` — the user's personal models""",
        """- `./.mental-models/*.md` — the working directory's own models
- `~/mental-models/*.md` — the user's own models, kept across tasks
- any model files attached to the current task""",
    ),
]

# Appended verbatim. Keeps the five steps untouched: this says how to run
# them well on this host, not what they are.
MANUS_SECTION = """

## Working in Manus

The five steps above are the skill. This section is about running them well
here.

**Ground the facts before you reason.** Thinking Steps ask for real numbers —
an expected load, a conversion rate, a competitor's price, how long a
migration took last time. If the user has not given you one and you can
find it, find it: search the web, read the files they attached, query a
connected data source. A model walked against invented numbers produces
confident, specific, wrong answers. If a number cannot be found, say which
one is missing and what you assumed instead.

**Ask before you guess at the problem.** If the user's description is too
thin to tell which models fit, ask one or two sharp questions first. One
good question beats a generic analysis.

**Match the output to the request.** In chat — including Telegram — reply in
chat: short paragraphs, no tables, no deep nesting. Produce a document or a
deliverable file only when the user asks for one, or when the analysis is
genuinely too large to read in a message.

**Reply in the user's language.** Translate the headings; keep the model
names in English so they stay searchable.

**Stay an advisor.** You reason about the structure of a decision. You do
not give medical, legal or financial advice as though licensed — when that
is what the situation needs, say so and say why.
"""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class Problem(Exception):
    pass


def parse_frontmatter(text: str) -> tuple[str, str]:
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", text, re.DOTALL)
    if not match:
        raise Problem("no YAML frontmatter")
    return match.group(1), match.group(2)


def validate(root: Path) -> list[str]:
    """Check the package the way a host would. Returns a list of notes."""
    notes: list[str] = []

    skill_md = root / "SKILL.md"
    if not skill_md.is_file():
        raise Problem("SKILL.md missing from the skill root")

    front, body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    name = re.search(r"^name:\s*(.+)$", front, re.M)
    description = re.search(r"^description:\s*(.+)$", front, re.M)
    if not name:
        raise Problem("SKILL.md frontmatter has no `name` (this is the /command)")
    if not description:
        raise Problem("SKILL.md frontmatter has no `description` (this is how the host decides to load it)")
    if name.group(1).strip() != SKILL_NAME:
        raise Problem(f"frontmatter name is {name.group(1)!r}, expected {SKILL_NAME!r}")
    notes.append(f"invoked as /{name.group(1).strip()}")
    notes.append(f"description {len(description.group(1).strip())} chars")

    models_dir = root / "models"
    if not models_dir.is_dir():
        raise Problem("models/ directory missing")

    model_files = sorted(p for p in models_dir.glob("*.md") if p.name != "index.md")
    if not model_files:
        raise Problem("models/ has no model files")

    for path in model_files:
        model_front, model_body = parse_frontmatter(path.read_text(encoding="utf-8"))
        if not re.search(r"^type:\s*\S", model_front, re.M):
            raise Problem(f"{path.name}: OKF requires a non-empty `type` in frontmatter")
        source_ids = set(re.findall(r"^- id:\s*(\S+)", model_front, re.M))
        used = set(re.findall(r"\[\^([a-z0-9_-]+)\]", model_body, re.I))
        orphans = used - source_ids
        if orphans:
            raise Problem(f"{path.name}: footnote(s) with no matching source id: {sorted(orphans)}")
        for heading in ("When to Avoid", "Thinking Steps", "Coaching Questions"):
            if f"## {heading}" not in model_body:
                raise Problem(f"{path.name}: missing section '## {heading}'")

    index = models_dir / "index.md"
    if not index.is_file():
        raise Problem("models/index.md missing — a model absent from it is invisible to the agent")
    linked = set(re.findall(r"\]\(([a-z0-9-]+)\.md\)", index.read_text(encoding="utf-8")))
    actual = {p.stem for p in model_files}
    if actual - linked:
        raise Problem(f"not listed in index.md: {sorted(actual - linked)}")
    if linked - actual:
        raise Problem(f"index.md links missing files: {sorted(linked - actual)}")
    notes.append(f"{len(model_files)} models, all indexed")

    # Nothing may still point at a Claude-Code-only location.
    leftovers = []
    for path in root.rglob("*.md"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "~/.claude" in line:
                leftovers.append(f"{path.relative_to(root)}:{lineno}")
    if leftovers:
        raise Problem(f"host-specific path ~/.claude still present at: {leftovers}")

    # Relative links inside the package must resolve.
    broken = []
    for path in root.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\]\((?!https?://|#)([^)]+)\)", text):
            target = target.split("#")[0].strip()
            if not target:
                continue
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(root)} -> {target}")
    if broken:
        raise Problem(f"broken relative links: {broken}")
    notes.append("all relative links resolve")

    return notes


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def apply_rewrites(root: Path) -> list[str]:
    """Apply every rewrite across the staged package, exactly once each."""
    files = sorted(root.rglob("*.md"))
    applied = []

    for label, old, new in REWRITES:
        hits = [p for p in files if old in p.read_text(encoding="utf-8")]
        total = sum(p.read_text(encoding="utf-8").count(old) for p in hits)
        if total == 0:
            raise Problem(
                f"adaptation {label!r} did not match any file.\n"
                "The skill changed upstream; update REWRITES in manus/build.py "
                "rather than shipping a package that still names a Claude Code path."
            )
        if total > 1:
            raise Problem(
                f"adaptation {label!r} matched {total} times "
                f"in {[p.name for p in hits]}; the anchor is ambiguous"
            )
        target = hits[0]
        target.write_text(
            target.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
        )
        applied.append(f"{label}  ({target.relative_to(root)})")

    return applied


def build(check_only: bool = False) -> int:
    if not SOURCE.is_dir():
        print(f"error: source skill not found at {SOURCE}", file=sys.stderr)
        return 1

    if BUILD.exists():
        shutil.rmtree(BUILD)
    staged = BUILD / SKILL_NAME
    shutil.copytree(SOURCE, staged)

    try:
        applied = apply_rewrites(staged)
    except Problem as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    skill_md = staged / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8").rstrip("\n")
    skill_md.write_text(text + "\n" + MANUS_SECTION.rstrip("\n") + "\n", encoding="utf-8")

    try:
        notes = validate(staged)
    except Problem as exc:
        print(f"error: package invalid — {exc}", file=sys.stderr)
        return 1

    print(f"source      {SOURCE.relative_to(REPO)}")
    for label in applied:
        print(f"adapted     {label}")
    print("adapted     appended '## Working in Manus'")
    for note in notes:
        print(f"ok          {note}")

    if check_only:
        shutil.rmtree(BUILD)
        print("\ncheck only — nothing written")
        return 0

    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / f"{SKILL_NAME}-manus.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(staged.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(BUILD))

    total = sum(1 for p in staged.rglob("*") if p.is_file())
    print(f"\nwrote       {archive}")
    print(f"            {total} files, {archive.stat().st_size // 1024} KB")
    print(f"            archive root is {SKILL_NAME}/ — upload this zip to Manus as-is")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    args = parser.parse_args()
    return build(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
