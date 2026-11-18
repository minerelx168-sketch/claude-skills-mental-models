"""
Loader and selector for the `mental-models` OKF bundle.

Mirrors skills/mental-models/SKILL.md:
  Step 1 - load the index, then the user's own model directories
  Step 2 - select 2-4 models from different areas
  Step 3 - retrieve each pick
  Step 4 - apply (the caller walks the Thinking Steps)

Pure stdlib. No PyYAML: the frontmatter in this bundle is flat enough
(scalars + simple string lists) that a small parser is honest and keeps
the bot dependency-free.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Frontmatter + body parsing
# --------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.DOTALL)


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (frontmatter_dict, body).

    Handles top-level `key: value` scalars and `key:` followed by `- item`
    lists. Nested blocks (e.g. `sources:`) are collected as raw lines under
    the key so nothing is silently lost, but are not deeply parsed -- the
    bot only ever needs title/description/tags/status/type.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    raw, body = match.group(1), match.group(2)
    data: dict = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for line in raw.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        # A list item at the top level of the current key.
        if line.startswith("- ") or line.startswith("  - "):
            if current_list is not None:
                current_list.append(_strip_quotes(line.split("- ", 1)[1].strip()))
            continue

        # Indented continuation of a nested block we do not parse deeply.
        if line.startswith(" ") or line.startswith("\t"):
            continue

        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if value:
            data[key] = _strip_quotes(value)
            current_key, current_list = None, None
        else:
            current_list = []
            current_key = key
            data[key] = current_list

    # Drop keys that opened a block but collected nothing (e.g. `sources:`).
    for key in [k for k, v in data.items() if v == []]:
        del data[key]

    return data, body


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def _split_sections(body: str) -> tuple[str, dict[str, str]]:
    """Split a model body into (lead prose, {heading: section text})."""
    parts = re.split(r"^##\s+(.+?)\s*$", body, flags=re.MULTILINE)
    lead = parts[0].strip()
    sections: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        sections[parts[i].strip()] = parts[i + 1].strip()
    return lead, sections


# --------------------------------------------------------------------------
# Model record
# --------------------------------------------------------------------------


@dataclass
class Model:
    slug: str
    title: str
    description: str
    tags: list[str]
    lead: str
    sections: dict[str, str]
    path: Path
    origin: str  # "bundled" | "project" | "personal"

    @property
    def when_to_avoid(self) -> str:
        return self.sections.get("When to Avoid", "")

    @property
    def thinking_steps(self) -> str:
        return self.sections.get("Thinking Steps", "")

    @property
    def coaching_questions(self) -> str:
        return self.sections.get("Coaching Questions", "")

    def full_text(self) -> str:
        """The model as the skill intends it to be read, minus frontmatter."""
        out = [f"# {self.title}", "", self.lead]
        for heading in ("When to Avoid", "Thinking Steps", "Coaching Questions"):
            if self.sections.get(heading):
                out += ["", f"## {heading}", "", self.sections[heading]]
        # Any section the bundle gains later still gets through.
        for heading, text in self.sections.items():
            if heading not in ("When to Avoid", "Thinking Steps", "Coaching Questions"):
                out += ["", f"## {heading}", "", text]
        return "\n".join(out)


def _load_model(path: Path, origin: str) -> Model | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    front, body = _parse_frontmatter(text)
    lead, sections = _split_sections(body)
    slug = path.stem

    tags = front.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    title = front.get("title") or slug.replace("-", " ").title()
    description = front.get("description") or lead.split("\n\n")[0][:200]

    return Model(
        slug=slug,
        title=str(title),
        description=" ".join(str(description).split()),
        tags=[str(t) for t in tags],
        lead=lead,
        sections=sections,
        path=path,
        origin=origin,
    )


# --------------------------------------------------------------------------
# Bundle
# --------------------------------------------------------------------------


@dataclass
class Bundle:
    models: dict[str, Model] = field(default_factory=dict)
    index_text: str = ""
    skill_text: str = ""
    sources: list[str] = field(default_factory=list)

    def get(self, slug: str) -> Model | None:
        return self.models.get(slug.strip().lower().replace("_", "-"))

    def search(self, needle: str) -> list[Model]:
        needle = needle.strip().lower()
        if not needle:
            return []
        hits = [m for m in self.models.values() if needle in m.slug]
        hits += [
            m
            for m in self.models.values()
            if m not in hits and needle in m.title.lower()
        ]
        return hits

    def index_lines(self) -> list[str]:
        """One line per model -- what Step 2 selects from."""
        lines = []
        for m in sorted(self.models.values(), key=lambda m: m.title.lower()):
            mark = "" if m.origin == "bundled" else f" [{m.origin}]"
            lines.append(f"- {m.slug}{mark}: {m.title} - {m.description}")
        return lines


def _model_dirs(repo_root: Path, workdir: Path) -> list[tuple[Path, str]]:
    """Bundled first, then the two user directories SKILL.md Step 1 names.

    Later entries override earlier ones on a slug collision, so a user model
    always wins -- "if someone wrote their own inversion.md, they meant it".
    """
    return [
        (repo_root / "skills" / "mental-models" / "models", "bundled"),
        (workdir / ".mental-models", "project"),
        (Path.home() / ".claude" / "mental-models", "personal"),
    ]


def load_bundle(repo_root: Path | None = None, workdir: Path | None = None) -> Bundle:
    repo_root = Path(repo_root or Path(__file__).resolve().parent.parent)
    workdir = Path(workdir or os.getcwd())

    bundle = Bundle()

    skill_path = repo_root / "skills" / "mental-models" / "SKILL.md"
    if skill_path.is_file():
        bundle.skill_text = skill_path.read_text(encoding="utf-8")

    for directory, origin in _model_dirs(repo_root, workdir):
        if not directory.is_dir():
            continue
        bundle.sources.append(f"{directory} ({origin})")
        for path in sorted(directory.glob("*.md")):
            if path.name.lower() in ("index.md", "template.md", "readme.md"):
                if path.name.lower() == "index.md" and origin == "bundled":
                    bundle.index_text = path.read_text(encoding="utf-8")
                continue
            model = _load_model(path, origin)
            if model:
                bundle.models[model.slug] = model  # later origin wins

    return bundle
