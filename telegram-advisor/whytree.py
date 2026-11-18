"""
The problem tree: ask "why?" of a problem, and of each answer, until the
cause at the bottom is one the user can act on.

Offline (no API key) the bot cannot reason about the user's case, so it does
the next best honest thing: it asks the questions, the user supplies the
causes, and the bot holds the structure. With a key, Claude drafts the whole
tree in one reply instead (see llm.ANALYSE_SYSTEM_BRIEF) and this module is
not involved.
"""

from __future__ import annotations

import re
import time

MAX_DEPTH = 3            # three whys: short enough for chat, deep enough to leave the symptom
MAX_CAUSES = 4           # siblings kept per answer
NODE_CHARS = 70          # longer lines wrap badly in a Telegram code block
SESSION_TTL = 6 * 3600   # an abandoned tree should not swallow tomorrow's question

STOP_WORDS = {"พอ", "พอแล้ว", "จบ", "หยุด", "done", "stop", "enough"}
UNKNOWN_HINTS = (
    "ไม่รู้", "ไม่ทราบ", "ไม่แน่ใจ", "นึกไม่ออก",
    "don't know", "dont know", "not sure", "no idea",
)


def clip(text: str, limit: int = NODE_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def is_stop(text: str) -> bool:
    return text.strip().lower().rstrip(".!") in STOP_WORDS


def is_unknown(text: str) -> bool:
    """A short "I don't know" -- not a sentence that happens to contain one."""
    lowered = text.strip().lower()
    return len(lowered) <= 30 and any(hint in lowered for hint in UNKNOWN_HINTS)


def split_causes(text: str) -> list[str]:
    """One answer may name several causes: one per line, or `;` / ` / ` separated.

    Commas are deliberately not separators -- they split ordinary sentences.
    """
    parts = [p for p in re.split(r"\n+", text) if p.strip()]
    if len(parts) == 1:
        parts = [p for p in re.split(r"\s*;\s*|\s+/\s+", parts[0]) if p.strip()]

    causes = []
    for part in parts:
        part = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", part)   # bullets, "1." "2)"
        part = clip(part)
        if part and part not in causes:
            causes.append(part)
    return causes[:MAX_CAUSES]


class WhyTree:
    """A tree of causes with a cursor on the branch being drilled into."""

    def __init__(self, problem: str) -> None:
        self.root = {"text": clip(problem), "children": []}
        self.path: list[int] = []      # child indexes from the root to the focus
        self.unknown = False           # the user hit "I don't know" at the focus
        self.started = time.time()

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "path": self.path,
            "unknown": self.unknown,
            "started": self.started,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WhyTree":
        tree = cls(data["root"]["text"])
        tree.root = data["root"]
        tree.path = list(data.get("path", []))
        tree.unknown = bool(data.get("unknown", False))
        tree.started = float(data.get("started", time.time()))
        return tree

    @property
    def expired(self) -> bool:
        return time.time() - self.started > SESSION_TTL

    # -- structure ----------------------------------------------------------

    @property
    def problem(self) -> str:
        return self.root["text"]

    @property
    def focus(self) -> dict:
        node = self.root
        for index in self.path:
            node = node["children"][index]
        return node

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def complete(self) -> bool:
        return self.unknown or self.depth >= MAX_DEPTH

    def add_causes(self, causes: list[str]) -> None:
        """Hang causes under the focus, then drill into the first one.

        The user is asked to put the heaviest cause first, so the first is
        the branch worth following; the rest stay on the tree as siblings.
        """
        if not causes:
            return
        focus = self.focus
        focus["children"] = [{"text": c, "children": []} for c in causes]
        self.path.append(0)

    def all_text(self) -> str:
        """Everything on the tree -- a better signal for model selection
        than the opening one-liner alone."""
        out: list[str] = []

        def walk(node: dict) -> None:
            out.append(node["text"])
            for child in node["children"]:
                walk(child)

        walk(self.root)
        return "\n".join(out)

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        """Plain-text tree. The caller escapes it and wraps it in <pre>."""
        focus = self.focus if self.path else None
        marker = "  ◀ ยังไม่รู้ว่าทำไม" if self.unknown else "  ◀ แก่น"
        lines = [self.root["text"]]

        def walk(node: dict, prefix: str) -> None:
            children = node["children"]
            for index, child in enumerate(children):
                last = index == len(children) - 1
                mark = marker if child is focus else ""
                lines.append(f"{prefix}{'└─ ' if last else '├─ '}{child['text']}{mark}")
                walk(child, prefix + ("   " if last else "│  "))

        walk(self.root, "")
        return "\n".join(lines)
