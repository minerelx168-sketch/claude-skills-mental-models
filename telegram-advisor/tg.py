"""
A small Telegram Bot API client -- long polling, HTML-safe sending, and
message chunking. Standard library only.
"""

from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API_ROOT = "https://api.telegram.org"
MAX_MESSAGE = 4096          # Telegram's hard limit, in UTF-16 code units
CHUNK_TARGET = 3500         # leave room for the HTML tags we add


class TelegramError(RuntimeError):
    def __init__(self, method: str, description: str, code: int | None = None):
        super().__init__(f"{method}: {description}")
        self.method = method
        self.description = description
        self.code = code


class Telegram:
    def __init__(self, token: str, timeout: int = 60):
        self._token = token
        self._timeout = timeout
        self.offset: int | None = None

    # -- transport --------------------------------------------------------

    def call(self, method: str, **params) -> dict:
        url = f"{API_ROOT}/bot{self._token}/{method}"
        data = json.dumps(params).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                raise TelegramError(method, body or str(exc), exc.code) from exc

            description = payload.get("description", body)
            # 429: Telegram tells us exactly how long to wait.
            retry_after = payload.get("parameters", {}).get("retry_after")
            if retry_after:
                time.sleep(int(retry_after) + 1)
                return self.call(method, **params)
            raise TelegramError(method, description, exc.code) from exc

        if not payload.get("ok"):
            raise TelegramError(method, payload.get("description", "unknown error"))
        return payload.get("result")

    # -- api --------------------------------------------------------------

    def get_me(self) -> dict:
        return self.call("getMe")

    def get_updates(self, poll_timeout: int = 50) -> list[dict]:
        params = {
            "timeout": poll_timeout,
            "allowed_updates": ["message"],
        }
        if self.offset is not None:
            params["offset"] = self.offset

        updates = self.call("getUpdates", **params) or []
        if updates:
            self.offset = updates[-1]["update_id"] + 1
        return updates

    def drop_pending(self) -> None:
        """Skip messages queued while the bot was down."""
        updates = self.call("getUpdates", timeout=0, offset=-1) or []
        if updates:
            self.offset = updates[-1]["update_id"] + 1

    def send_typing(self, chat_id: int) -> None:
        try:
            self.call("sendChatAction", chat_id=chat_id, action="typing")
        except TelegramError:
            pass  # cosmetic only

    def send(self, chat_id: int, text: str, reply_to: int | None = None) -> None:
        """Send text as HTML, split into Telegram-sized chunks."""
        chunks = split_message(to_telegram_html(text))
        for index, chunk in enumerate(chunks):
            params = {
                "chat_id": chat_id,
                "text": chunk,
                "parse_mode": "HTML",
                "link_preview_options": {"is_disabled": True},
            }
            if index == 0 and reply_to:
                params["reply_to_message_id"] = reply_to
                params["allow_sending_without_reply"] = True
            try:
                self.call("sendMessage", **params)
            except TelegramError as exc:
                # Unbalanced tags are the only realistic HTML failure; retry
                # the same chunk as plain text rather than dropping it.
                if "parse" in exc.description.lower() or "entity" in exc.description.lower():
                    params.pop("parse_mode", None)
                    params["text"] = strip_tags(chunk)
                    self.call("sendMessage", **params)
                else:
                    raise


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

_ALLOWED_TAGS = ("b", "i", "u", "s", "code", "pre", "a", "blockquote")
_TAG_RE = re.compile(
    r"</?(?:" + "|".join(_ALLOWED_TAGS) + r")(?:\s[^<>]*)?>", re.IGNORECASE
)


def to_telegram_html(text: str) -> str:
    """Convert the markdown-ish text Claude writes into Telegram HTML.

    Telegram accepts only a short tag whitelist, so everything else is
    escaped. Tags the model already emitted (the system prompt asks for
    <b>) are preserved by escaping around them.
    """
    # Protect the tags we allow, escape the rest, then restore.
    placeholders: list[str] = []

    def stash(match: re.Match) -> str:
        placeholders.append(match.group(0))
        return f"\x00{len(placeholders) - 1}\x00"

    text = _TAG_RE.sub(stash, text)
    text = html.escape(text, quote=False)

    # Markdown that survives into the output.
    text = re.sub(r"```[a-zA-Z0-9_+-]*\n(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    text = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", text)
    text = re.sub(r"^#{1,6}\s*(.+?)\s*$", r"<b>\1</b>", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+", "• ", text, flags=re.MULTILINE)
    # Footnote markers from the OKF bundle ([^jacobi]) are noise in a chat.
    text = re.sub(r"\s*\[\^[a-z0-9_-]+\]", "", text, flags=re.IGNORECASE)
    # Markdown links -> Telegram anchors.
    text = re.sub(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', text)

    def restore(match: re.Match) -> str:
        return placeholders[int(match.group(1))]

    text = re.sub(r"\x00(\d+)\x00", restore, text)
    return text.strip()


def strip_tags(text: str) -> str:
    return html.unescape(re.sub(r"<[^<>]+>", "", text))


def split_message(text: str, limit: int = CHUNK_TARGET) -> list[str]:
    """Split on paragraph, then line, then hard boundaries.

    Never splits inside a tag, because a chunk is only cut at a newline.
    """
    if len(text) <= limit:
        return [text] if text else ["(ไม่มีข้อความ)"]

    chunks: list[str] = []
    for block in text.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        if chunks and len(chunks[-1]) + len(block) + 2 <= limit:
            chunks[-1] += "\n\n" + block
            continue
        if len(block) <= limit:
            chunks.append(block)
            continue
        # A single oversized paragraph: break it on lines, then on length.
        current = ""
        for line in block.split("\n"):
            while len(line) > limit:
                chunks.append(line[:limit])
                line = line[limit:]
            if len(current) + len(line) + 1 > limit:
                chunks.append(current)
                current = line
            else:
                current = f"{current}\n{line}" if current else line
        if current:
            chunks.append(current)

    return [c for c in chunks if c.strip()] or ["(ไม่มีข้อความ)"]
