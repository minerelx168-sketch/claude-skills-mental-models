"""Per-chat conversation history, kept small and persisted as JSON."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

log = logging.getLogger("advisor.state")


class Conversations:
    def __init__(self, path: Path, max_messages: int = 8) -> None:
        self.path = Path(path)
        self.max_messages = max_messages
        self._lock = threading.Lock()
        self._data: dict[str, list[dict]] = self._load()

    def _load(self) -> dict[str, list[dict]]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read state (%s); starting fresh", exc)
            return {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            temp.replace(self.path)
        except OSError as exc:
            log.warning("could not persist state: %s", exc)

    def history(self, chat_id: int) -> list[dict]:
        with self._lock:
            return list(self._data.get(str(chat_id), []))

    def append(self, chat_id: int, role: str, content: str) -> None:
        with self._lock:
            key = str(chat_id)
            messages = self._data.setdefault(key, [])
            messages.append({"role": role, "content": content})
            # Keep the window even, so history always starts on a user turn.
            if len(messages) > self.max_messages:
                del messages[: len(messages) - self.max_messages]
                while messages and messages[0]["role"] != "user":
                    messages.pop(0)
            self._save()

    def clear(self, chat_id: int) -> None:
        with self._lock:
            self._data.pop(str(chat_id), None)
            self._save()


class Store:
    """A tiny persisted key-value store (one JSON value per chat)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        if self.path.is_file():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("could not read %s (%s); starting fresh", self.path, exc)

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
            temp.replace(self.path)
        except OSError as exc:
            log.warning("could not persist %s: %s", self.path, exc)

    def get(self, key: int) -> dict | None:
        with self._lock:
            return self._data.get(str(key))

    def put(self, key: int, value: dict) -> None:
        with self._lock:
            self._data[str(key)] = value
            self._save()

    def drop(self, key: int) -> None:
        with self._lock:
            if self._data.pop(str(key), None) is not None:
                self._save()
