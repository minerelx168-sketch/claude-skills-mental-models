#!/usr/bin/env python3
"""
Mental-models advisor as a Telegram bot.

Runs the `mental-models` skill's procedure over Telegram and answers the
way a chat wants: short, with a problem tree that drills from the symptom
down to a cause the user can act on.

Two transports, same bot:
  * webhook.py -- Telegram POSTs updates to a public HTTPS URL (recommended)
  * bot.py     -- long polling; no public URL needed (this file's main)

Two brains, chosen automatically:
  * LLM mode     -- ANTHROPIC_API_KEY is set: Claude picks the models and
                    drafts the tree and the next steps in one reply.
  * offline mode -- no key: the bot asks "why?" level by level and builds
                    the tree from the user's own answers.

Configuration (environment, or a .env file beside this script):
  TELEGRAM_BOT_TOKEN   required
  TELEGRAM_ALLOWED_IDS required -- comma-separated Telegram user ids
  ANTHROPIC_API_KEY    optional -- enables LLM mode
  ADVISOR_STYLE        optional -- "brief" (default) or "full"
  ADVISOR_MODEL        optional -- defaults to claude-opus-5-5
  ADVISOR_STATE_FILE   optional -- where to persist conversations
"""

from __future__ import annotations

import html
import logging
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm import Advisor, LLMUnavailable  # noqa: E402
from mental_models import Model, load_bundle  # noqa: E402
from selector import matched_areas, select_models  # noqa: E402
from state import Conversations, Store  # noqa: E402
from tg import Telegram, TelegramError  # noqa: E402
from whytree import MAX_DEPTH, WhyTree, is_stop, is_unknown, split_causes  # noqa: E402

log = logging.getLogger("advisor")

MAX_PROBLEM_CHARS = 6000
HISTORY_TURNS = 8  # user+assistant messages kept per chat


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def load_dotenv(path: Path) -> None:
    """Minimal .env reader -- real values never belong in the repo."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        os.environ.setdefault(key, value)


class Config:
    def __init__(self) -> None:
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        raw_ids = os.environ.get("TELEGRAM_ALLOWED_IDS", "")
        self.allowed = {
            int(part) for part in raw_ids.replace(" ", "").split(",") if part.isdigit()
        }
        self.state_file = Path(
            os.environ.get("ADVISOR_STATE_FILE", HERE / ".state" / "conversations.json")
        )
        style = os.environ.get("ADVISOR_STYLE", "brief").strip().lower()
        self.style = style if style in ("brief", "full") else "brief"

    def check(self) -> list[str]:
        problems = []
        if not self.token:
            problems.append("TELEGRAM_BOT_TOKEN ยังไม่ได้ตั้งค่า")
        if not self.allowed:
            problems.append(
                "TELEGRAM_ALLOWED_IDS ยังไม่ได้ตั้งค่า — บอทจะไม่ตอบใครเลยถ้าไม่ระบุ"
            )
        return problems


# --------------------------------------------------------------------------
# Static replies
# --------------------------------------------------------------------------

WELCOME = """\
<b>ที่ปรึกษา Mental Models</b>
เล่าปัญหามาได้เลย — ผมตอบสั้นๆ พร้อมต้นไม้ปัญหาที่ไล่ “ทำไม” ลงไปจนถึงแก่น

/tree ดูต้นไม้ตอนนี้ · /reset เริ่มเรื่องใหม่
/models กรอบคิดทั้งหมด · /model &lt;ชื่อ&gt; อ่านเต็ม
/status โหมดที่ใช้อยู่"""

OFFLINE_NOTE = """\
<i>โหมดออฟไลน์ (ยังไม่มี ANTHROPIC_API_KEY): ผมจะถาม “ทำไม” ทีละชั้น \
แล้วประกอบต้นไม้จากคำตอบของคุณเอง — ผมจัดโครงให้ แต่ยังไม่ได้วิเคราะห์แทนคุณ</i>"""

OFFLINE_FULL_NOTE = """\
<i>โหมดออฟไลน์: ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY ผมจึงเลือกกรอบคิดด้วยคีย์เวิร์ด \
และส่งขั้นตอนของแต่ละกรอบมาให้คุณเดินเอง — ยังไม่ได้วิเคราะห์เคสของคุณโดยตรง</i>"""


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def render_models_list(bundle) -> str:
    lines = ["<b>กรอบคิดที่มีอยู่</b>", ""]
    for line in bundle.index_lines():
        slug, _, rest = line[2:].partition(": ")
        lines.append(f"• <code>{slug}</code> — {rest}")
    lines.append("")
    lines.append("อ่านตัวไหนเต็มๆ พิมพ์  <code>/model ชื่อ</code>")
    return "\n".join(lines)


def render_model(model: Model) -> str:
    parts = [f"<b>{model.title}</b>", "", model.lead]
    if model.when_to_avoid:
        parts += ["", "<b>เมื่อไรไม่ควรใช้</b>", model.when_to_avoid]
    if model.thinking_steps:
        parts += ["", "<b>ขั้นตอนการคิด</b>", model.thinking_steps]
    if model.coaching_questions:
        parts += ["", "<b>คำถามชวนคิด</b>", model.coaching_questions]
    if model.origin != "bundled":
        parts += ["", f"<i>(กรอบคิดของคุณเอง — {model.origin})</i>"]
    return "\n".join(parts)


def render_offline_advice(problem: str, models: list[Model]) -> str:
    """The long offline answer (ADVISOR_STYLE=full): the models' own steps."""
    areas = matched_areas(problem)
    parts = [OFFLINE_FULL_NOTE, "", "<b>กรอบคิดที่เลือกให้</b>"]
    for model in models:
        parts.append(f"• <b>{model.title}</b> — {model.description}")
    if areas:
        parts.append("")
        parts.append("<i>เลือกจากสัญญาณ: " + " / ".join(areas) + "</i>")

    for model in models:
        parts += ["", "─" * 18, f"<b>{model.title}</b>"]
        if model.thinking_steps:
            parts += ["", "<b>เดินตามขั้นตอนนี้</b>", model.thinking_steps]
        if model.when_to_avoid:
            parts += ["", "<b>เมื่อไรไม่ควรใช้</b>", model.when_to_avoid]
        if model.coaching_questions:
            parts += ["", "<b>ถามตัวเอง</b>", model.coaching_questions]
    return "\n".join(parts)


def render_why_question(tree: WhyTree) -> str:
    """The next "why?" -- raw Telegram HTML, user text escaped."""
    level = tree.depth + 1
    if tree.depth == 0:
        return (
            f"<b>ปัญหา:</b> {esc(tree.problem)}\n"
            f"<b>ทำไม? ({level}/{MAX_DEPTH})</b> อะไรทำให้เกิดสิ่งนี้\n"
            "<i>ตอบสั้นๆ · หลายสาเหตุให้ขึ้นบรรทัดใหม่ ตัวหนักสุดไว้บนสุด · พิมพ์ พอ เพื่อดูต้นไม้</i>"
        )
    return (
        f"<b>ทำไม? ({level}/{MAX_DEPTH})</b> "
        f"แล้วอะไรทำให้ “{esc(tree.focus['text'])}”"
    )


def render_tree_result(tree: WhyTree, models: list[Model]) -> str:
    """The finished tree -- raw Telegram HTML, user text escaped."""
    focus = esc(tree.focus["text"])
    parts = ["<b>ต้นไม้ปัญหา</b>", f"<pre>{esc(tree.render())}</pre>"]
    if tree.unknown:
        parts.append(f"<b>แก่น:</b> ยังไม่รู้ว่าทำไม “{focus}” — นี่คือสิ่งที่ต้องหาคำตอบก่อน")
        parts.append("<b>ก้าวแรก:</b> ใคร หรือข้อมูลอะไร ตอบคำถามนี้ได้")
    else:
        parts.append(f"<b>แก่น:</b> {focus}")
        parts.append(f"<b>ก้าวแรก:</b> สัปดาห์นี้ทำอะไรได้หนึ่งอย่างเพื่อแก้ “{focus}”")
    if models:
        names = " · ".join(esc(m.title) for m in models)
        parts.append(f"<b>มองต่อผ่าน:</b> {names}  (<code>/model {models[0].slug}</code>)")
    return "\n".join(parts)


def render_status(bundle, advisor: Advisor, style: str) -> str:
    mode = "LLM (Claude)" if advisor.available else "ออฟไลน์ (ถาม-ตอบสร้างต้นไม้)"
    lines = ["<b>สถานะ</b>", "", f"โหมด: <b>{mode}</b>", f"รูปแบบคำตอบ: <b>{style}</b>"]
    if advisor.available:
        lines.append(f"โมเดล: <code>{advisor.model}</code>")
    else:
        lines.append(f"เหตุผล: {advisor.why_unavailable()}")
    lines += [
        f"กรอบคิดที่โหลดได้: <b>{len(bundle.models)}</b>",
        "",
        "<b>แหล่งที่มา</b>",
    ]
    lines += [f"• <code>{source}</code>" for source in bundle.sources]
    extra = [m for m in bundle.models.values() if m.origin != "bundled"]
    if extra:
        lines.append("")
        lines.append(f"กรอบคิดของคุณเอง: {len(extra)} ตัว")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Bot
# --------------------------------------------------------------------------


class Bot:
    """Transport-agnostic: polling and the webhook both feed `handle()`."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bundle = load_bundle(repo_root=HERE.parent)
        self.advisor = Advisor(self.bundle)
        self.tg = Telegram(config.token)
        self.conversations = Conversations(config.state_file, max_messages=HISTORY_TURNS)
        self.trees = Store(config.state_file.with_name("trees.json"))
        self.running = True

    # -- polling lifecycle --------------------------------------------------

    def stop(self, *_args) -> None:
        log.info("shutting down")
        self.running = False

    def run(self, skip_pending: bool = True) -> None:
        me = self.tg.get_me()
        log.info(
            "polling as @%s (id %s) | %d models | mode=%s | style=%s",
            me.get("username"),
            me.get("id"),
            len(self.bundle.models),
            "llm" if self.advisor.available else "offline",
            self.config.style,
        )
        if skip_pending:
            self.tg.drop_pending()

        backoff = 1
        while self.running:
            try:
                updates = self.tg.get_updates()
                backoff = 1
            except TelegramError as exc:
                if exc.code == 409:
                    log.error(
                        "a webhook is registered for this bot, so polling is refused. "
                        "Run `python3 webhook.py --delete` first, or use webhook.py."
                    )
                    return
                log.warning("getUpdates failed: %s", exc)
                time.sleep(min(backoff, 60))
                backoff *= 2
                continue
            except Exception as exc:  # network blip, DNS, timeout
                log.warning("poll error: %s", exc)
                time.sleep(min(backoff, 60))
                backoff *= 2
                continue

            for update in updates:
                if not self.running:
                    break
                try:
                    self.handle(update)
                except Exception:
                    log.exception("failed handling update %s", update.get("update_id"))

    # -- dispatch -----------------------------------------------------------

    def handle(self, update: dict) -> None:
        message = update.get("message")
        if not message:
            return

        chat_id = message["chat"]["id"]
        user_id = (message.get("from") or {}).get("id")
        text = (message.get("text") or "").strip()
        message_id = message.get("message_id")

        if user_id not in self.config.allowed:
            log.warning("ignored message from unauthorised user %s", user_id)
            # Tell them once rather than leaving a silent bot, but say nothing
            # about what this bot is or who owns it.
            self.tg.send(chat_id, "บอทนี้เป็นบอทส่วนตัว ไม่เปิดให้ใช้งานทั่วไป")
            return

        if not text:
            self.tg.send(
                chat_id, "ผมอ่านได้เฉพาะข้อความตัวอักษร — พิมพ์ปัญหามาได้เลย", message_id
            )
            return

        if text.startswith("/"):
            self.command(chat_id, user_id, text, message_id)
            return

        self.advise(chat_id, text, message_id)

    def command(self, chat_id: int, user_id: int, text: str, message_id: int) -> None:
        command, _, argument = text.partition(" ")
        command = command.split("@", 1)[0].lower()
        argument = argument.strip()

        if command in ("/start", "/help"):
            self.tg.send(chat_id, WELCOME, message_id)
            if not self.advisor.available:
                self.tg.send(chat_id, OFFLINE_NOTE)

        elif command == "/models":
            self.tg.send(chat_id, render_models_list(self.bundle), message_id)

        elif command == "/model":
            if not argument:
                self.tg.send(
                    chat_id, "ระบุชื่อกรอบคิดด้วย เช่น <code>/model inversion</code>", message_id
                )
                return
            model = self.bundle.get(argument)
            if not model:
                hits = self.bundle.search(argument)
                if len(hits) == 1:
                    model = hits[0]
                elif hits:
                    options = " ".join(f"<code>{m.slug}</code>" for m in hits[:8])
                    self.tg.send(chat_id, f"หมายถึงตัวไหน: {options}", message_id)
                    return
                else:
                    self.tg.send(
                        chat_id,
                        f"ไม่พบ <code>{esc(argument)}</code> — ดูรายการที่ /models",
                        message_id,
                        raw=True,
                    )
                    return
            self.tg.send(chat_id, render_model(model), message_id)

        elif command == "/tree":
            tree = self._load_tree(chat_id)
            if tree is None:
                self.tg.send(chat_id, "ยังไม่มีต้นไม้ค้างอยู่ — เล่าปัญหามาได้เลย", message_id)
            else:
                self._finish_tree(chat_id, tree, message_id)

        elif command == "/reset":
            self.conversations.clear(chat_id)
            self.trees.drop(chat_id)
            self.tg.send(chat_id, "ล้างแล้ว เริ่มเรื่องใหม่ได้เลย", message_id)

        elif command == "/status":
            self.tg.send(
                chat_id,
                render_status(self.bundle, self.advisor, self.config.style),
                message_id,
            )

        elif command == "/whoami":
            self.tg.send(chat_id, f"user id ของคุณคือ <code>{user_id}</code>", message_id)

        else:
            self.tg.send(chat_id, "ไม่รู้จักคำสั่งนี้ — ดูที่ /help", message_id)

    # -- advising -----------------------------------------------------------

    def advise(self, chat_id: int, problem: str, message_id: int) -> None:
        if len(problem) > MAX_PROBLEM_CHARS:
            problem = problem[:MAX_PROBLEM_CHARS]
            self.tg.send(chat_id, "<i>ข้อความยาวมาก ผมอ่านเฉพาะส่วนต้นนะ</i>", message_id)

        self.tg.send_typing(chat_id)

        if self.advisor.available:
            history = self.conversations.history(chat_id)
            try:
                answer, selection = self.advisor.advise(problem, history, self.config.style)
            except LLMUnavailable as exc:
                # A bad key should degrade the bot, not silence it.
                log.error("llm unavailable: %s", exc)
                self.tg.send(chat_id, f"<i>{esc(str(exc))} — ใช้โหมดออฟไลน์แทน</i>", raw=True)
            except RuntimeError as exc:
                log.error("llm call failed: %s", exc)
                self.tg.send(chat_id, f"เรียกโมเดลไม่สำเร็จ: {exc}", message_id)
                return
            else:
                self.conversations.append(chat_id, "user", problem)
                self.conversations.append(chat_id, "assistant", answer)
                if selection.source == "offline":
                    answer += "\n\n<i>(เลือกกรอบคิดด้วยคีย์เวิร์ด — ตัวเลือกจากโมเดลอ่านไม่ออก)</i>"
                self.tg.send(chat_id, answer, message_id)
                return

        if self.config.style == "full":
            models = select_models(problem, self.bundle)
            self.tg.send(chat_id, render_offline_advice(problem, models), message_id)
        else:
            self._why_step(chat_id, problem, message_id)

    # -- offline problem tree ----------------------------------------------

    def _load_tree(self, chat_id: int) -> WhyTree | None:
        data = self.trees.get(chat_id)
        if not data:
            return None
        try:
            tree = WhyTree.from_dict(data)
        except (KeyError, TypeError, ValueError):
            self.trees.drop(chat_id)
            return None
        if tree.expired:
            self.trees.drop(chat_id)
            return None
        return tree

    def _why_step(self, chat_id: int, text: str, message_id: int) -> None:
        """One turn of the drill-down: record the answer, ask the next why."""
        tree = self._load_tree(chat_id)

        if tree is None:
            if is_stop(text):
                self.tg.send(chat_id, "ยังไม่มีต้นไม้ค้างอยู่ — เล่าปัญหามาได้เลย", message_id)
                return
            tree = WhyTree(text)
            self.trees.put(chat_id, tree.to_dict())
            self.tg.send(chat_id, render_why_question(tree), message_id, raw=True)
            return

        if is_stop(text):
            self._finish_tree(chat_id, tree, message_id)
            return

        if is_unknown(text):
            # "I don't know why" is a finding: it marks where the digging
            # has to continue outside this chat.
            tree.unknown = True
        else:
            tree.add_causes(split_causes(text))

        if tree.complete:
            self._finish_tree(chat_id, tree, message_id)
        else:
            self.trees.put(chat_id, tree.to_dict())
            self.tg.send(chat_id, render_why_question(tree), message_id, raw=True)

    def _finish_tree(self, chat_id: int, tree: WhyTree, message_id: int) -> None:
        self.trees.drop(chat_id)
        if tree.depth == 0 and not tree.unknown:
            self.tg.send(
                chat_id, "ต้นไม้ยังไม่มีสาเหตุเลย — เล่าปัญหาใหม่ได้ทุกเมื่อ", message_id
            )
            return
        # Select on everything the user said, not just the opening line.
        models = select_models(tree.all_text(), self.bundle, want=2)[:2]
        self.tg.send(chat_id, render_tree_result(tree, models), message_id, raw=True)


# --------------------------------------------------------------------------


def configure() -> Config | None:
    """Shared startup for both transports. Returns None if misconfigured."""
    logging.basicConfig(
        level=os.environ.get("ADVISOR_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    )
    load_dotenv(HERE / ".env")

    config = Config()
    problems = config.check()
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print(f"\nคัดลอก .env.example เป็น .env แล้วกรอกค่า  ({HERE / '.env'})", file=sys.stderr)
        return None
    return config


def main() -> int:
    config = configure()
    if config is None:
        return 1
    bot = Bot(config)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
