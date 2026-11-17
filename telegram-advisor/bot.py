#!/usr/bin/env python3
"""
Mental-models advisor as a Telegram bot.

Runs the `mental-models` skill's procedure over Telegram: read the problem,
pick 2-4 models from different areas, walk their Thinking Steps against the
user's facts, surface the "when to avoid" conditions, end with concrete
next steps.

Two modes, chosen automatically:
  * LLM mode     -- ANTHROPIC_API_KEY is set: Claude selects and analyses.
  * offline mode -- no key: keyword heuristics select, and the bot hands
                    back the models' own Thinking Steps to work through.

Configuration (environment, or a .env file beside this script):
  TELEGRAM_BOT_TOKEN   required
  TELEGRAM_ALLOWED_IDS required -- comma-separated Telegram user ids
  ANTHROPIC_API_KEY    optional -- enables LLM mode
  ADVISOR_MODEL        optional -- defaults to claude-opus-5-5
  ADVISOR_STATE_FILE   optional -- where to persist conversations
"""

from __future__ import annotations

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
from state import Conversations  # noqa: E402
from tg import Telegram, TelegramError  # noqa: E402

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

ผมช่วยคุณ <i>คิด</i> ไม่ใช่ตัดสินใจแทน — เล่าปัญหามาตามปกติ ผมจะเลือกกรอบคิด 2–4 ตัว \
จากคนละสาย แล้วเดินตามขั้นตอนของแต่ละกรอบกับข้อเท็จจริงของคุณจริงๆ บอกทั้งจุดที่กรอบ \
เหล่านั้นเห็นตรงกัน จุดที่เห็นต่าง และเงื่อนไขที่กรอบนั้นใช้ไม่ได้

<b>ลองพิมพ์แบบนี้</b>
• ควรลาออกมาทำธุรกิจตัวเองไหม เงินเก็บพอ 8 เดือน
• ทีมส่งงานไม่ทันตลอด แก้ตรงไหนก่อนดี
• ยอดขายพุ่ง 40% หลังเปลี่ยนโฆษณา ควรทุ่มงบเพิ่มไหม

<b>คำสั่ง</b>
/models — รายการกรอบคิดทั้งหมด
/model &lt;ชื่อ&gt; — อ่านกรอบคิดตัวนั้นเต็มๆ
/reset — ล้างบทสนทนา เริ่มเรื่องใหม่
/status — โหมดที่ใช้อยู่ และจำนวนกรอบคิดที่โหลดได้
/help — ข้อความนี้

ยิ่งเล่าละเอียด (ตัวเลข ข้อจำกัด สิ่งที่ลองมาแล้ว) คำตอบยิ่งใช้ได้จริง"""

OFFLINE_NOTE = """\
<i>โหมดออฟไลน์: ยังไม่ได้ตั้งค่า ANTHROPIC_API_KEY ผมจึงเลือกกรอบคิดด้วยคีย์เวิร์ด \
และส่งขั้นตอนของแต่ละกรอบมาให้คุณเดินเอง — ยังไม่ได้วิเคราะห์เคสของคุณโดยตรง</i>"""


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


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
    areas = matched_areas(problem)
    parts = [OFFLINE_NOTE, "", "<b>กรอบคิดที่เลือกให้</b>"]
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

    parts += [
        "",
        "─" * 18,
        "ตอบคำถามพวกนี้กลับมาได้เลย ผมจะช่วยไล่ต่อ "
        "— หรือตั้ง <code>ANTHROPIC_API_KEY</code> เพื่อให้ผมวิเคราะห์เคสคุณเองเต็มรูปแบบ",
    ]
    return "\n".join(parts)


def render_status(bundle, advisor: Advisor) -> str:
    mode = "LLM (Claude)" if advisor.available else "ออฟไลน์ (คีย์เวิร์ด)"
    lines = [
        "<b>สถานะ</b>",
        "",
        f"โหมด: <b>{mode}</b>",
    ]
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
    def __init__(self, config: Config) -> None:
        self.config = config
        self.bundle = load_bundle(repo_root=HERE.parent)
        self.advisor = Advisor(self.bundle)
        self.tg = Telegram(config.token)
        self.conversations = Conversations(config.state_file, max_messages=HISTORY_TURNS)
        self.running = True

    # -- lifecycle --------------------------------------------------------

    def stop(self, *_args) -> None:
        log.info("shutting down")
        self.running = False

    def run(self, skip_pending: bool = True) -> None:
        me = self.tg.get_me()
        log.info(
            "connected as @%s (id %s) | %d models | mode=%s",
            me.get("username"),
            me.get("id"),
            len(self.bundle.models),
            "llm" if self.advisor.available else "offline",
        )
        if skip_pending:
            self.tg.drop_pending()

        backoff = 1
        while self.running:
            try:
                updates = self.tg.get_updates()
                backoff = 1
            except TelegramError as exc:
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

    # -- dispatch ---------------------------------------------------------

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
                chat_id,
                "ผมอ่านได้เฉพาะข้อความตัวอักษร — พิมพ์ปัญหาที่อยากปรึกษามาได้เลย",
                message_id,
            )
            return

        if text.startswith("/"):
            self.command(chat_id, user_id, text, message_id)
            return

        self.advise(chat_id, user_id, text, message_id)

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
                        f"ไม่พบ <code>{argument}</code> — ดูรายการทั้งหมดที่ /models",
                        message_id,
                    )
                    return
            self.tg.send(chat_id, render_model(model), message_id)

        elif command == "/reset":
            self.conversations.clear(chat_id)
            self.tg.send(chat_id, "ล้างบทสนทนาแล้ว เริ่มเรื่องใหม่ได้เลย", message_id)

        elif command == "/status":
            self.tg.send(chat_id, render_status(self.bundle, self.advisor), message_id)

        elif command == "/whoami":
            self.tg.send(
                chat_id, f"user id ของคุณคือ <code>{user_id}</code>", message_id
            )

        else:
            self.tg.send(chat_id, "ไม่รู้จักคำสั่งนี้ — ดูที่ /help", message_id)

    # -- the actual advising ----------------------------------------------

    def advise(self, chat_id: int, user_id: int, problem: str, message_id: int) -> None:
        if len(problem) > MAX_PROBLEM_CHARS:
            problem = problem[:MAX_PROBLEM_CHARS]
            self.tg.send(chat_id, "<i>ข้อความยาวมาก ผมอ่านเฉพาะส่วนต้นนะ</i>", message_id)

        self.tg.send_typing(chat_id)
        history = self.conversations.history(chat_id)

        if not self.advisor.available:
            models = select_models(problem, self.bundle)
            self.tg.send(chat_id, render_offline_advice(problem, models), message_id)
            return

        try:
            answer, selection = self.advisor.advise(problem, history)
        except LLMUnavailable as exc:
            log.error("llm unavailable: %s", exc)
            models = select_models(problem, self.bundle)
            self.tg.send(chat_id, f"<i>{exc}</i>", message_id)
            self.tg.send(chat_id, render_offline_advice(problem, models))
            return
        except RuntimeError as exc:
            log.error("llm call failed: %s", exc)
            self.tg.send(chat_id, f"เรียกโมเดลไม่สำเร็จ: {exc}", message_id)
            return

        self.conversations.append(chat_id, "user", problem)
        self.conversations.append(chat_id, "assistant", answer)

        if selection.source == "offline":
            answer += "\n\n<i>(เลือกกรอบคิดด้วยคีย์เวิร์ด — ตัวเลือกจากโมเดลอ่านไม่ออก)</i>"
        self.tg.send(chat_id, answer, message_id)


# --------------------------------------------------------------------------


def main() -> int:
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
        return 1

    bot = Bot(config)
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
