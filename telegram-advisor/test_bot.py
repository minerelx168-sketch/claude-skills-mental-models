#!/usr/bin/env python3
"""
Test suite for the advisor bot. No network, no credentials.

Telegram's API and the Anthropic client are stubbed; the webhook tests run
the real HTTP server on a local port and POST to it the way Telegram does.

    python3 test_bot.py
"""

from __future__ import annotations

import http.client
import json
import contextlib
import io
import re
import socket
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bot as B  # noqa: E402
import llm as L  # noqa: E402
import webhook as W  # noqa: E402
import whytree  # noqa: E402
from tg import Telegram, TelegramError, split_message, to_telegram_html  # noqa: E402

OWNER = 1234567890
CHAT = 777
SECRET = "test-secret-0123456789abcdef"


def assert_balanced(text: str) -> None:
    """Telegram rejects a message whose tags do not balance."""
    stack = []
    for match in re.finditer(r"<(/?)([a-z]+)(?:\s[^<>]*)?>", text):
        closing, tag = match.group(1), match.group(2)
        if closing:
            assert stack and stack[-1] == tag, f"unbalanced </{tag}> in: {text[:200]}"
            stack.pop()
        else:
            stack.append(tag)
    assert not stack, f"unclosed {stack} in: {text[:200]}"


class FakeTelegram(Telegram):
    """Real formatting and chunking; only the HTTP call is replaced."""

    def __init__(self) -> None:
        super().__init__("fake-token")
        self.sent: list[dict] = []
        self.calls: list[tuple[str, dict]] = []
        self.fail_with: TelegramError | None = None

    def call(self, method: str, **params):
        self.calls.append((method, params))
        if self.fail_with and method == "getUpdates":
            raise self.fail_with
        if method == "getMe":
            return {"id": 8888888888, "username": "test_bot"}
        if method == "sendMessage":
            assert len(params["text"]) <= 4096, "chunk exceeds Telegram's limit"
            if params.get("parse_mode") == "HTML":
                assert_balanced(params["text"])
            self.sent.append(params)
        return {}


def make_bot(style: str = "brief") -> B.Bot:
    config = B.Config.__new__(B.Config)
    config.token = "fake-token"
    config.allowed = {OWNER}
    config.state_file = Path(tempfile.mkdtemp()) / "conversations.json"
    config.style = style
    bot = B.Bot(config)
    bot.tg = FakeTelegram()
    bot.advisor._client = None  # offline unless a test installs a stub
    return bot


def update(text: str | None, user: int = OWNER, update_id: int = 1, **extra) -> dict:
    message = {"message_id": 42, "chat": {"id": CHAT}, "from": {"id": user}, **extra}
    if text is not None:
        message["text"] = text
    return {"update_id": update_id, "message": message}


def say(bot: B.Bot, text: str, user: int = OWNER) -> list[str]:
    bot.tg.sent.clear()
    bot.handle(update(text, user))
    return [m["text"] for m in bot.tg.sent]


class StubAdvisor(L.Advisor):
    """An Advisor whose API call returns canned text."""

    def __init__(self, bundle, replies, error: Exception | None = None):
        self.bundle = bundle
        self.model = "claude-opus-5-5"
        self._client = object()
        self.replies = list(replies)
        self.error = error
        self.calls: list[dict] = []

    def _create(self, system, messages, max_tokens, effort):
        self.calls.append(
            {"system": system, "messages": messages, "max_tokens": max_tokens, "effort": effort}
        )
        if self.error:
            raise self.error
        return self.replies.pop(0)


BRIEF_ANSWER = """<b>แก่นของปัญหา:</b> review ติดอยู่ที่คนคนเดียว
<pre>
ส่งงานไม่ทัน
├─ รอ review นาน
│  └─ reviewer คนเดียว  ◀ แก่น
└─ scope เปลี่ยนบ่อย (?)
</pre>
<b>กรอบคิด:</b> Bottlenecks · Inversion
<b>ทำต่อ:</b>
1. ให้คนที่สองเริ่ม review PR เล็กสัปดาห์นี้
<b>ถามกลับ:</b> งานค้างที่ review เฉลี่ยกี่วัน"""


# --------------------------------------------------------------------------


class Commands(unittest.TestCase):
    def setUp(self):
        self.bot = make_bot()

    def test_start_is_short_and_explains_offline_mode(self):
        out = say(self.bot, "/start")
        self.assertEqual(len(out), 2)
        self.assertLess(len(out[0]), 450)
        self.assertIn("ต้นไม้ปัญหา", out[0])
        self.assertIn("โหมดออฟไลน์", out[1])

    def test_status_models_whoami(self):
        self.assertIn("ออฟไลน์", say(self.bot, "/status")[0])
        self.assertIn("inversion", say(self.bot, "/models")[0])
        self.assertIn(str(OWNER), say(self.bot, "/whoami")[0])
        self.assertIn("ไม่รู้จักคำสั่ง", say(self.bot, "/bogus")[0])

    def test_model_lookup(self):
        self.assertIn("Inversion", say(self.bot, "/model inversion")[0])
        self.assertIn("Bottlenecks", say(self.bot, "/model bottle")[0])  # unique prefix
        self.assertIn("ระบุชื่อ", say(self.bot, "/model")[0])

    def test_model_not_found_escapes_user_text(self):
        out = say(self.bot, "/model <b>nope")[0]
        self.assertIn("&lt;b&gt;nope", out)

    def test_stranger_gets_a_flat_refusal_and_nothing_else(self):
        out = say(self.bot, "ช่วยคิดหน่อย", user=111)
        self.assertEqual(out, ["บอทนี้เป็นบอทส่วนตัว ไม่เปิดให้ใช้งานทั่วไป"])
        self.assertIsNone(self.bot.trees.get(CHAT))

    def test_non_text_message(self):
        self.bot.handle(update(None, photo=[{}]))
        self.assertIn("เฉพาะข้อความตัวอักษร", self.bot.tg.sent[0]["text"])

    def test_update_without_message_is_ignored(self):
        self.bot.handle({"update_id": 5, "edited_message": {}})
        self.assertEqual(self.bot.tg.sent, [])


class OfflineTree(unittest.TestCase):
    """No API key: the bot asks why, the user answers, the bot holds the tree."""

    def setUp(self):
        self.bot = make_bot()

    def test_three_whys_to_a_root_cause(self):
        first = say(self.bot, "ทีมส่งงานไม่ทัน deadline ตลอด")[0]
        self.assertIn("ทำไม? (1/3)", first)
        second = say(self.bot, "รอ review นาน\nscope เปลี่ยนบ่อย")[0]
        self.assertIn("ทำไม? (2/3)", second)
        self.assertIn("รอ review นาน", second)  # drills into the first-listed cause
        third = say(self.bot, "reviewer มีคนเดียว")[0]
        self.assertIn("ทำไม? (3/3)", third)
        final = say(self.bot, "ไม่เคยเทรนคนอื่น")[0]

        self.assertIn("<pre>", final)
        self.assertIn("ไม่เคยเทรนคนอื่น  ◀ แก่น", final)
        self.assertIn("scope เปลี่ยนบ่อย", final)  # the sibling stayed on the tree
        self.assertIn("<b>แก่น:</b> ไม่เคยเทรนคนอื่น", final)
        self.assertIn("ก้าวแรก", final)
        self.assertIn("/model ", final)  # points at a model to read next
        self.assertIsNone(self.bot.trees.get(CHAT))  # session closed

    def test_capacity_root_cause_points_at_bottlenecks(self):
        for text in ["ลูกค้ายกเลิกเยอะ", "ใช้งานยากช่วงแรก", "ไม่มีคนสอนตอนเริ่มใช้"]:
            say(self.bot, text)
        final = say(self.bot, "ทีมซัพพอร์ตมีแค่ 2 คน")[0]
        self.assertIn("Bottlenecks", final)

    def test_every_reply_is_short(self):
        replies = []
        for text in ["ยอดขายตก", "ลูกค้าเก่าไม่ซื้อซ้ำ", "ของส่งช้า", "สต็อกไม่พอ"]:
            replies += say(self.bot, text)
        for reply in replies:
            self.assertLess(len(reply), 600, reply)

    def test_stop_word_finishes_early(self):
        say(self.bot, "นอนดึกทุกคืน")
        say(self.bot, "เล่นมือถือบนเตียง")
        final = say(self.bot, "พอ")[0]
        self.assertIn("เล่นมือถือบนเตียง  ◀ แก่น", final)

    def test_stop_with_nothing_on_the_tree(self):
        say(self.bot, "นอนดึกทุกคืน")
        self.assertIn("ยังไม่มีสาเหตุ", say(self.bot, "พอ")[0])
        self.assertIn("ยังไม่มีต้นไม้", say(self.bot, "พอ")[0])

    def test_dont_know_marks_the_frontier(self):
        say(self.bot, "ลูกค้ายกเลิกเยอะ")
        say(self.bot, "onboarding ยาก")
        final = say(self.bot, "ไม่รู้เหมือนกัน")[0]
        self.assertIn("onboarding ยาก  ◀ ยังไม่รู้ว่าทำไม", final)
        self.assertIn("ต้องหาคำตอบก่อน", final)

    def test_tree_command_shows_progress_and_reset_clears(self):
        self.assertIn("ยังไม่มีต้นไม้", say(self.bot, "/tree")[0])
        say(self.bot, "งานล้น")
        say(self.bot, "รับงานเกินกำลัง")
        self.assertIn("รับงานเกินกำลัง  ◀ แก่น", say(self.bot, "/tree")[0])
        say(self.bot, "งานล้น")
        say(self.bot, "/reset")
        self.assertIn("ทำไม? (1/3)", say(self.bot, "เรื่องใหม่")[0])

    def test_user_text_cannot_break_the_html(self):
        say(self.bot, "กำไร < ต้นทุน & <b>ขาดทุน")
        say(self.bot, "ราคา <i>ต่ำไป</pre>")
        final = say(self.bot, "พอ")[0]  # FakeTelegram asserts the tags balance
        self.assertIn("&lt;i&gt;ต่ำไป&lt;/pre&gt;", final)

    def test_expired_session_starts_over(self):
        say(self.bot, "เรื่องเมื่อวาน")
        data = self.bot.trees.get(CHAT)
        data["started"] = time.time() - whytree.SESSION_TTL - 1
        self.bot.trees.put(CHAT, data)
        self.assertIn("ทำไม? (1/3)", say(self.bot, "เรื่องวันนี้")[0])

    def test_tree_survives_a_restart(self):
        say(self.bot, "งานล้น")
        say(self.bot, "รับงานเกินกำลัง")
        reborn = B.Bot(self.bot.config)
        reborn.tg = FakeTelegram()
        reborn.advisor._client = None
        self.assertIn("ทำไม? (3/3)", say(reborn, "ปฏิเสธลูกค้าไม่เป็น")[0])

    def test_full_style_keeps_the_long_answer(self):
        bot = make_bot(style="full")
        out = say(bot, "ทีมส่งงานไม่ทัน แก้ตรงไหนก่อนดี")
        self.assertGreater(sum(len(m) for m in out), 3000)
        self.assertIn("เดินตามขั้นตอนนี้", "".join(out))


class WhyTreeUnit(unittest.TestCase):
    def test_split_causes(self):
        self.assertEqual(whytree.split_causes("1. ก\n- ข\n• ค"), ["ก", "ข", "ค"])
        self.assertEqual(whytree.split_causes("ก ; ข / ค"), ["ก", "ข", "ค"])
        # commas split ordinary sentences, so they are not separators
        self.assertEqual(len(whytree.split_causes("เหนื่อย, เบื่อ และไม่มีเวลา")), 1)
        self.assertEqual(len(whytree.split_causes("ก\nข\nค\nง\nจ\nฉ")), whytree.MAX_CAUSES)

    def test_long_nodes_are_clipped(self):
        tree = whytree.WhyTree("ก" * 300)
        self.assertLessEqual(len(tree.problem), whytree.NODE_CHARS)

    def test_unknown_and_stop_do_not_fire_inside_sentences(self):
        self.assertTrue(whytree.is_unknown("ไม่แน่ใจ"))
        self.assertFalse(whytree.is_unknown("เพราะหัวหน้าไม่รู้ว่าทีมทำอะไรอยู่และไม่เคยถามสักครั้ง"))
        self.assertTrue(whytree.is_stop("พอ"))
        self.assertFalse(whytree.is_stop("พอดีว่างานเยอะ"))


class LLMMode(unittest.TestCase):
    SELECT = '{"slugs":["bottlenecks","inversion"],"why":"systems; risk"}'

    def test_brief_reply_is_the_default_and_carries_a_tree(self):
        bot = make_bot()
        bot.advisor = StubAdvisor(bot.bundle, [self.SELECT, BRIEF_ANSWER])
        out = say(bot, "ทีมส่งงานไม่ทัน")
        self.assertEqual(len(out), 1)
        self.assertIn("<pre>", out[0])
        self.assertIn("│  └─ reviewer คนเดียว  ◀ แก่น", out[0])  # tree survives formatting

        select, analyse = bot.advisor.calls
        self.assertEqual((select["effort"], analyse["effort"]), ("low", "high"))
        self.assertIn("At most 120 words", analyse["system"])
        self.assertIn("ends with (?)", analyse["system"])  # guesses must be marked
        self.assertIn("# Bottlenecks", analyse["system"])
        self.assertNotIn("# Emergence", analyse["system"])  # only the picked models

    def test_full_style_uses_the_long_prompt(self):
        bot = make_bot(style="full")
        bot.advisor = StubAdvisor(bot.bundle, [self.SELECT, "คำตอบยาว"])
        say(bot, "ทีมส่งงานไม่ทัน")
        self.assertIn("250-450 words", bot.advisor.calls[1]["system"])

    def test_history_is_kept_and_threaded(self):
        bot = make_bot()
        bot.advisor = StubAdvisor(bot.bundle, [self.SELECT, "หนึ่ง", self.SELECT, "สอง"])
        say(bot, "คำถามแรก")
        say(bot, "คำถามต่อ")
        roles = [m["role"] for m in bot.advisor.calls[3]["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_bad_key_degrades_to_the_offline_tree(self):
        bot = make_bot()
        bot.advisor = StubAdvisor(bot.bundle, [], error=L.LLMUnavailable("API key ไม่ถูกต้อง"))
        out = say(bot, "ทีมส่งงานไม่ทัน")
        self.assertIn("โหมดออฟไลน์แทน", out[0])
        self.assertIn("ทำไม? (1/3)", out[1])

    def test_transient_failure_is_reported_not_swallowed(self):
        bot = make_bot()
        bot.advisor = StubAdvisor(bot.bundle, [], error=RuntimeError("ลองใหม่ใน 30 วินาที"))
        out = say(bot, "ทีมส่งงานไม่ทัน")
        self.assertEqual(len(out), 1)
        self.assertIn("เรียกโมเดลไม่สำเร็จ", out[0])

    def test_selector_reply_in_prose_or_garbage(self):
        bot = make_bot()
        prose = StubAdvisor(bot.bundle, ["I would use inversion and bottlenecks."])
        self.assertEqual(prose.select("x").source, "llm")
        junk = StubAdvisor(bot.bundle, ["???"])
        picked = junk.select("ทีมช้ามาก คอขวดอยู่ตรงไหน")
        self.assertEqual(picked.source, "offline")
        self.assertGreaterEqual(len(picked.models), 2)


class Formatting(unittest.TestCase):
    def test_markdown_becomes_telegram_html(self):
        out = to_telegram_html("## หัวข้อ\n**หนา** a < b\n- ข้อ `code` [^src]")
        self.assertEqual(out, "<b>หัวข้อ</b>\n<b>หนา</b> a &lt; b\n• ข้อ <code>code</code>")

    def test_unknown_tags_are_escaped(self):
        self.assertIn("&lt;script&gt;", to_telegram_html("<script>x</script>"))

    def test_long_text_is_split_under_the_limit(self):
        chunks = split_message("ก" * 4000 + "\n\n" + "ข" * 6000)
        self.assertTrue(all(len(c) <= 3500 for c in chunks))
        self.assertEqual(sum(len(c) for c in chunks), 10000)


class Polling(unittest.TestCase):
    def test_polling_stops_cleanly_when_a_webhook_is_registered(self):
        bot = make_bot()
        bot.tg.fail_with = TelegramError("getUpdates", "Conflict: webhook is active", 409)
        with self.assertLogs("advisor", level="ERROR") as logs:
            bot.run(skip_pending=False)  # returns instead of looping forever
        self.assertIn("webhook.py --delete", logs.output[0])


class WebhookHTTP(unittest.TestCase):
    """The real HTTP server on a local port, POSTed to the way Telegram does."""

    def setUp(self):
        self.bot = make_bot()
        self.app = W.WebhookApp(self.bot, SECRET, "/telegram/webhook", sync=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), W.make_handler(self.app))
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.app.close()

    def request(self, method, path, body=None, secret=SECRET, raw=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {}
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        if data is not None:
            headers["Content-Type"] = "application/json"
        if secret is not None:
            headers[W.SECRET_HEADER] = secret
        conn.request(method, path, body=data, headers=headers)
        response = conn.getresponse()
        response.read()
        conn.close()
        return response.status

    def post(self, body, **kwargs):
        return self.request("POST", "/telegram/webhook", body, **kwargs)

    def sent(self):
        return [m["text"] for m in self.bot.tg.sent]

    def test_health_check(self):
        self.assertEqual(self.request("GET", "/healthz", secret=None), 200)

    def test_valid_update_is_answered(self):
        self.assertEqual(self.post(update("/whoami")), 200)
        self.assertIn(str(OWNER), self.sent()[0])

    def test_missing_or_wrong_secret_is_rejected_before_the_bot_runs(self):
        self.assertEqual(self.post(update("/whoami"), secret=None), 403)
        self.assertEqual(self.post(update("/whoami"), secret="wrong-secret-value-000"), 403)
        self.assertEqual(self.sent(), [])

    def test_wrong_path_and_wrong_method(self):
        self.assertEqual(self.request("POST", "/elsewhere", update("/whoami")), 404)
        self.assertEqual(self.request("GET", "/telegram/webhook"), 404)
        self.assertEqual(self.sent(), [])

    def test_redelivered_update_is_answered_once(self):
        self.assertEqual(self.post(update("/whoami", update_id=900)), 200)
        self.assertEqual(self.post(update("/whoami", update_id=900)), 200)  # still 2xx
        self.assertEqual(len(self.sent()), 1)

    def test_malformed_bodies(self):
        self.assertEqual(self.post(None, raw=b"{not json"), 400)
        self.assertEqual(self.post(None, raw=b"[1, 2]"), 400)
        self.assertEqual(self.sent(), [])

    def test_oversized_body_is_refused_without_being_read(self):
        # Announce a huge body but send none: the server must answer from
        # the headers alone rather than wait to read a megabyte.
        with socket.create_connection(("127.0.0.1", self.port), timeout=10) as sock:
            sock.sendall(
                (
                    "POST /telegram/webhook HTTP/1.1\r\n"
                    "Host: localhost\r\n"
                    f"{W.SECRET_HEADER}: {SECRET}\r\n"
                    f"Content-Length: {W.MAX_BODY + 1}\r\n\r\n"
                ).encode()
            )
            status_line = sock.makefile("rb").readline().decode()
        self.assertIn(" 413 ", status_line)

    def test_a_bot_crash_does_not_take_the_server_down(self):
        self.bot.handle = lambda _update: 1 / 0
        with self.assertLogs("advisor.webhook", level="ERROR"):
            self.assertEqual(self.post(update("boom", update_id=901)), 200)
        self.assertEqual(self.request("GET", "/healthz", secret=None), 200)

    def test_whole_conversation_over_http(self):
        texts = ["ยอดขายตก", "ลูกค้าเก่าไม่ซื้อซ้ำ", "ของส่งช้า", "สต็อกไม่พอ"]
        for number, text in enumerate(texts):
            self.assertEqual(self.post(update(text, update_id=1000 + number)), 200)
        final = self.sent()[-1]
        self.assertIn("สต็อกไม่พอ  ◀ แก่น", final)
        self.assertIn("<pre>", final)

    def test_async_mode_acks_first_and_still_answers(self):
        app = W.WebhookApp(self.bot, SECRET, "/telegram/webhook", sync=False)
        app.dispatch(update("/whoami", update_id=950))
        app.close()  # waits for the worker pool to drain
        self.assertIn(str(OWNER), self.sent()[0])


class WebhookSetup(unittest.TestCase):
    def test_secret_rules(self):
        self.assertIsNotNone(W.check_secret(""))
        self.assertIsNotNone(W.check_secret("short"))
        self.assertIsNotNone(W.check_secret("has spaces and !!! 0123456789"))
        self.assertIsNone(W.check_secret(SECRET))

    def test_register_sends_url_secret_and_message_only_updates(self):
        bot = make_bot()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(W.register(bot, "https://example.com/", "/telegram/webhook", SECRET))
        method, params = bot.tg.calls[-1]
        self.assertEqual(method, "setWebhook")
        self.assertEqual(params["url"], "https://example.com/telegram/webhook")
        self.assertEqual(params["secret_token"], SECRET)
        self.assertEqual(params["allowed_updates"], ["message"])

    def test_register_refuses_plain_http(self):
        bot = make_bot()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(W.register(bot, "http://example.com", "/telegram/webhook", SECRET))
        self.assertEqual(bot.tg.calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
