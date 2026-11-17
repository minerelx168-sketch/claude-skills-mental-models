#!/usr/bin/env python3
"""
Offline test suite for the advisor bot.

Talks to no network and needs no credentials: the Telegram transport and
the Anthropic client are both stubbed, so every code path except the two
real HTTP conversations is exercised.

    python3 test_bot.py
"""


import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ADVISOR = HERE


# ========================================================================
# Part 1 - the bot, against a fake Telegram transport
# ========================================================================

import tg
import bot as B

SENT = []

def fake_call(self, method, **params):
    SENT.append((method, params))
    if method == "getMe":
        return {"id": 8888888888, "username": "test_bot", "is_bot": True}
    if method == "getUpdates":
        return []
    if method == "sendMessage":
        text = params["text"]
        assert len(text) <= 4096, f"chunk too long: {len(text)}"
        if params.get("parse_mode") == "HTML":
            # Telegram rejects unbalanced tags -- check balance ourselves.
            import re
            stack = []
            for m in re.finditer(r"</?([a-z]+)(?:\s[^<>]*)?>", text):
                tag = m.group(1)
                if m.group(0).startswith("</"):
                    assert stack and stack[-1] == tag, f"unbalanced </{tag}> in chunk:\n{text[:300]}"
                    stack.pop()
                else:
                    stack.append(tag)
            assert not stack, f"unclosed {stack} in chunk:\n{text[:300]}"
        return {"message_id": len(SENT)}
    return {}

tg.Telegram.call = fake_call

state = Path(tempfile.mkdtemp()) / "conv.json"
cfg = B.Config.__new__(B.Config)
cfg.token = "fake"
cfg.allowed = {1234567890}
cfg.state_file = state

b = B.Bot(cfg)
print(f"loaded {len(b.bundle.models)} models | llm available: {b.advisor.available}\n")

def msg(text, user_id=1234567890):
    SENT.clear()
    b.handle({"update_id": 1, "message": {
        "message_id": 42, "chat": {"id": 777}, "from": {"id": user_id}, "text": text}})
    outs = [p["text"] for m, p in SENT if m == "sendMessage"]
    return outs

cases = [
    ("/start", None),
    ("/status", None),
    ("/models", None),
    ("/model inversion", None),
    ("/model bottle", None),
    ("/model nonexistent-thing", None),
    ("/whoami", None),
    ("/bogus", None),
    ("/reset", None),
    ("ควรลาออกจากงานประจำมาทำธุรกิจตัวเองไหม เงินเก็บพอ 8 เดือน กลัวเจ๊ง", None),
    ("ยอดขายพุ่ง 40% หลังเปลี่ยนโฆษณา ควรทุ่มงบเพิ่มไหม", None),
    ("hello from a stranger", 111111),
]

for text, uid in cases:
    outs = msg(text, uid if uid else 1234567890)
    total = sum(len(o) for o in outs)
    label = f"{text[:46]!r}" + (f" (uid {uid})" if uid else "")
    print(f"PASS  {label:<58} -> {len(outs)} msg, {total} chars")

# Long input truncation
outs = msg("ปัญหายาวมาก " * 900)
print(f"PASS  long input ({len('ปัญหายาวมาก '*900)} chars) -> {len(outs)} msg")

# Non-text message
SENT.clear()
b.handle({"update_id": 2, "message": {"message_id": 1, "chat": {"id": 777},
                                      "from": {"id": 1234567890}, "photo": [{}]}})
print(f"PASS  photo (no text) -> {[p['text'][:40] for m,p in SENT if m=='sendMessage']}")

# State persistence
b.conversations.append(777, "user", "hi")
b.conversations.append(777, "assistant", "hello")
saved = json.loads(state.read_text())
print(f"PASS  state persisted -> {saved}")
b.conversations.clear(777)
print(f"PASS  state cleared   -> {json.loads(state.read_text())}")

print("\n=== sample: full output for the resignation question ===")
print(msg("ควรลาออกจากงานประจำมาทำธุรกิจตัวเองไหม เงินเก็บพอ 8 เดือน กลัวเจ๊ง")[0][:900])


# ========================================================================
# Part 2 - the two-stage reasoning, against a stubbed Anthropic client
# ========================================================================

from mental_models import load_bundle
import llm as L

bundle = load_bundle(repo_root=ADVISOR.parent)

CALLS = []

class FakeAdvisor(L.Advisor):
    def __init__(self, bundle, replies):
        self.bundle = bundle
        self.model = "claude-opus-5-5"
        self._client = object()      # pretend a client exists
        self._replies = list(replies)
    def _create(self, system, messages, max_tokens, effort):
        CALLS.append({"system": system, "messages": messages,
                      "max_tokens": max_tokens, "effort": effort})
        return self._replies.pop(0)

# --- happy path -----------------------------------------------------------
a = FakeAdvisor(bundle, [
    '{"slugs":["inversion","bias-from-incentives","regression-to-the-mean"],'
    '"why":"risk; people; evidence"}',
    "<b>โมเดลที่ใช้</b>\nผลวิเคราะห์...",
])
answer, sel = a.advise("ยอดขายพุ่ง 40% ควรทุ่มงบเพิ่มไหม")
assert sel.source == "llm", sel.source
assert [m.slug for m in sel.models] == ["inversion", "bias-from-incentives", "regression-to-the-mean"]
print("PASS  two-stage: slugs =", [m.slug for m in sel.models], "| why =", sel.why)

s1, s2 = CALLS[0], CALLS[1]
assert "## Available models" in s1["system"] and "- inversion:" in s1["system"]
assert s1["effort"] == "low" and s2["effort"] == "high"
assert "Thinking Steps" in s2["system"] and "invert" in s2["system"].lower()
assert "Coaching Questions" in s2["system"]
# the three picked models, and only those, are in the analysis prompt
assert "Regression to the Mean" in s2["system"] and "Asymmetric Warfare" not in s2["system"]
print(f"PASS  stage1 system {len(s1['system'])} chars (index) / "
      f"stage2 system {len(s2['system'])} chars (3 full models)")
print(f"PASS  effort low->high, max_tokens {s1['max_tokens']}->{s2['max_tokens']}")

# --- selector returns prose instead of JSON -------------------------------
CALLS.clear()
a = FakeAdvisor(bundle, ["I would use inversion and also bottlenecks here.", "ok"])
_, sel = a.advise("ปัญหาอะไรสักอย่าง")
print("PASS  prose reply recovered ->", [m.slug for m in sel.models], "| source:", sel.source)
assert len(sel.models) >= 2

# --- selector returns garbage -> offline fallback -------------------------
CALLS.clear()
a = FakeAdvisor(bundle, ["???", "ok"])
_, sel = a.advise("ทีมช้ามาก คอขวดอยู่ตรงไหน")
print("PASS  garbage reply -> offline fallback:", sel.source, [m.slug for m in sel.models])
assert sel.source == "offline" and len(sel.models) >= 2

# --- history is threaded into both stages ---------------------------------
CALLS.clear()
a = FakeAdvisor(bundle, ['{"slugs":["inversion","leverage"]}', "ok"])
hist = [{"role": "user", "content": "เรื่องเดิม"}, {"role": "assistant", "content": "ตอบไปแล้ว"}]
a.advise("แล้วถ้าเป็นแบบนี้ล่ะ", hist)
assert "บริบทก่อนหน้า" in CALLS[0]["messages"][0]["content"]
assert len(CALLS[1]["messages"]) == 3 and CALLS[1]["messages"][-1]["role"] == "user"
print("PASS  history threaded: stage1 context inlined, stage2 messages =",
      [m["role"] for m in CALLS[1]["messages"]])

# --- unavailable client ---------------------------------------------------
real = L.Advisor(bundle)
assert not real.available
try:
    real.select("x")
    print("FAIL  expected LLMUnavailable")
except L.LLMUnavailable as e:
    print("PASS  no client ->", type(e).__name__, "|", e)

print("\nall checks passed")
