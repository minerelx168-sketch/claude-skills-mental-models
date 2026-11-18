"""
Offline model selection -- SKILL.md Step 2 without an LLM.

SKILL.md is explicit that selection is a reasoning task and that "you are
better at it than any keyword matcher". This module *is* the keyword
matcher, so it is the fallback, not the default: when an API key is
configured the bot asks the model to choose instead (see llm.py).

What it does preserve is the mechanical part of Step 2 -- the eight
discovery heuristics, and the rule that picks must span different areas so
a single-area pick does not leave blind spots unchecked.
"""

from __future__ import annotations

import re
import unicodedata

from mental_models import Bundle, Model

# The eight discovery heuristics from SKILL.md Step 2, verbatim in their
# model lists. Keywords are Thai + English because the advisor is bilingual.
# Thai is written without spaces, so these are matched as substrings; keep
# them long enough not to fire inside unrelated words.
HEURISTICS: list[dict] = [
    {
        "area": "risk",
        "label": "ความเสี่ยง / ความไม่แน่นอน / ย้อนกลับไม่ได้",
        "models": ["inversion", "margin-of-safety", "randomness"],
        "keywords": [
            "เสี่ยง", "ไม่แน่นอน", "ล้มเหลว", "เจ๊ง", "ขาดทุน", "หนี้", "ล้มละลาย",
            "ย้อนกลับ", "แก้ไม่ได้", "สำรอง", "เผื่อไว้", "กันไว้", "ป้องกัน",
            "กลัว", "worst case", "เลวร้ายที่สุด", "ถ้าพลาด", "เดิมพัน",
            "risk", "risky", "uncertain", "uncertainty", "fail", "failure",
            "downside", "irreversible", "safety", "buffer", "hedge", "gamble",
        ],
    },
    {
        "area": "reframing",
        "label": "ตัน / มองไม่เห็นทางเลือก",
        "models": ["first-principles-thinking", "second-order-thinking", "framing"],
        "keywords": [
            "คิดไม่ออก", "ทางตัน", "ไม่รู้จะ", "ทางเลือก", "ตัวเลือก", "มองไม่เห็น",
            "ติดอยู่", "วนอยู่", "สับสน", "เริ่มยังไง", "ทำยังไงดี", "แก้ยังไง",
            "สมมติฐาน", "ตั้งต้นใหม่", "คิดใหม่", "มุมมอง",
            "stuck", "options", "alternative", "confused", "rethink",
            "from scratch", "assumption", "reframe", "blocked", "no idea",
        ],
    },
    {
        "area": "conflict",
        "label": "ความขัดแย้ง / เจรจา / แข่งขัน",
        "models": ["asymmetric-warfare", "trade-offs", "bias-from-incentives"],
        "keywords": [
            "คู่แข่ง", "แข่งขัน", "ตัดราคา", "เจรจา", "ต่อรอง", "ขัดแย้ง", "ทะเลาะ",
            "แย่งลูกค้า", "ฝ่ายตรงข้าม", "รายใหญ่", "รายเล็ก", "ยักษ์ใหญ่", "สู้กับ",
            "เอาชนะ", "ชิง", "สงคราม",
            "compete", "competitor", "competition", "negotiate", "negotiation",
            "conflict", "rival", "incumbent", "undercut", "bargain", "price war",
        ],
    },
    {
        "area": "systems",
        "label": "ระบบซับซ้อน / ผลข้างเคียงที่ไม่ตั้งใจ",
        "models": ["feedback-loops", "emergence", "bottlenecks", "leverage"],
        "keywords": [
            "ระบบ", "ซับซ้อน", "ผลข้างเคียง", "ลูกโซ่", "กระทบ", "วงจร", "พันกัน",
            "ไม่ตั้งใจ", "บานปลาย", "ยิ่งแก้ยิ่ง", "ภาพรวม", "องค์รวม", "ส่วนประกอบ",
            "ทบต้น", "ยิ่งนานยิ่ง", "ลามไป",
            "system", "complex", "side effect", "unintended", "cascade", "loop",
            "spiral", "knock-on", "ecosystem", "emergent", "holistic", "compound",
        ],
    },
    {
        "area": "performance",
        "label": "ประสิทธิภาพ / การเพิ่มผลลัพธ์",
        "models": ["bottlenecks", "leverage", "activation-energy"],
        "keywords": [
            "ช้ามาก", "ล่าช้า", "คอขวด", "ประสิทธิภาพ", "เร่งงาน", "ปรับปรุง",
            "ผลผลิต", "ทำได้มากขึ้น", "คุ้มที่สุด", "จุดคุ้ม", "ไม่ทัน", "งานล้น",
            "กำลังการผลิต", "ขยาย", "โต", "สเกล",
            "มีแค่", "คนเดียว", "ไม่เพียงพอ", "รับไม่ไหว", "คนไม่พอ",
            "slow", "bottleneck", "throughput", "performance", "efficiency",
            "scale", "speed up", "productivity", "capacity", "constraint", "optimize",
        ],
    },
    {
        "area": "people",
        "label": "คน / ทีม / พฤติกรรม",
        "models": ["social-proof", "confirmation-bias", "bias-from-incentives"],
        "keywords": [
            "ทีม", "ลูกน้อง", "หัวหน้า", "พนักงาน", "เพื่อนร่วมงาน", "พฤติกรรม",
            "วัฒนธรรม", "แรงจูงใจ", "โบนัส", "ค่าคอม", "ลาออก", "จ้าง", "สัมภาษณ์",
            "คนอื่น", "ใครๆ", "ตามกระแส", "ความเห็น", "เชื่อว่า", "มั่นใจว่า",
            "team", "manager", "employee", "hire", "culture", "behaviour", "behavior",
            "incentive", "bonus", "commission", "quit", "morale", "peer", "everyone",
        ],
    },
    {
        "area": "evidence",
        "label": "ประเมินหลักฐาน / ตัวเลข / ผลลัพธ์",
        "models": ["sampling", "regression-to-the-mean", "randomness"],
        "keywords": [
            "ข้อมูล", "ตัวเลข", "สถิติ", "ผลลัพธ์", "วัดผล", "ทดลอง", "ทดสอบ",
            "สำรวจ", "แบบสอบถาม", "กลุ่มตัวอย่าง", "พิสูจน์", "หลักฐาน",
            "ยอดขาย", "ยอดเข้าชม", "ยอดผู้ใช้", "คอนเวอร์ชัน", "เมตริก", "รีวิว",
            "พุ่งขึ้น", "พุ่ง", "ดีขึ้น", "แย่ลง", "ตกลง", "ลดลง", "เพิ่มขึ้น",
            "เปอร์เซ็นต์", "เดือนนี้", "ไตรมาส", "เฉลี่ย", "สถิติดีขึ้น",
            "data", "metric", "statistic", "sample", "survey", "a/b", "ab test",
            "experiment", "evidence", "spike", "dropped", "improved", "conversion",
            "benchmark", "average", "outlier",
        ],
    },
    {
        "area": "change",
        "label": "การเปลี่ยนแปลงที่ไม่เริ่ม หรือไม่หยุด",
        "models": ["activation-energy", "inertia"],
        "keywords": [
            "เริ่มไม่ได้", "ไม่ได้สักที", "ผัดวัน", "ไม่ยอมเปลี่ยน", "ติดนิสัย",
            "เคยชิน", "เลิกไม่ได้", "อยากเลิก", "ดื้อ", "ต่อต้าน", "ยังทำแบบเดิม",
            "นิสัย", "ย้ายระบบ", "ของเดิม", "ทำมาตลอด", "ไม่อยากเปลี่ยน",
            "procrastinate", "habit", "resist", "legacy", "migrate", "momentum",
            "inertia", "adopt", "roll out", "can't start", "cant start",
        ],
    },
]

# Triggers for models that no single heuristic owns, or that are so
# commonly relevant they deserve their own hook.
_EXTRA_TRIGGERS: dict[str, list[str]] = {
    "trade-offs": [
        "เลือกระหว่าง", "ควรเลือก", "ดีกว่ากัน", "คุ้มไหม", "คุ้มมั้ย", "แลกกับ",
        "ดีไหม", "ดีมั้ย", "หรือจะ", "เทียบกับ", "เสียโอกาส",
        "versus", " vs ", "choose between", "worth it", "opportunity cost",
        "either", "instead of",
    ],
    "second-order-thinking": [
        "แล้วไง", "ระยะยาว", "ผลตามมา", "ผลกระทบ", "อนาคต", "ปีหน้า", "ต่อไป",
        "long term", "long-term", "consequence", "then what", "downstream",
    ],
    "creative-destruction": [
        "ดิสรัป", "เทคโนโลยีใหม่", "ล้าสมัย", "ตกยุค", "มาแทนที่", "เอไอจะแทน",
        "disrupt", "obsolete", "replace us", "new technology", "innovation",
    ],
    "scarcity": [
        "เงินไม่พอ", "เวลาไม่พอ", "งบจำกัด", "จำกัด", "กดดัน", "รีบ", "ขาดแคลน",
        "ไม่พอ", "กระแสเงินสด", "เดดไลน์",
        "deadline", "tight budget", "no time", "running out", "cash flow", "scarce",
    ],
    "margin-of-safety": [
        "เงินเก็บ", "เงินสำรอง", "ทุนสำรอง", "แผนสอง", "แผนสำรอง",
        "savings", "runway", "reserve", "contingency", "plan b",
    ],
    "bottlenecks": [
        "ติดขัด", "รอ", "คิวยาว", "ค้าง", "งานกอง",
        "มีแค่", "คนเดียว", "คนไม่พอ",
        "queue", "waiting", "backlog", "blocker",
    ],
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", text)


def _fired_areas(text: str) -> dict[str, float]:
    """Area -> strength, for every heuristic with at least one keyword hit."""
    fired: dict[str, float] = {}
    for heuristic in HEURISTICS:
        hits = sum(1 for kw in heuristic["keywords"] if _normalize(kw) in text)
        if hits:
            # Diminishing returns: three hits in one bucket is not three
            # times the evidence, it is one theme stated three ways.
            fired[heuristic["area"]] = 1.0 + min(hits, 4) * 0.5
    return fired


def score_models(problem: str, bundle: Bundle) -> tuple[dict[str, float], dict[str, float]]:
    """Return (model scores, fired area strengths)."""
    text = _normalize(problem)
    scores: dict[str, float] = {slug: 0.0 for slug in bundle.models}
    fired = _fired_areas(text)

    for heuristic in HEURISTICS:
        weight = fired.get(heuristic["area"])
        if not weight:
            continue
        for slug in heuristic["models"]:
            if slug in scores:
                scores[slug] += weight

    for slug, keywords in _EXTRA_TRIGGERS.items():
        if slug not in scores:
            continue
        hits = sum(1 for kw in keywords if _normalize(kw) in text)
        if hits:
            scores[slug] += 1.0 + min(hits, 3) * 0.4

    for slug, model in bundle.models.items():
        # A user's own model is written for their domain; give it the edge
        # SKILL.md says it has earned -- but only if it already matched.
        if model.origin != "bundled" and scores[slug] > 0:
            scores[slug] += 1.5
        # Let a model's own tags vote, so user models absent from
        # HEURISTICS can still surface.
        for tag in model.tags:
            if _normalize(tag.replace("-", " ")) in text:
                scores[slug] += 0.75

    return scores, fired


def _areas_of(model: Model) -> list[str]:
    """Every heuristic area a model belongs to (several models sit in two)."""
    areas = [h["area"] for h in HEURISTICS if model.slug in h["models"]]
    return areas or [model.tags[0] if model.tags else model.slug]


def _claim_area(model: Model, fired: dict[str, float], taken: set[str]) -> str | None:
    """Which area this pick should count against.

    A model in two buckets (bias-from-incentives is in both conflict and
    people) should be credited to a bucket that actually fired and is still
    free -- otherwise it blocks an area it was never evidence for.
    """
    candidates = _areas_of(model)
    free = [a for a in candidates if a not in taken]
    if not free:
        return None
    # Prefer the free area with the weakest independent support, so the
    # strongest area stays open for a model that belongs to it alone.
    return min(free, key=lambda a: fired.get(a, 0.0))


def select_models(
    problem: str, bundle: Bundle, want: int = 3, max_models: int = 4
) -> list[Model]:
    """Pick 2-4 models, biased toward covering different areas.

    Cross-area coverage is the point of a latticework, so a second model
    from an area already covered is taken only after every area with a
    positive score has had a turn.
    """
    want = max(2, min(want, max_models))
    scores, fired = score_models(problem, bundle)
    ranked = sorted(
        bundle.models.values(),
        key=lambda m: (-scores[m.slug], m.title.lower()),
    )

    picked: list[Model] = []
    taken: set[str] = set()

    # First pass: best model from each distinct area, positive scores only.
    for model in ranked:
        if len(picked) >= want:
            break
        if scores[model.slug] <= 0:
            break
        area = _claim_area(model, fired, taken)
        if area is None:
            continue
        picked.append(model)
        taken.add(area)

    # Second pass: fill from the remaining positives regardless of area.
    if len(picked) < want:
        for model in ranked:
            if len(picked) >= want:
                break
            if scores[model.slug] <= 0 or model in picked:
                continue
            picked.append(model)

    # Nothing matched: fall back to broad, general-purpose frames rather
    # than returning an empty hand.
    if len(picked) < 2:
        for slug in ("first-principles-thinking", "inversion", "second-order-thinking",
                     "trade-offs", "bottlenecks"):
            model = bundle.get(slug)
            if model and model not in picked:
                picked.append(model)
            if len(picked) >= want:
                break

    return picked[:max_models]


def matched_areas(problem: str) -> list[str]:
    """Labels of the heuristics that fired, strongest first -- used to tell
    the user *why* these models, as SKILL.md Step 5 requires."""
    fired = _fired_areas(_normalize(problem))
    labels = {h["area"]: h["label"] for h in HEURISTICS}
    return [labels[a] for a, _ in sorted(fired.items(), key=lambda kv: -kv[1])]
