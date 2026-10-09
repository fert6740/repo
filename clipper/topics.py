"""Шаг 5, проход 1: карта стрима — темы и кандидаты в шорты."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .config import ClipperError
from .digest import Line, build_lines, estimate_tokens, render, split_windows
from .llm import LLM
from .record import Record, fmt_duration, fmt_time, parse_time, read_json, write_json

log = logging.getLogger("clipper")

KINDS = ["story", "gameplay", "discussion", "reaction", "donations_block", "pause", "tech", "other"]
TIME = {"type": "string", "pattern": r"^\d{1,2}:\d{2}:\d{2}$"}


def pass1_schema(windowed: bool) -> dict:
    topic = {
        "type": "object",
        "properties": {
            "start": TIME, "end": TIME,
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "kind": {"type": "string", "enum": KINDS},
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "why": {"type": "string"},
            "self_contained": {"type": "boolean"},
        },
        "required": ["start", "end", "title", "summary", "kind", "score", "why", "self_contained"],
        "additionalProperties": False,
    }
    short = {
        "type": "object",
        "properties": {"start": TIME, "end": TIME, "hook": {"type": "string"},
                       "score": {"type": "integer", "minimum": 0, "maximum": 10}, "why": {"type": "string"}},
        "required": ["start", "end", "hook", "score", "why"],
        "additionalProperties": False,
    }
    props = {"topics": {"type": "array", "items": topic}, "shorts": {"type": "array", "items": short}}
    req = ["topics", "shorts"]
    if windowed:
        topic["properties"]["continues"] = {"type": "boolean"}
        props["recap"] = {"type": "string"}
        req.append("recap")
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}


SYSTEM_PASS1 = """Ты — опытный монтажёр YouTube-канала с нарезками стримов. Тебе дают расшифровку записи стрима \
(или её часть) и нужно составить карту стрима: разбить его на темы и найти моменты для коротких вертикальных роликов.

Формат расшифровки: каждая строка — кусок речи стримера на 20–30 секунд, в начале время от начала записи [ч:мм:сс]. \
Пометки в угловых скобках получены автоматически:
⟨чат ×N⟩ — чат в N раз активнее обычного; ⟨смех в чате⟩ — зрители смеются; ⟨крик⟩ — стример заметно громче обычного; \
⟨донат⟩ — на экране плашка доната; ⟨тишина Nс⟩ — N секунд без речи; ⟨сцена: …⟩ — сменилась раскладка экрана; \
⟨звук вырезан Nс⟩ — Twitch заглушил звук (чужая музыка).
Распознавание речи автоматическое и может ошибаться в словах — суди по смыслу. Озвучка донатов синтезированным \
голосом тоже попадает в расшифровку как речь.

Что нужно:
1. topics — темы подряд по всему тексту, без дыр и наложений (конец одной = начало следующей). Тема — связный \
кусок, который можно выложить отдельным роликом: история, обсуждение одного вопроса, игровой эпизод, реакция на \
видео и т.п. Скучные и служебные куски тоже размечай темами нужного типа, чтобы карта была сплошной.
   - start, end — время в формате ч:мм:сс, бери из меток строк; начало темы — строка, с которой она начинается, \
конец — время начала первой строки следующей темы.
   - title — рабочее название (по-русски, коротко); summary — 2–3 предложения, о чём тема.
   - kind: story (история из жизни), gameplay (игровой эпизод), discussion (обсуждение, ответы на вопросы), \
reaction (реакция на видео/новость), donations_block (стример остановился и читает/отвечает на донаты не по теме), \
pause (перерыв, отошёл, заставка), tech (настройка звука, оборудования, тех. проблемы), other.
   - score 0–10 — насколько интересно это будет зрителю YouTube отдельным роликом (10 — точно залетит, \
6 — нормальный ролик, ниже 5 — не стоит). Учитывай реакцию чата, эмоции, законченность сюжета.
   - why — коротко, почему такая оценка.
   - self_contained — понятна ли тема без остального стрима.
   Хорошая длина темы для ролика — от 4 до 25 минут. Не дроби одну историю на мелкие куски и не склеивай разные \
истории в одну тему только ради длины.
2. shorts — самые яркие моменты на 20–60 секунд для вертикальных роликов: смешная фраза, бурная реакция, \
неожиданный поворот, сильная эмоция. start/end — по меткам времени; hook — дословная первая фраза стримера, с которой начнётся ролик \
(из расшифровки, не пересказ); score 0–10; why. \
Шорт должен быть понятен без контекста. Обычно 1–3 шорта на час стрима, только действительно яркие.

Пиши по-русски. Отвечай строго по схеме."""

WINDOW_NOTE = """

Расшифровка длинная, поэтому её дают тебе по частям (окнами с перекрытием). Дополнительно:
- если последняя тема окна продолжается за его концом, поставь ей end = время последней строки окна и continues = true \
(у остальных тем continues = false);
- если в начале окна продолжается тема, начатая раньше (о ней сказано перед расшифровкой), верни её с тем же \
названием и с её настоящим началом из предыдущей части;
- recap — краткий пересказ всего, что было на стриме до конца этого окна (вместе с пересказом прошлых частей), \
не длиннее {recap_words} слов: он будет передан при разборе следующей части."""


def user_prompt(rec: Record, text: str, part: tuple[int, int] | None = None, recap: str = "",
                open_topic: dict | None = None) -> str:
    info = rec.load_info()
    sc = rec.streamer_cfg
    head = [f"Стример: {sc.get('display_name') or info.get('uploader') or rec.streamer}",
            f"Название стрима: {info.get('title') or '—'}",
            f"Длительность записи: {fmt_time(info.get('duration') or 0)}"]
    ex = sc.get("style", {}).get("title_examples") or []
    if ex:
        head.append("Примеры удачных названий роликов этого стримера (образец стиля):\n" + "\n".join("- " + e for e in ex))
    banned = sc.get("style", {}).get("banned_topics") or []
    if banned:
        head.append("Темы, которые не брать (ставь им score 0):\n" + "\n".join("- " + b for b in banned))
    if part:
        head.append(f"Это часть {part[0]} из {part[1]}.")
    if recap:
        head.append("Пересказ предыдущих частей:\n" + recap)
    if open_topic:
        head.append(f"Незакрытая тема: «{open_topic['title']}» началась в {fmt_time(open_topic['start'])} "
                    f"и не закончена в предыдущей части ({open_topic['summary']}).")
    return "\n\n".join(head) + "\n\nРасшифровка:\n" + text


# ---------- склейка окон ----------

def overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def merge_windows(results: list[dict], bounds: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """Склейка тем из окон с перекрытием.

    Внутри перекрытия решает более позднее окно (оно видит продолжение). Тема, пересекающая начало нового окна
    (в том числе помеченная предыдущим окном как незакрытая), объединяется с первой темой нового окна.
    """
    topics: list[dict] = []
    shorts: list[dict] = []
    for wi, (res, (wa, wb)) in enumerate(zip(results, bounds)):
        new = sorted(res["topics"], key=lambda t: t["start"])
        if wi > 0 and topics:
            spanning = [p for p in topics if p["start"] < wa < p["end"] or p.get("continues")]
            topics = [p for p in topics if p["start"] < wa]
            if spanning and new:
                p = min(spanning, key=lambda x: x["start"]) if any(x["start"] < wa for x in spanning) else spanning[0]
                t = new[0]
                if t["start"] <= wa + 90 or abs(t["start"] - p["start"]) <= 60:
                    t["start"] = min(t["start"], p["start"])
                    t["merged_from"] = p.get("merged_from", []) + [p["title"]]
                    topics = [x for x in topics if x is not p]
            for p in topics:
                p["continues"] = False
                if new and p["end"] > new[0]["start"]:
                    p["end"] = new[0]["start"]
            topics = [p for p in topics if p["end"] - p["start"] > 1]
        topics.extend(new)
        for s in res.get("shorts", []):
            dup = next((x for x in shorts
                        if overlap(x, s) > 0.5 * min(x["end"] - x["start"], s["end"] - s["start"])), None)
            if dup is None:
                shorts.append(s)
            elif s["score"] > dup["score"]:
                shorts[shorts.index(dup)] = s
    topics.sort(key=lambda t: t["start"])
    shorts.sort(key=lambda s: s["start"])
    return topics, shorts


def seam_problems(topics: list[dict], bounds: list[tuple[float, float]], tol: float) -> list[str]:
    """Проверка, что тема не порвалась на границе окна."""
    probs = []
    for t in topics:
        if t.get("continues"):
            probs.append(f"тема «{t['title']}» ({fmt_time(t['start'])}) осталась незакрытой")
    for i, (_, wb) in enumerate(bounds[:-1]):
        for t in topics:
            if abs(t["end"] - wb) <= tol:
                probs.append(f"тема «{t['title']}» заканчивается у стыка окон {i + 1}/{i + 2} ({fmt_time(wb)})")
    return probs


# ---------- отбор ----------

def select(topics: list[dict], shorts: list[dict], muted: list[list[float]], cfg: dict, sc: dict) -> None:
    tc = cfg["topics"]
    lo = sc["clips"].get("long_min_minutes", tc["min_minutes"]) * 60
    hi = sc["clips"].get("long_max_minutes", tc["max_minutes"]) * 60
    for t in topics:
        m = sum(overlap(t, {"start": a, "end": b}) for a, b in muted)
        t["muted_sec"] = round(m, 1)
        reasons = []
        if t["score"] < tc["min_score"]:
            reasons.append(f"оценка ниже {tc['min_score']}")
        d = t["end"] - t["start"]
        if d < lo:
            reasons.append(f"короче {lo // 60:.0f} мин")
        if d > hi:
            reasons.append(f"длиннее {hi // 60:.0f} мин")
        if t["kind"] in tc["excluded_kinds"]:
            reasons.append(f"тип {t['kind']}")
        if m > tc["muted_overlap_sec"]:
            reasons.append("звук вырезан Twitch")
        t["selected"] = not reasons
        t["reject"] = reasons
    smin = sc["clips"].get("short_min_sec", tc["short_min_sec"])
    smax = sc["clips"].get("short_max_sec", tc["short_max_sec"])
    for s in shorts:
        m = sum(overlap(s, {"start": a, "end": b}) for a, b in muted)
        d = s["end"] - s["start"]
        reasons = []
        if m > 0:
            reasons.append("звук вырезан Twitch")
        if not (smin * 0.5 <= d <= smax * 1.5):  # точные границы шорта уточняются позже
            reasons.append(f"длина {d:.0f} с")
        s["selected"] = not reasons
        s["reject"] = reasons


def to_seconds(items: list[dict]) -> None:
    for x in items:
        x["start"], x["end"] = parse_time(x["start"]), parse_time(x["end"])
        if x["end"] < x["start"]:
            x["start"], x["end"] = x["end"], x["start"]


# ---------- проход целиком ----------

def load_lines(rec: Record) -> list[Line]:
    if not rec.transcript.exists():
        raise ClipperError("Нет transcript.json — сначала шаг transcribe.")
    segs = read_json(rec.transcript)["segments"]
    sig = pd.read_parquet(rec.signals) if rec.signals.exists() else None
    muted = read_json(rec.muted)["muted"] if rec.muted.exists() else []
    return build_lines(segs, sig, muted, rec.cfg["digest"], rec.duration())


def pass1(rec: Record, llm: LLM, model: str, step: str = "pass1") -> dict:
    cfg = rec.cfg
    dc, lc = cfg["digest"], cfg["llm"]
    lines = load_lines(rec)
    full = render(lines)
    est = estimate_tokens(full, dc["chars_per_token"])
    windows = split_windows(lines, lc["window_tokens"], lc["window_overlap_sec"], dc["chars_per_token"])
    log.info("  текст для нейросети: %d строк, %d символов, ~%d токенов; окон: %d",
             len(lines), len(full), est, len(windows))
    windowed = len(windows) > 1
    system = SYSTEM_PASS1 + (WINDOW_NOTE.format(recap_words=int(lc["recap_tokens"] / 2)) if windowed else "")
    schema = pass1_schema(windowed)
    results, bounds = [], []
    recap, open_topic = "", None
    for i, w in enumerate(windows):
        prompt = user_prompt(rec, render(w), (i + 1, len(windows)) if windowed else None, recap, open_topic)
        res = llm.call(step if not windowed else f"{step}.w{i + 1}", model, system, prompt, schema).data
        to_seconds(res["topics"])
        to_seconds(res["shorts"])
        results.append(res)
        bounds.append((w[0].start, w[-1].end))
        recap = res.get("recap", "")
        open_topic = next((t for t in res["topics"] if t.get("continues")), None)
    if windowed:
        topics, shorts = merge_windows(results, bounds)
    else:
        topics, shorts = results[0]["topics"], results[0]["shorts"]
    problems = seam_problems(topics, bounds, cfg["topics"]["seam_check_sec"]) if windowed else []
    for p in problems:
        log.warning("  проверка стыков: %s", p)
    muted = read_json(rec.muted)["muted"] if rec.muted.exists() else []
    select(topics, shorts, muted, cfg, rec.streamer_cfg)
    for i, t in enumerate(topics, 1):
        t["id"] = i
        t.pop("continues", None)
    for i, s in enumerate(shorts, 1):
        s["id"] = i
    return {"model": model, "windows": [list(b) for b in bounds], "seam_problems": problems,
            "est_tokens": est, "topics": topics, "shorts": shorts}


def run_topics(rec: Record, force: bool = False, model: str | None = None) -> None:
    if rec.topics.exists() and not force:
        log.info("topics.json уже есть, пропускаю (--force чтобы пересчитать)")
        return
    model = model or rec.cfg["llm"]["pass1_model"]
    with rec.step("topics"):
        llm = LLM(rec.cfg, rec.usage_log)
        out = pass1(rec, llm, model)
        write_json(rec.topics, out)
        sel = [t for t in out["topics"] if t["selected"]]
        log.info("  тем: %d (отобрано %d, всего %s), шортов: %d", len(out["topics"]), len(sel),
                 fmt_duration(sum(t["end"] - t["start"] for t in sel)), len(out["shorts"]))


def topics_file_for(rec: Record, model: str) -> Path:
    return rec.path("compare") / f"topics_{model}.json"
