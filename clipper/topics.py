"""Шаг 5, проход 1: карта стрима — темы и кандидаты в шорты."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .clips import group_clips
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
            "key": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "kind": {"type": "string", "enum": KINDS},
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "why": {"type": "string"},
            "self_contained": {"type": "boolean"},
        },
        "required": ["start", "end", "key", "title", "summary", "kind", "score", "why", "self_contained"],
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
⟨звук вырезан Nс⟩ — Twitch заглушил звук (чужая музыка); ⟨зрители сделали клип: N просмотров, «название»⟩ — \
зрители сами вырезали клип из этого места (самый сильный признак яркого момента, особенно для шортов); \
⟨категория на Twitch: …⟩ — стример сменил игру или категорию.
Распознавание речи автоматическое и может ошибаться в словах — суди по смыслу. Озвучка донатов синтезированным \
голосом тоже попадает в расшифровку как речь.

Что нужно:
1. topics — сплошная разметка всего текста на куски, без дыр и наложений (конец одного = начало следующего). \
Тема — то, что можно выложить отдельным роликом: история, обсуждение одного вопроса, игровой эпизод, реакция на \
видео и т.п. Скучные и служебные куски тоже размечай темами нужного типа, чтобы карта была сплошной.
   - start, end — время куска в формате ч:мм:сс, бери из меток строк; начало — строка, с которой кусок начинается, \
конец — время начала первой строки следующего куска.
   - key — короткий ярлык темы (2–4 слова). Обычно у каждого куска свой key. Но если стример позже возвращается \
к уже начатой теме (начал историю, отвлёкся на игру или донаты, потом досказал; вернулся к тому же обсуждению), \
дай новому куску тот же key, что и началу темы, — такие куски смонтируют в один ролик. Возврат может быть и через \
несколько часов. У кусков одной темы title, summary, kind, score, why и self_contained описывают тему целиком \
и совпадают. Не объединяй одним key разные темы и не считай возвратом мимолётное упоминание.
   - title — рабочее название (по-русски, коротко); summary — 2–3 предложения, о чём тема.
   - kind: story (история из жизни), gameplay (игровой эпизод), discussion (обсуждение, ответы на вопросы), \
reaction (реакция на видео/новость), donations_block (стример остановился и читает/отвечает на донаты не по теме), \
pause (перерыв, отошёл, заставка), tech (настройка звука, оборудования, тех. проблемы), other.
   - score 0–10 — насколько интересно это будет зрителю YouTube отдельным роликом (10 — точно залетит, \
6 — нормальный ролик, ниже 5 — не стоит). Учитывай реакцию чата, эмоции, законченность сюжета.
   - why — коротко, почему такая оценка.
   - self_contained — понятна ли тема без остального стрима.
   Хорошая длина темы для ролика (сумма её кусков) — от 4 до 25 минут. Не дроби одну историю на мелкие куски и не склеивай разные \
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
- перед расшифровкой перечислены темы предыдущих частей с их key: если в этом окне стример возвращается к одной \
из них, дай куску её key; новым темам давай новые key, не совпадающие с перечисленными;
- recap — краткий пересказ всего, что было на стриме до конца этого окна (вместе с пересказом прошлых частей), \
не длиннее {recap_words} слов: он будет передан при разборе следующей части."""


def user_prompt(rec: Record, text: str, part: tuple[int, int] | None = None, recap: str = "",
                open_topic: dict | None = None, prev_topics: list[dict] | None = None) -> str:
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
    if prev_topics:
        seen: dict[str, dict] = {}
        for t in prev_topics:
            if t["kind"] not in ("pause", "tech") and t.get("key"):
                seen.setdefault(t["key"], t)
        head.append("Темы предыдущих частей (key — название — где началась):\n" + "\n".join(
            f"- {k} — {t['title']} — {fmt_time(t['start'])}" for k, t in list(seen.items())[-80:]))
    if open_topic:
        head.append(f"Незакрытая тема: «{open_topic['title']}» (key: {open_topic.get('key', '—')}) началась в "
                    f"{fmt_time(open_topic['start'])} и не закончена в предыдущей части ({open_topic['summary']}).")
    return "\n\n".join(head) + "\n\nРасшифровка:\n" + text


# ---------- склейка окон ----------

def overlap(a: dict, b: dict) -> float:
    return max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def merge_windows(results: list[dict], bounds: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """Склейка кусков из окон с перекрытием.

    Новое окно отвечает за время с начала своего первого куска (оно видит продолжение); старые куски до этого
    момента остаются. Старый кусок, который заходит в новое окно или помечен незакрытым, продолжается куском
    нового окна с тем же key, а если такого нет и он явно заходит внутрь — первым куском нового окна.
    """
    topics: list[dict] = []
    shorts: list[dict] = []
    for wi, (res, (wa, wb)) in enumerate(zip(results, bounds)):
        new = sorted(res["topics"], key=lambda t: t["start"])
        if wi > 0 and topics and new:
            cov = new[0]["start"]
            keep = [x for x in topics if x["start"] < cov]
            p = max(keep, key=lambda x: x["start"]) if keep else None
            if p is not None:
                t = next((x for x in new if p.get("key") and x.get("key") == p["key"] and x["start"] <= cov + 90), None)
                if t is None and p["end"] > cov + 30:
                    t = new[0]
                if t is not None:
                    t["start"] = min(t["start"], p["start"])
                    t["merged_from"] = p.get("merged_from", []) + [p["title"]]
                    keep.remove(p)
                    old = t.get("key")
                    if p.get("key") and old and old != p["key"]:  # продолжение темы из прошлого окна — её key
                        for x in new:
                            if x.get("key") == old:
                                x["key"] = p["key"]
            for x in keep:
                x["continues"] = False
                x["end"] = min(x["end"], cov)
            topics = [x for x in keep if x["end"] - x["start"] > 1]
        elif wi > 0 and not new:
            continue
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


# ---------- темы из нескольких кусков ----------

def group_topics(pieces: list[dict], max_parts: int = 6) -> list[dict]:
    """Куски с одинаковым key — одна тема (стример вернулся к ней позже). Описание берётся у последнего куска."""
    by_key: dict[str, list[dict]] = {}
    for i, p in enumerate(sorted(pieces, key=lambda x: x["start"])):
        by_key.setdefault(p.get("key") or f"_{i}", []).append(p)
    topics = []
    for key, ps in by_key.items():
        if len(ps) > max_parts:  # скорее ошибка модели — не склеиваем
            for j, p in enumerate(ps):
                topics.append(group_topics([{**p, "key": f"{key}#{j}"}])[0])
            continue
        parts: list[list[float]] = []
        for p in ps:
            if parts and p["start"] - parts[-1][1] <= 2:
                parts[-1][1] = max(parts[-1][1], p["end"])
            else:
                parts.append([p["start"], p["end"]])
        last = ps[-1]
        kinds: dict[str, float] = {}
        for p in ps:
            kinds[p["kind"]] = kinds.get(p["kind"], 0) + p["end"] - p["start"]
        t = {k: v for k, v in last.items() if k not in ("start", "end", "continues")}
        t.update(key=key, start=parts[0][0], end=parts[-1][1], parts=parts,
                 duration=round(sum(b - a for a, b in parts), 1), kind=max(kinds, key=kinds.get),
                 score=max(p["score"] for p in ps))
        merged = [m for p in ps for m in p.get("merged_from", [])]
        if merged:
            t["merged_from"] = merged
        topics.append(t)
    topics.sort(key=lambda t: t["start"])
    return topics


def parts_of(t: dict) -> list[list[float]]:
    return t.get("parts") or [[t["start"], t["end"]]]


# ---------- отбор ----------

def select(topics: list[dict], shorts: list[dict], muted: list[list[float]], cfg: dict, sc: dict) -> None:
    tc = cfg["topics"]
    lo = sc["clips"].get("long_min_minutes", tc["min_minutes"]) * 60
    hi = sc["clips"].get("long_max_minutes", tc["max_minutes"]) * 60
    for t in topics:
        m = sum(overlap({"start": pa, "end": pb}, {"start": a, "end": b}) for pa, pb in parts_of(t) for a, b in muted)
        t["muted_sec"] = round(m, 1)
        reasons = []
        if t["score"] < tc["min_score"]:
            reasons.append(f"оценка ниже {tc['min_score']}")
        d = sum(b - a for a, b in parts_of(t))
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
    clips_path = rec.path("clips.json")
    groups = group_clips(read_json(clips_path)["clips"]) if clips_path.exists() else []
    chapters = rec.load_info().get("chapters") or []
    return build_lines(segs, sig, muted, rec.cfg["digest"], rec.duration(), groups, chapters)


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
    recap, open_topic, prev = "", None, []
    for i, w in enumerate(windows):
        prompt = user_prompt(rec, render(w), (i + 1, len(windows)) if windowed else None, recap, open_topic, prev)
        res = llm.call(step if not windowed else f"{step}.w{i + 1}", model, system, prompt, schema).data
        to_seconds(res["topics"])
        to_seconds(res["shorts"])
        results.append(res)
        bounds.append((w[0].start, w[-1].end))
        recap = res.get("recap", "")
        open_topic = next((t for t in res["topics"] if t.get("continues")), None)
        prev += res["topics"]
    if windowed:
        pieces, shorts = merge_windows(results, bounds)
    else:
        pieces, shorts = results[0]["topics"], results[0]["shorts"]
    problems = seam_problems(pieces, bounds, cfg["topics"]["seam_check_sec"]) if windowed else []
    topics = group_topics(pieces, cfg["topics"].get("max_parts", 6))
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
                 fmt_duration(sum(t.get("duration", t["end"] - t["start"]) for t in sel)), len(out["shorts"]))


def topics_file_for(rec: Record, model: str) -> Path:
    return rec.path("compare") / f"topics_{model}.json"
