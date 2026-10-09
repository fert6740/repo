"""Расход токенов и времени по записи, пересчёт в цены API."""
from __future__ import annotations

import json
from collections import defaultdict

from .config import Config
from .record import Record, fmt_duration, read_json


def model_family(name: str) -> str:
    n = (name or "").lower()
    for fam in ("opus", "sonnet", "haiku"):
        if fam in n:
            return fam
    return n


def api_cost(cfg: Config, model: str, inp: int, out: int) -> float:
    p = cfg["prices"]
    fam = model_family(model)
    if fam not in p:
        return 0.0
    pin, pout = p[fam]
    if fam == "haiku" and inp > p.get("long_threshold", 100000) and "haiku_long" in p:
        pin, pout = p["haiku_long"]
    return inp / 1e6 * pin + out / 1e6 * pout


def read_usage(rec: Record) -> list[dict]:
    if not rec.usage_log.exists():
        return []
    rows = []
    with open(rec.usage_log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def summarize(cfg: Config, rows: list[dict], step_prefix: str | None = None) -> dict:
    """Итог по строкам журнала: {вход, выход, секунды, $ по ценам API, запросов, ошибок}."""
    s = {"input": 0, "output": 0, "seconds": 0.0, "usd": 0.0, "calls": 0, "errors": 0, "chars": 0}
    for r in rows:
        if step_prefix and not r["step"].startswith(step_prefix):
            continue
        if not r.get("ok"):
            s["errors"] += 1
            continue
        s["calls"] += 1
        s["input"] += r.get("input_tokens", 0)
        s["output"] += r.get("output_tokens", 0)
        s["seconds"] += r.get("seconds", 0)
        s["chars"] += r.get("prompt_chars", 0)
        s["usd"] += api_cost(cfg, r.get("model") or r.get("model_alias", ""), r.get("input_tokens", 0),
                             r.get("output_tokens", 0))
    return s


def report(rec: Record) -> str:
    cfg = rec.cfg
    rows = read_usage(rec)
    lines = [f"Запись {rec.vod_id} ({rec.streamer})", ""]
    if rows:
        groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in rows:
            step = r["step"].split(".w")[0]
            groups[(step, model_family(r.get("model") or r.get("model_alias", "")))].append(r)
        lines.append(f"{'Шаг':<22}{'Модель':<9}{'Запр.':>6}{'Вход':>11}{'Выход':>9}{'Время':>12}{'API $':>8}")
        tot = summarize(cfg, rows)
        for (step, fam), rs in sorted(groups.items()):
            s = summarize(cfg, rs)
            err = f"  (ошибок: {s['errors']})" if s["errors"] else ""
            lines.append(f"{step:<22}{fam:<9}{s['calls']:>6}{s['input']:>11,}{s['output']:>9,}"
                         f"{fmt_duration(s['seconds']):>12}{s['usd']:>8.2f}{err}".replace(",", " "))
        lines.append(f"{'Итого':<31}{tot['calls']:>6}{tot['input']:>11,}{tot['output']:>9,}"
                     f"{fmt_duration(tot['seconds']):>12}{tot['usd']:>8.2f}".replace(",", " "))
        big = [r for r in rows if r.get("ok") and r.get("input_tokens", 0) >= 20000]
        if big:  # на маленьких запросах отношение искажают накладные расходы
            ratio = sum(r["prompt_chars"] for r in big) / sum(r["input_tokens"] for r in big)
            lines.append(f"\nФактически символов на токен (вход): {ratio:.2f} "
                         f"(в настройках digest.chars_per_token = {cfg['digest']['chars_per_token']})")
        lines.append("Цены API — только для сравнения; по подписке тратятся лимиты, а не деньги.")
    else:
        lines.append("Обращений к нейросети ещё не было.")
    if rec.timings.exists():
        t = read_json(rec.timings)
        lines += ["", "Время шагов:"]
        for k, v in t.items():
            lines.append(f"  {k:<24}{fmt_duration(v['seconds']):>14}")
        lines.append(f"  {'всего':<24}{fmt_duration(sum(v['seconds'] for v in t.values())):>14}")
    return "\n".join(lines)
