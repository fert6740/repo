"""Проверка, какого размера запрос проходит через claude -p (от этого зависит размер окна прохода 1)."""
from __future__ import annotations

import logging
import random

from .config import ClipperError, Config
from .llm import LLM
from .record import fmt_time

log = logging.getLogger("clipper")

WORDS = ("короче вот значит смотрите ребята сейчас мы будем играть дальше это вообще было очень смешно когда "
         "я пошёл в магазин там стоит мужик говорит слушай брат а я ему такой ну давай и мы короче поехали "
         "потом оказалось что всё сломалось чат пишет что я не прав но я считаю что прав кто донатил спасибо "
         "большое за поддержку сегодня у нас длинный стрим будет много интересного вчера я смотрел видео где "
         "человек построил дом из бутылок это реально круто но непонятно зачем игра вообще сложная босс опять "
         "убил меня третий раз подряд надо качаться сначала").split()

SCHEMA = {"type": "object",
          "properties": {"last_marker": {"type": "integer"}, "marker_count": {"type": "integer"}},
          "required": ["last_marker", "marker_count"], "additionalProperties": False}
SYSTEM = "Ты проверяешь длинный текст. Отвечай строго по схеме."


def num(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def filler(chars: int) -> tuple[str, int]:
    """Псевдо-расшифровка нужной длины с десятком меток «[метка N]» по всему тексту."""
    rnd = random.Random(7)
    every = max(3, chars // 400 // 10)
    lines, size, n, t = [], 0, 0, 0
    while size < chars:
        i = len(lines)
        line = f"[{fmt_time(t)}] " + " ".join(rnd.choice(WORDS) for _ in range(rnd.randint(40, 70)))
        if i % every == every // 2:
            n += 1
            line += f" [метка {n}]"
        lines.append(line)
        size += len(line) + 1
        t += 25
    return "\n".join(lines), n


def run_probe(cfg: Config, model: str, sizes: list[int], models_info: list[str]) -> None:
    usage_log = cfg.data_dir / "llm_probe.jsonl"
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(cfg, usage_log)
    cpt = cfg["digest"]["chars_per_token"]
    tiny = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    print("Окно контекста моделей (по ответу claude -p):")
    overhead = {}
    for m in dict.fromkeys([*models_info, model]):
        r = llm.call("probe.tiny", m, SYSTEM, "Ответь ok = true.", tiny)
        overhead[m] = r.usage.input_tokens
        print(f"  {m:<7} → {r.usage.model}: окно {num(r.usage.context_window)} токенов, "
              f"накладные расходы на запрос {num(r.usage.input_tokens)} токенов")
    print(f"\nПроверка длинных запросов моделью {model}:")
    best = 0
    for tok in sizes:
        text, n = filler(int(tok * cpt))
        prompt = ("Ниже длинный текст. В нём встречаются метки вида [метка N]. Найди номер последней метки "
                  "и посчитай, сколько всего меток.\n\n" + text)
        try:
            r = llm.call(f"probe.{tok}", model, SYSTEM, prompt, SCHEMA)
        except ClipperError as e:
            print(f"  ~{num(tok)} токенов: НЕ ПРОШЁЛ — {e}")
            break
        u = r.usage
        ok = r.data.get("last_marker") == n
        ratio = len(prompt) / max(u.input_tokens - overhead[model], 1)
        print(f"  ~{num(tok)} токенов: прошёл; на деле {num(u.input_tokens)} токенов, {ratio:.2f} символа на токен, "
              f"{u.seconds:.0f} с; последняя метка {'найдена верно' if ok else 'НЕ найдена'} "
              f"({r.data.get('last_marker')} из {n})")
        if ok:
            best = u.input_tokens
    if best:
        print(f"\nЗапросы проходят как минимум до {num(best)} токенов — window_tokens в config.toml можно ставить "
              f"не больше этого (с запасом на инструкции, ~{num(int(best * 0.9 // 1000 * 1000))}).")
    print(f"Журнал: {usage_log}")
