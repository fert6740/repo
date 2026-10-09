"""Сжатый текст стрима для нейросети: куски по 20–30 с, время один раз на кусок, пометки из сигналов."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .record import fmt_time


@dataclass
class Line:
    start: float
    end: float
    text: str

    def render(self) -> str:
        return f"[{fmt_time(self.start)}] {self.text}"


def build_chunks(segments: list[dict], cmin: float, cmax: float, gap_break: float) -> list[dict]:
    chunks: list[dict] = []
    cur = None
    for s in segments:
        if cur is not None:
            dur = cur["end"] - cur["start"]
            if dur >= cmin or s["end"] - cur["start"] > cmax or s["start"] - cur["end"] >= gap_break:
                chunks.append(cur)
                cur = None
        if cur is None:
            cur = {"start": s["start"], "end": s["end"], "texts": [s["text"].strip()]}
        else:
            cur["end"] = max(cur["end"], s["end"])
            cur["texts"].append(s["text"].strip())
    if cur is not None:
        chunks.append(cur)
    for ch in chunks:
        ch["text"] = " ".join(t for t in ch.pop("texts") if t)
    return chunks


def markers(sig: pd.DataFrame | None, a: float, b: float, c: dict) -> list[str]:
    if sig is None or len(sig) == 0:
        return []
    i, j = max(int(a), 0), min(int(np.ceil(b)) + 1, len(sig))
    w = sig.iloc[i:j]
    if len(w) == 0:
        return []
    out = []
    base = np.maximum(w["chat_base"].to_numpy(), 1.0)
    chat = w["chat"].to_numpy()
    ratio = chat / base
    k = int(np.argmax(ratio))
    if ratio[k] >= c["chat_ratio_mark"] and chat[k] >= 3:
        out.append(f"⟨чат ×{int(round(ratio[k]))}⟩")
    if ((w["laugh_frac"] >= c["laugh_fraction_mark"]) & (w["laugh"] >= c["laugh_count_mark"])).any():
        out.append("⟨смех в чате⟩")
    if ((w["loud_db_z"] >= c["shout_z"]) & (w["speech"] >= 0.5)).any():
        out.append("⟨крик⟩")
    if "alert" in w and (w["alert"] > 0).any():
        out.append("⟨донат⟩")
    return out


def clip_mark(g: dict) -> str:
    who = f"{g['count']} клипа" if 2 <= g["count"] <= 4 else (f"{g['count']} клипов" if g["count"] > 4 else "клип")
    title = f", «{g['title'][:60]}»" if g.get("title") else ""
    return f"⟨зрители сделали {who}: {g['views']} просмотров{title}⟩"


def build_lines(segments: list[dict], sig: pd.DataFrame | None, muted: list[list[float]], c: dict,
                duration: float | None = None, clip_groups: list[dict] | None = None,
                chapters: list[dict] | None = None) -> list[Line]:
    chunks = build_chunks(segments, c["chunk_min_sec"], c["chunk_max_sec"], c["silence_mark_sec"])
    extra: dict[int, list[str]] = {}
    for g in clip_groups or []:  # клип — к куску речи, с которым он больше всего пересекается
        best = max(range(len(chunks)), default=None,
                   key=lambda i: min(chunks[i]["end"], g["end"]) - max(chunks[i]["start"], g["start"]))
        if best is not None and min(chunks[best]["end"], g["end"]) > max(chunks[best]["start"], g["start"]) - 30:
            extra.setdefault(best, []).append(clip_mark(g))
    events: list[Line] = []
    for i, ch in enumerate(chunks):
        marks = markers(sig, ch["start"], ch["end"], c) + extra.get(i, [])
        events.append(Line(ch["start"], ch["end"], " ".join([ch["text"], *marks]).strip()))
    if chapters and len(chapters) > 1:
        for ch in chapters:
            events.append(Line(float(ch["start"]), float(ch["start"]), f"⟨категория на Twitch: {ch['title']}⟩"))
    for a, b in muted:
        events.append(Line(a, b, f"⟨звук вырезан {int(round(b - a))}с⟩"))
    # тишина между кусками речи (если она не объясняется вырезанным звуком)
    speech_spans = sorted((ch["start"], ch["end"]) for ch in chunks)
    prev_end = 0.0
    for a, b in speech_spans + ([(duration, duration)] if duration else []):
        gap = a - prev_end
        if gap >= c["silence_mark_sec"]:
            mut = sum(max(0.0, min(b2, a) - max(a2, prev_end)) for a2, b2 in muted)
            if gap - mut >= c["silence_mark_sec"]:
                events.append(Line(prev_end, a, f"⟨тишина {int(round(gap))}с⟩"))
        prev_end = max(prev_end, b)
    events.sort(key=lambda l: (l.start, l.end))
    return events


def estimate_tokens(text: str, chars_per_token: float) -> int:
    return int(len(text) / chars_per_token) + 1


def render(lines: list[Line]) -> str:
    return "\n".join(l.render() for l in lines)


def split_windows(lines: list[Line], max_tokens: int, overlap_sec: float, chars_per_token: float) -> list[list[Line]]:
    """Окна не больше max_tokens; каждое следующее начинается на overlap_sec раньше конца предыдущего."""
    total = estimate_tokens(render(lines), chars_per_token)
    if total <= max_tokens or not lines:
        return [lines]
    windows = []
    i = 0
    while i < len(lines):
        size, j = 0, i
        while j < len(lines):
            t = estimate_tokens(lines[j].render(), chars_per_token) + 1
            if size + t > max_tokens and j > i:
                break
            size += t
            j += 1
        windows.append(lines[i:j])
        if j >= len(lines):
            break
        end_t = lines[j - 1].end
        k = j
        while k > i + 1 and lines[k - 1].start > end_t - overlap_sec:
            k -= 1
        i = max(k, i + 1)
    return windows
