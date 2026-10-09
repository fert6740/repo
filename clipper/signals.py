"""Шаг 3. Сигналы раз в секунду: громкость, речь, чат (видеосигналы — на этапе 3)."""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from .config import ClipperError
from .media import iter_wav_blocks, wav_info
from .record import Record, read_json

log = logging.getLogger("clipper")


def loudness_db(rec: Record) -> np.ndarray:
    """Среднеквадратичная громкость по секундам, дБ относительно полной шкалы."""
    sr, _ = wav_info(rec.audio)
    out = []
    for _, x in iter_wav_blocks(rec.audio, block_sec=600.0):
        n = int(np.ceil(len(x) / sr))
        pad = np.zeros(n * sr, dtype=np.float32)
        pad[: len(x)] = x
        rms = np.sqrt((pad.reshape(n, sr) ** 2).mean(axis=1))
        out.append(20 * np.log10(rms + 1e-6))
    return np.concatenate(out) if out else np.zeros(0)


def speech_fraction(segments: list[dict], n: int) -> np.ndarray:
    """Доля каждой секунды, занятая словами."""
    cov = np.zeros(n * 10, dtype=bool)  # шаг 0,1 с
    for s in segments:
        for w in s.get("words") or []:
            a, b = int(w["start"] * 10), int(np.ceil(w["end"] * 10))
            cov[max(a, 0): min(b, len(cov))] = True
    return cov.reshape(n, 10).mean(axis=1)


def laugh_regex(patterns: list[str]) -> re.Pattern:
    return re.compile("|".join(re.escape(p.lower()) for p in patterns if p) or r"(?!x)x")


def chat_series(messages: list[dict], n: int, delay: float, window: int, laugh: re.Pattern):
    """Сообщений за окно (центрированное, после сдвига на задержку чата) и доля «смеховых»."""
    per_sec = np.zeros(n)
    laugh_sec = np.zeros(n)
    for m in messages:
        t = int(m["time"] - delay)
        if 0 <= t < n:
            per_sec[t] += 1
            if laugh.search(m["text"].lower()):
                laugh_sec[t] += 1
    k = np.ones(window)
    count = np.convolve(per_sec, k, mode="same")
    lcount = np.convolve(laugh_sec, k, mode="same")
    frac = np.divide(lcount, count, out=np.zeros(n), where=count > 0)
    return per_sec, count, lcount, frac


def deviation(x: pd.Series, window: int, floor: float) -> tuple[pd.Series, pd.Series]:
    """Отклонение от обычного уровня: (x − скользящая медиана) / разброс (MAD), разброс не меньше floor."""
    minp = max(1, window // 10)
    med = x.rolling(window, center=True, min_periods=minp).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=minp).median()
    scale = (1.4826 * mad).clip(lower=floor)
    return med, (x - med) / scale


def compute_signals(rec: Record) -> pd.DataFrame:
    if not rec.audio.exists():
        raise ClipperError("Нет audio.wav — сначала шаг download.")
    if not rec.transcript.exists():
        raise ClipperError("Нет transcript.json — сначала шаг transcribe.")
    c = rec.cfg["signals"]
    sc = rec.streamer_cfg
    loud = loudness_db(rec)
    n = len(loud)
    segs = read_json(rec.transcript)["segments"]
    msgs = read_json(rec.chat)["messages"] if rec.chat.exists() else []
    if not rec.chat.exists():
        log.warning("  нет chat.json — сигналы чата будут нулевые")
    per_sec, chat, lcount, lfrac = chat_series(msgs, n, sc["chat"]["delay_sec"], c["chat_window_sec"],
                                               laugh_regex(sc["chat"]["laugh_patterns"]))
    df = pd.DataFrame({
        "t": np.arange(n, dtype=np.int32),
        "loud_db": loud.astype(np.float32),
        "speech": speech_fraction(segs, n).astype(np.float32),
        "chat_1s": per_sec.astype(np.int16),
        "chat": chat.astype(np.float32),
        "laugh": lcount.astype(np.float32),
        "laugh_frac": lfrac.astype(np.float32),
    })
    muted = np.zeros(n, dtype=bool)
    if rec.muted.exists():
        for a, b in read_json(rec.muted)["muted"]:
            muted[int(a): int(np.ceil(b))] = True
    df["muted"] = muted
    win = c["baseline_window_sec"]
    floors = c["scale_floor"]
    for col, fl in (("loud_db", floors["loud_db"]), ("speech", floors["speech"]),
                    ("chat", floors["chat"]), ("laugh_frac", floors["laugh"])):
        med, z = deviation(df[col].astype(np.float64), win, fl)
        df[col + "_base"] = med.astype(np.float32)
        df[col + "_z"] = z.astype(np.float32)
    df["scene"] = ""  # заполняется на этапе 3
    return df


def run_signals(rec: Record, force: bool = False) -> None:
    if rec.signals.exists() and not force:
        log.info("signals.parquet уже есть, пропускаю (--force чтобы пересчитать)")
        return
    with rec.step("signals"):
        df = compute_signals(rec)
        tmp = rec.signals.with_name("signals.parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(rec.signals)
        log.info("  %d секунд, речь в %.0f%% времени, сообщений чата: %d",
                 len(df), 100 * (df["speech"] > 0.3).mean(), int(df["chat_1s"].sum()))
