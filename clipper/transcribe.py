"""Шаг 2. Распознавание речи faster-whisper с отбрасыванием выдуманного текста."""
from __future__ import annotations

import ctypes
import glob
import importlib.util
import logging
import re
import time
from pathlib import Path

import numpy as np

from .config import ClipperError
from .media import read_wav, wav_info
from .record import Record, fmt_duration, fmt_time, read_json, write_json

log = logging.getLogger("clipper")


# ---------- отбрасывание мусора ----------

def norm_text(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def filter_segments(segs: list[dict], c: dict) -> tuple[list[dict], list[dict]]:
    """Убирает выдуманные whisper отрезки. Возвращает (оставленные, отброшенные с причиной)."""
    kept, dropped = [], []
    phrases = [norm_text(p) for p in c.get("hallucination_phrases", [])]
    prev, prev_end, repeats = None, -1e9, 0
    window = c.get("repeat_window_sec", 30)
    for s in segs:
        text = s["text"].strip()
        n = norm_text(text)
        lp, nsp = s.get("avg_logprob", 0.0), s.get("no_speech_prob", 0.0)
        reason = None
        if not n:
            reason = "empty"
        elif lp < c["drop_very_low_logprob"]:
            reason = "low_confidence"
        elif lp < c["max_avg_logprob_drop"] and nsp > c["min_no_speech_prob_drop"]:
            reason = "no_speech"
        elif s.get("compression_ratio", 0.0) > c["max_compression_ratio"]:
            reason = "repetitive"
        elif any(p and p in n and len(n) <= len(p) + 25 for p in phrases):
            reason = "known_hallucination"
        if reason is None:
            if n == prev and s["start"] - prev_end <= window:
                repeats += 1
                if repeats >= c["max_repeats"]:
                    reason = "repeat"
            else:
                prev, repeats = n, 0
            prev_end = s["end"]
        if reason:
            dropped.append({**s, "reason": reason})
        else:
            kept.append(s)
    return kept, dropped


# ---------- куски аудио ----------

def quiet_point(audio: Path, t: float, radius: float = 20.0, hop: float = 0.1) -> float:
    """Самый тихий момент около t — там безопасно резать аудио на куски."""
    a = max(0.0, t - radius)
    x = read_wav(audio, a, 2 * radius)
    sr = 16000
    win = int(sr * hop)
    n = len(x) // win
    if n == 0:
        return t
    rms = np.sqrt((x[: n * win].reshape(n, win) ** 2).mean(axis=1))
    return round(a + float(np.argmin(rms)) * hop + hop / 2, 2)


def plan_chunks(audio: Path, total: float, chunk_sec: float) -> list[tuple[float, float]]:
    bounds = [0.0]
    t = chunk_sec
    while t < total - chunk_sec * 0.25:
        bounds.append(quiet_point(audio, t))
        t += chunk_sec
    bounds.append(total)
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


# ---------- модель ----------

def preload_cuda_libs() -> None:
    """Подгружает cuBLAS/cuDNN из pip-пакетов nvidia-*, чтобы не трогать системные библиотеки."""
    libs = []
    for pkg in ("nvidia.cuda_runtime", "nvidia.cublas", "nvidia.cudnn"):
        try:
            spec = importlib.util.find_spec(pkg)
        except ModuleNotFoundError:
            spec = None
        if spec and spec.submodule_search_locations:
            for loc in spec.submodule_search_locations:
                libs += sorted(glob.glob(str(Path(loc) / "lib" / "*.so*")))
    pending = libs
    for _ in range(4):  # зависимости друг от друга — несколько проходов
        failed = []
        for so in pending:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(so)
        if not failed or len(failed) == len(pending):
            break
        pending = failed


def is_oom(e: Exception) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda_error_out_of_memory" in s or "cudaerrormemoryallocation" in s


def load_model(c: dict):
    if c["device"] == "cuda":
        preload_cuda_libs()
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise ClipperError("Не установлен faster-whisper. Запустите ./install.sh") from e
    log.info("  загружаю модель %s (%s, %s); при первом запуске она скачивается (~1.6 ГБ)",
             c["model"], c["device"], c["compute_type"])
    try:
        return WhisperModel(c["model"], device=c["device"], compute_type=c["compute_type"])
    except Exception as e:  # noqa: BLE001 — превращаем в понятное сообщение
        if is_oom(e):
            raise ClipperError(
                "Не хватает видеопамяти для модели распознавания. Закройте программы, занимающие видеокарту "
                "(игры, браузер с видео), или поставьте в config.toml: compute_type = \"int8\", "
                "или model = \"medium\", или device = \"cpu\" (очень медленно).") from e
        if "libcublas" in str(e) or "libcudnn" in str(e) or "cuda" in str(e).lower():
            raise ClipperError(
                f"Видеокарта недоступна для распознавания: {e}\n"
                "Проверьте, что установлен драйвер NVIDIA (команда nvidia-smi должна работать). "
                "Временно можно поставить device = \"cpu\" в config.toml (очень медленно).") from e
        raise


def segments_to_dicts(segs, offset: float) -> list[dict]:
    out = []
    for s in segs:
        words = [{"start": round(w.start + offset, 2), "end": round(w.end + offset, 2),
                  "word": w.word, "prob": round(w.probability, 3)} for w in (s.words or [])]
        out.append({"start": round(s.start + offset, 2), "end": round(s.end + offset, 2), "text": s.text.strip(),
                    "words": words, "avg_logprob": round(s.avg_logprob, 3),
                    "no_speech_prob": round(s.no_speech_prob, 3),
                    "compression_ratio": round(s.compression_ratio, 2)})
    return out


class Transcriber:
    def __init__(self, c: dict):
        self.c = c
        self.model = load_model(c)
        self.batch = int(c.get("batch_size") or 0)
        self._batched = None

    def _run(self, audio: np.ndarray):
        c = self.c
        kw = dict(language=c["language"], beam_size=c["beam_size"], word_timestamps=True,
                  vad_filter=c["vad"], condition_on_previous_text=c["condition_on_previous_text"])
        if self.batch > 0:
            from faster_whisper import BatchedInferencePipeline
            if self._batched is None:
                self._batched = BatchedInferencePipeline(model=self.model)
            kw.pop("condition_on_previous_text")
            segs, _ = self._batched.transcribe(audio, batch_size=self.batch, **kw)
        else:
            segs, _ = self.model.transcribe(audio, **kw)
        return list(segs)

    def transcribe(self, audio: np.ndarray):
        while True:
            try:
                return self._run(audio)
            except Exception as e:  # noqa: BLE001
                if not is_oom(e):
                    raise
                if self.batch > 1:
                    self.batch //= 2
                    log.warning("  не хватает видеопамяти — уменьшаю batch_size до %d", self.batch)
                elif self.batch == 1:
                    self.batch = 0
                    log.warning("  не хватает видеопамяти — перехожу на обычный (непакетный) режим")
                else:
                    raise ClipperError(
                        "Не хватает видеопамяти даже в самом экономном режиме. Закройте программы, "
                        "занимающие видеокарту, или поставьте compute_type = \"int8\" в config.toml.") from e


# ---------- шаг целиком ----------

def run_transcribe(rec: Record, force: bool = False) -> None:
    if rec.transcript.exists() and not force:
        log.info("transcript.json уже есть, пропускаю (--force чтобы пересчитать)")
        return
    if not rec.audio.exists():
        raise ClipperError("Нет audio.wav — сначала шаг download.")
    c = rec.cfg["transcribe"]
    parts = rec.path("transcript_parts")
    if force and parts.exists():
        for p in parts.glob("*.json"):
            p.unlink()
    parts.mkdir(exist_ok=True)
    sr, n = wav_info(rec.audio)
    total = n / sr

    with rec.step("transcribe"):
        plan_path = parts / "chunks.json"
        if plan_path.exists():
            chunks = [tuple(x) for x in read_json(plan_path)]
        else:
            chunks = plan_chunks(rec.audio, total, c["chunk_minutes"] * 60)
            write_json(plan_path, chunks)
        todo = [i for i in range(len(chunks)) if not (parts / f"part_{i:03d}.json").exists()]
        if len(todo) < len(chunks):
            log.info("  продолжаю: готово %d из %d кусков", len(chunks) - len(todo), len(chunks))
        tr = Transcriber(c) if todo else None
        t0 = time.time()
        done_audio = 0.0
        for i in todo:
            a, b = chunks[i]
            audio = read_wav(rec.audio, a, b - a)
            segs = segments_to_dicts(tr.transcribe(audio), a)
            segs = [s for s in segs if s["start"] < b + 1.0]
            write_json(parts / f"part_{i:03d}.json", {"range": [a, b], "segments": segs}, indent=None)
            done_audio += b - a
            speed = done_audio / max(time.time() - t0, 1e-6)
            left = sum(chunks[j][1] - chunks[j][0] for j in todo if j > i) / speed
            log.info("  распознано до %s из %s (скорость ×%.1f, осталось ~%s)",
                     fmt_time(b), fmt_time(total), speed, fmt_duration(left))

        segs: list[dict] = []
        for i in range(len(chunks)):
            segs += read_json(parts / f"part_{i:03d}.json")["segments"]
        segs.sort(key=lambda s: s["start"])
        kept, dropped = filter_segments(segs, c)
        write_json(rec.path("transcript_dropped.json"), dropped)
        write_json(rec.transcript, {"model": c["model"], "language": c["language"], "duration": total,
                                    "segments": kept}, indent=None)
        words = sum(len(s["words"]) for s in kept)
        log.info("  отрезков: %d, слов: %d, отброшено как мусор: %d", len(kept), words, len(dropped))
