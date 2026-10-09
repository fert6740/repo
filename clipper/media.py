"""Обёртки над ffmpeg/ffprobe и чтение звука."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .config import ClipperError

log = logging.getLogger("clipper")


def require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise ClipperError(f"Не найдена программа {binary}. Установите её: sudo pacman -S {binary}")
    return path


def run(cmd: Sequence[str], check: bool = True, quiet: bool = False) -> subprocess.CompletedProcess:
    if not quiet:
        log.debug("$ %s", " ".join(map(str, cmd)))
    p = subprocess.run(list(map(str, cmd)), capture_output=True, text=True)
    if check and p.returncode != 0:
        tail = "\n".join((p.stderr or p.stdout or "").strip().splitlines()[-15:])
        raise subprocess.CalledProcessError(p.returncode, cmd, p.stdout, tail)
    return p


def probe(path: Path) -> dict:
    require("ffprobe")
    p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path], quiet=True)
    return json.loads(p.stdout)


def probe_duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def run_ffmpeg(args: Sequence[str], total: float | None = None, label: str = "ffmpeg",
               report_every: float = 60.0) -> None:
    """ffmpeg с отчётом о ходе работы раз в минуту (по длительности входа)."""
    require("ffmpeg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
           "-progress", "pipe:1", "-nostats", *map(str, args)]
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    last = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        if total and line.startswith("out_time_us=") and time.time() - last >= report_every:
            try:
                done = int(line.split("=", 1)[1]) / 1e6
            except ValueError:
                continue
            log.info("  %s: %d%%", label, min(100, int(100 * done / total)))
            last = time.time()
    err = proc.stderr.read() if proc.stderr else ""
    if proc.wait() != 0:
        tail = "\n".join(err.strip().splitlines()[-15:])
        raise subprocess.CalledProcessError(proc.returncode, cmd, "", tail)


def ffmpeg_try(variants: list[tuple[str, list[str]]], out: Path, total: float | None = None) -> str:
    """Пробует варианты команды ffmpeg по очереди (например, nvenc → libx264). Возвращает имя сработавшего."""
    errors = []
    for name, cmd in variants:
        tmp = out.with_name(out.stem + ".part" + out.suffix)
        try:
            run_ffmpeg([*cmd, tmp], total=total, label=out.name)
            tmp.replace(out)
            return name
        except subprocess.CalledProcessError as e:
            errors.append(f"[{name}] {e.stderr}")
            log.warning("ffmpeg (%s) не сработал, пробую следующий вариант", name)
            tmp.unlink(missing_ok=True)
    raise ClipperError("ffmpeg не смог создать " + out.name + ":\n" + "\n".join(errors))


def wav_info(path: Path) -> tuple[int, int]:
    """(частота, число отсчётов) для 16-битного моно wav."""
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ClipperError(f"{path.name}: ожидался 16-битный моно wav")
        return w.getframerate(), w.getnframes()


def read_wav(path: Path, start: float = 0.0, dur: float | None = None) -> np.ndarray:
    """Кусок 16-битного моно wav как float32 в диапазоне −1..1."""
    with wave.open(str(path), "rb") as w:
        sr, n = w.getframerate(), w.getnframes()
        a = min(int(start * sr), n)
        b = n if dur is None else min(int((start + dur) * sr), n)
        w.setpos(a)
        raw = w.readframes(b - a)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def iter_wav_blocks(path: Path, block_sec: float = 600.0) -> Iterator[tuple[float, np.ndarray]]:
    """Звук блоками (начало блока в секундах, отсчёты), чтобы не держать 6 часов в памяти."""
    sr, n = wav_info(path)
    step = int(block_sec * sr)
    with wave.open(str(path), "rb") as w:
        pos = 0
        while pos < n:
            raw = w.readframes(min(step, n - pos))
            yield pos / sr, np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            pos += step


def gray_frames(video: Path, start: float, dur: float, fps: float = 1.0, w: int = 64, h: int = 36) -> np.ndarray:
    """Маленькие серые кадры отрезка видео: массив (кадры, h, w)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{dur:.2f}",
           "-i", str(video), "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        return np.zeros((0, h, w), dtype=np.uint8)
    data = np.frombuffer(p.stdout, dtype=np.uint8)
    return data[: len(data) // (w * h) * (w * h)].reshape(-1, h, w)
