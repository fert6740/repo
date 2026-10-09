"""Рабочая папка одной записи: пути к файлам, журнал, учёт времени шагов."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ClipperError, Config

log = logging.getLogger("clipper")

TWITCH_VOD_RE = re.compile(r"twitch\.tv/(?:[^/]+/)?(?:videos?|v)/(\d+)", re.I)


def parse_vod_id(s: str) -> str | None:
    m = TWITCH_VOD_RE.search(s)
    if m:
        return m.group(1)
    s = s.strip().lstrip("v")
    return s if s.isdigit() else None


def write_json(path: Path, data: Any, indent: int | None = 1) -> None:
    """Запись через временный файл: недописанный файл никогда не считается готовым."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Record:
    def __init__(self, cfg: Config, streamer: str, vod_id: str):
        self.cfg = cfg
        self.streamer = streamer
        self.vod_id = vod_id
        self.dir = cfg.data_dir / streamer / vod_id

    # --- файлы ---
    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def source(self) -> Path: return self.dir / "source.mp4"
    @property
    def proxy(self) -> Path: return self.dir / "proxy.mp4"
    @property
    def audio(self) -> Path: return self.dir / "audio.wav"
    @property
    def chat(self) -> Path: return self.dir / "chat.json"
    @property
    def info(self) -> Path: return self.dir / "info.json"
    @property
    def muted(self) -> Path: return self.dir / "muted.json"
    @property
    def transcript(self) -> Path: return self.dir / "transcript.json"
    @property
    def signals(self) -> Path: return self.dir / "signals.parquet"
    @property
    def topics(self) -> Path: return self.dir / "topics.json"
    @property
    def usage_log(self) -> Path: return self.dir / "llm_usage.jsonl"
    @property
    def timings(self) -> Path: return self.dir / "timings.json"
    @property
    def url(self) -> str: return f"https://www.twitch.tv/videos/{self.vod_id}"

    @property
    def streamer_cfg(self) -> dict:
        return self.cfg.streamer(self.streamer)

    def load_info(self) -> dict:
        return read_json(self.info) if self.info.exists() else {}

    def duration(self) -> float:
        info = self.load_info()
        if info.get("duration"):
            return float(info["duration"])
        from .media import probe_duration
        for p in (self.audio, self.source):
            if p.exists():
                return probe_duration(p)
        raise ClipperError("Неизвестна длительность записи: нет ни info.json, ни аудио.")

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cfg.ensure_streamer(self.streamer)

    # --- учёт времени ---
    @contextmanager
    def step(self, name: str):
        """Отмечает начало и конец шага, пишет время работы в timings.json."""
        t0 = time.time()
        log.info("▶ %s", name)
        yield
        dt = time.time() - t0
        data = read_json(self.timings) if self.timings.exists() else {}
        data[name] = {"seconds": round(dt, 1), "finished": datetime.now().isoformat(timespec="seconds")}
        write_json(self.timings, data)
        log.info("✔ %s — %s", name, fmt_duration(dt))

    def attach_log(self) -> None:
        """Журнал записи: всё, что печатается, дублируется в log.txt её папки."""
        self.dir.mkdir(parents=True, exist_ok=True)
        target = str(self.dir / "log.txt")
        root = logging.getLogger("clipper")
        if any(getattr(h, "baseFilename", None) == target for h in root.handlers):
            return
        h = logging.FileHandler(target, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        root.addHandler(h)


def find_record(cfg: Config, ref: str, streamer: str | None = None) -> Record:
    """Запись по id, ссылке или пути к папке."""
    p = Path(ref).expanduser()
    if p.is_dir() and ("/" in ref or (p / "info.json").exists()):
        p = p.resolve()
        return Record(cfg, p.parent.name, p.name)
    vod = parse_vod_id(ref)
    if not vod:
        raise ClipperError(f"Не понял, какая это запись: {ref!r}. Нужна ссылка на Twitch, номер записи или путь к папке.")
    if streamer:
        return Record(cfg, streamer, vod)
    found = sorted(cfg.data_dir.glob(f"*/{vod}")) if cfg.data_dir.exists() else []
    found = [f for f in found if f.is_dir()]
    if not found:
        raise ClipperError(f"Запись {vod} не найдена в {cfg.data_dir}. Сначала: clipper run <ссылка> --streamer <имя>")
    if len(found) > 1:
        names = ", ".join(f.parent.name for f in found)
        raise ClipperError(f"Запись {vod} есть у нескольких стримеров ({names}) — укажите --streamer.")
    return Record(cfg, found[0].parent.name, vod)


def fmt_time(sec: float) -> str:
    sec = int(round(max(sec, 0)))
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def parse_time(s: str | int | float) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    parts = [float(x) for x in str(s).strip().split(":")]
    v = 0.0
    for x in parts:
        v = v * 60 + x
    return v


def fmt_duration(sec: float) -> str:
    sec = int(round(sec))
    if sec < 60:
        return f"{sec} с"
    if sec < 3600:
        return f"{sec // 60} мин {sec % 60} с"
    return f"{sec // 3600} ч {sec % 3600 // 60} мин"
