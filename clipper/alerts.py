"""Плашки донатов: поиск картинок-образцов стримера в кадре (streamers/<имя>/alerts/*.png)."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

from .config import ClipperError
from .media import probe
from .record import Record, fmt_duration, write_json

log = logging.getLogger("clipper")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def alert_dir(rec: Record) -> Path:
    return rec.cfg.streamer_dir(rec.streamer) / "alerts"


def load_templates(folder: Path, scale: float) -> list[tuple[str, np.ndarray]]:
    """Образцы плашек в сером, уменьшенные под разрешение proxy.mp4."""
    import cv2
    out = []
    for p in sorted(folder.iterdir()) if folder.exists() else []:
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning("  не удалось прочитать образец %s", p.name)
            continue
        h, w = img.shape
        img = cv2.resize(img, (max(8, round(w * scale)), max(8, round(h * scale))), interpolation=cv2.INTER_AREA)
        out.append((p.name, img))
    return out


def region_px(region: list[float] | None, w: int, h: int) -> tuple[int, int, int, int]:
    if not region:
        return 0, 0, w, h
    x, y, rw, rh = region
    return int(x * w), int(y * h), max(1, int(rw * w)), max(1, int(rh * h))


def frame_scores(frames, templates: list[tuple[str, np.ndarray]], region) -> np.ndarray:
    """Лучшее совпадение с любым образцом для каждого кадра (0..1)."""
    import cv2
    scores = []
    for fr in frames:
        x, y, w, h = region
        roi = fr[y:y + h, x:x + w]
        best = 0.0
        for _, t in templates:
            if t.shape[0] > roi.shape[0] or t.shape[1] > roi.shape[1]:
                continue
            best = max(best, float(cv2.minMaxLoc(cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED))[1]))
        scores.append(best)
    return np.array(scores, dtype=np.float32)


def iter_frames(video: Path, fps: float, w: int, h: int):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
           "-vf", f"fps={fps},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size = w * h
    assert proc.stdout is not None
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
    proc.wait()


def to_intervals(per_sec: np.ndarray, thr: float, gap: int = 2, min_len: int = 1) -> list[list[int]]:
    on = per_sec >= thr
    out: list[list[int]] = []
    for i in np.flatnonzero(on):
        if out and i - out[-1][1] <= gap:
            out[-1][1] = int(i) + 1
        else:
            out.append([int(i), int(i) + 1])
    return [x for x in out if x[1] - x[0] >= min_len]


def run_alerts(rec: Record, force: bool = False) -> None:
    out = rec.path("alerts.json")
    if out.exists() and not force:
        log.info("alerts.json уже есть, пропускаю (--force чтобы пересчитать)")
        return
    folder = alert_dir(rec)
    if not folder.exists() or not any(p.suffix.lower() in IMAGE_EXT for p in folder.iterdir()):
        log.info("Нет образцов плашек доната (%s) — шаг пропущен", folder)
        return
    if not rec.proxy.exists():
        raise ClipperError("Нет proxy.mp4 — сначала шаг download.")
    ac = rec.streamer_cfg.get("alerts", {})
    src_h = next((s["height"] for s in probe(rec.source)["streams"] if s.get("codec_type") == "video"), 1080) \
        if rec.source.exists() else ac.get("source_height", 1080)
    pv = next(s for s in probe(rec.proxy)["streams"] if s.get("codec_type") == "video")
    w, h = int(pv["width"]), int(pv["height"])
    templates = load_templates(folder, h / src_h)
    if not templates:
        return
    fps = float(ac.get("fps", 2))
    thr = float(ac.get("threshold", 0.7))
    region = region_px(ac.get("region") or None, w, h)
    with rec.step("alerts"):
        log.info("  образцов: %d, ищу в %s", len(templates), "заданной области" if ac.get("region") else "всём кадре")
        scores = []
        batch = []
        for fr in iter_frames(rec.proxy, fps, w, h):
            batch.append(fr)
            if len(batch) >= 256:
                scores.append(frame_scores(batch, templates, region))
                batch = []
        if batch:
            scores.append(frame_scores(batch, templates, region))
        s = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
        n = int(np.ceil(len(s) / fps))
        per_sec = np.zeros(n, dtype=np.float32)
        for i, v in enumerate(s):
            j = int(i / fps)
            per_sec[j] = max(per_sec[j], v)
        intervals = to_intervals(per_sec, thr)
        write_json(out, {"threshold": thr, "intervals": intervals,
                         "score": [round(float(x), 3) for x in per_sec]}, indent=None)
        log.info("  плашек доната найдено: %d, на экране всего %s", len(intervals),
                 fmt_duration(sum(b - a for a, b in intervals)))
