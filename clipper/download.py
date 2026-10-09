"""Шаг 1. Скачивание записи и чата, proxy.mp4, audio.wav, поиск заглушённых Twitch кусков."""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .config import ClipperError
from .media import ffmpeg_try, gray_frames, iter_wav_blocks, probe_duration, wav_info
from .record import Record, fmt_duration, fmt_time, read_json, write_json

log = logging.getLogger("clipper")

GB = 1024 ** 3


def ytdlp_cmd(*args: str) -> list[str]:
    return [sys.executable, "-m", "yt_dlp", "--no-warnings", *args]


def fetch_info(url: str) -> dict:
    p = subprocess.run(ytdlp_cmd("--dump-single-json", "--skip-download", url), capture_output=True, text=True)
    if p.returncode != 0:
        raise ClipperError("yt-dlp не смог получить сведения о записи:\n" + p.stderr.strip()[-2000:])
    return json.loads(p.stdout)


def best_format(info: dict, max_height: int) -> dict | None:
    """Формат, который выберет yt-dlp по строке из pick_format."""
    fmts = [f for f in info.get("formats", []) if f.get("vcodec") not in (None, "none") and f.get("height")]
    ok = [f for f in fmts if f["height"] <= max_height] or fmts
    return max(ok, key=lambda f: (f["height"], f.get("fps") or 0, f.get("tbr") or 0)) if ok else None


def pick_format(info: dict, max_height: int) -> tuple[str, float]:
    """Строка формата для yt-dlp и оценка размера файла в байтах."""
    dur = float(info.get("duration") or 0)
    best = best_format(info, max_height)
    est = 0.0
    if best:
        est = best.get("filesize") or best.get("filesize_approx") or 0
        if not est and best.get("tbr") and dur:
            est = best["tbr"] * 1000 / 8 * dur
    if not est:
        est = 3.5 * GB * dur / 3600  # 1080p60 около 7–8 Мбит/с
    return f"b[height<={max_height}][vcodec!=none]/b", float(est)


def check_disk(path: Path, need: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < need:
        raise ClipperError(
            f"Мало места на диске: нужно не меньше {need / GB:.1f} ГБ, свободно {free / GB:.1f} ГБ "
            f"(папка {path}). Освободите место или укажите другую папку data_dir в config.toml.")


def download_video(rec: Record, info: dict) -> None:
    cfg = rec.cfg["download"]
    fmt, est = pick_format(info, cfg["max_height"])
    need = est * cfg["disk_reserve_factor"]
    log.info("Размер записи примерно %.1f ГБ, проверяю место (нужно %.1f ГБ)", est / GB, need / GB)
    check_disk(rec.dir, need)
    cmd = ytdlp_cmd("-f", fmt, "-P", str(rec.dir), "-o", "source.%(ext)s",
                    "--concurrent-fragments", str(cfg["concurrent_fragments"]),
                    "--retries", "20", "--fragment-retries", "50", "--retry-sleep", "fragment:exp=1:30",
                    "--remux-video", "mp4", "--newline", "--continue", rec.url)
    log.debug("$ %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last, tail = 0.0, []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-20:]
        if "[download]" in line and "%" in line and time.time() - last > 60:
            log.info("  %s", line.replace("[download]", "скачивание:").strip())
            last = time.time()
    if proc.wait() != 0:
        raise ClipperError("yt-dlp не смог скачать запись:\n" + "\n".join(tail))
    got = sorted(rec.dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    got = [p for p in got if p.suffix in (".mp4", ".mkv", ".ts", ".webm")]
    if not got:
        raise ClipperError("yt-dlp завершился, но файл записи не найден в " + str(rec.dir))
    if got[0] != rec.source:
        got[0].replace(rec.source)


# ---------- чат ----------

def parse_chat(raw) -> list[dict]:
    """Чат из формата TwitchDownloader / старого rechat в список {time, author, text}."""
    comments = raw.get("comments", []) if isinstance(raw, dict) else raw
    out = []
    for c in comments or []:
        t = c.get("content_offset_seconds")
        if t is None:
            continue
        who = c.get("commenter") or {}
        author = who.get("display_name") or who.get("name") or who.get("login") or ""
        msg = c.get("message") or {}
        text = msg.get("body")
        if text is None:
            text = "".join(f.get("text", "") for f in msg.get("fragments") or [])
        out.append({"time": round(float(t), 2), "author": author, "text": text})
    out.sort(key=lambda m: m["time"])
    return out


def twitch_downloader(rec: Record) -> Path:
    found = shutil.which("TwitchDownloaderCLI")
    if found:
        return Path(found)
    tools = rec.cfg.tools_dir
    exe = tools / "TwitchDownloaderCLI"
    if exe.exists():
        return exe
    log.info("  скачиваю TwitchDownloaderCLI в %s", tools)
    tools.mkdir(parents=True, exist_ok=True)
    api = "https://api.github.com/repos/lay295/TwitchDownloader/releases/latest"
    with urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "stream-clipper"}), timeout=60) as r:
        rel = json.load(r)
    assets = [a for a in rel.get("assets", []) if "CLI" in a["name"] and "Linux-x64" in a["name"]]
    if not assets:
        raise ClipperError("Не нашёл сборку TwitchDownloaderCLI для Linux в последнем выпуске.")
    zpath = tools / assets[0]["name"]
    urllib.request.urlretrieve(assets[0]["browser_download_url"], zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(tools)
    zpath.unlink()
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def chat_via_twitch_downloader(rec: Record) -> list[dict] | None:
    try:
        exe = twitch_downloader(rec)
    except Exception as e:  # сеть, GitHub
        log.warning("  не удалось получить TwitchDownloaderCLI: %s", e)
        return None
    raw = rec.path("chat_raw.json")
    raw.unlink(missing_ok=True)
    env = dict(os.environ, DOTNET_SYSTEM_GLOBALIZATION_INVARIANT="1")
    p = subprocess.run([str(exe), "chatdownload", "--id", rec.vod_id, "-o", str(raw),
                        "--collision", "Overwrite", "--banner", "false"],
                       capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL)
    if p.returncode != 0 or not raw.exists():
        log.warning("  TwitchDownloaderCLI не скачал чат: %s", (p.stderr or p.stdout).strip()[-500:])
        return None
    msgs = parse_chat(read_json(raw))
    raw.unlink(missing_ok=True)
    return msgs


def download_chat(rec: Record) -> None:
    # yt-dlp чат Twitch больше не скачивает (в его коде нет rechat) — сразу TwitchDownloaderCLI
    msgs = chat_via_twitch_downloader(rec)
    if msgs is not None:
        write_json(rec.chat, {"source": "TwitchDownloaderCLI", "messages": msgs}, indent=None)
        log.info("  чат: %d сообщений", len(msgs))
        return
    log.warning("Чат скачать не удалось — продолжаю без него (сигналы чата будут нулевые). "
                "Повторный запуск попробует снова.")


# ---------- звук и proxy ----------

def make_audio(rec: Record, total: float) -> None:
    ffmpeg_try([("wav", ["-i", rec.source, "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le"])],
               rec.audio, total=total)


def make_proxy(rec: Record, total: float) -> None:
    c = rec.cfg["download"]
    vf = f"scale=-2:{c['proxy_height']},fps={c['proxy_fps']}"
    g = str(int(c["proxy_fps"] * c["proxy_keyint_sec"]))
    audio = ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart"]
    nv = ["-c:v", c["proxy_encoder"], "-preset", "p4", "-cq", "30", "-g", g, "-bf", "0"]
    x264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-g", g, "-keyint_min", g, "-sc_threshold", "0"]
    variants = []
    if c["hwaccel"]:
        variants.append(("nvenc+cuda", ["-hwaccel", "cuda", "-i", rec.source, "-vf", vf, *nv, *audio]))
    variants.append(("nvenc", ["-i", rec.source, "-vf", vf, *nv, *audio]))
    variants.append(("libx264", ["-i", rec.source, "-vf", vf, *x264, *audio]))
    used = ffmpeg_try(variants, rec.proxy, total=total)
    log.info("  proxy.mp4 сделан (%s)", used)


# ---------- заглушённые куски ----------

MUTED_SEG_RE = re.compile(r"-muted\.ts(\?|$)")


def muted_from_playlist(text: str) -> list[list[float]]:
    """Twitch помечает заглушённые куски записи в плейлисте именами вида «123-muted.ts»."""
    out: list[list[float]] = []
    t, dur = 0.0, None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            try:
                dur = float(line[8:].split(",")[0])
            except ValueError:
                dur = None
        elif line and not line.startswith("#") and dur is not None:
            if MUTED_SEG_RE.search(line):
                if out and abs(out[-1][1] - t) < 1e-3:
                    out[-1][1] = round(t + dur, 3)
                else:
                    out.append([round(t, 3), round(t + dur, 3)])
            t += dur
            dur = None
    return out


def fetch_muted_playlist(rec: Record, info: dict) -> None:
    """Сохраняет заглушённые куски по плейлисту выбранного качества (ссылки на плейлист быстро устаревают)."""
    f = best_format(info, rec.cfg["download"]["max_height"])
    url = f and f.get("url")
    if not url or ".m3u8" not in url:
        return
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001 — не страшно, есть запасной способ по звуку
        log.warning("  не удалось прочитать плейлист записи (%s) — заглушённые куски будут искаться по звуку", e)
        return
    if "#EXTINF" not in text:
        return
    write_json(rec.path("muted_playlist.json"), muted_from_playlist(text))

def silent_runs(rec: Record, peak: float, min_sec: float, hop: float = 0.1) -> list[tuple[float, float]]:
    """Отрезки цифровой тишины не короче min_sec."""
    sr, _ = wav_info(rec.audio)
    win = int(sr * hop)
    flags = []
    for _, x in iter_wav_blocks(rec.audio, block_sec=600.0):
        n = len(x) // win
        if n:
            flags.append(np.abs(x[: n * win]).reshape(n, win).max(axis=1) < peak)
        if len(x) % win:
            flags.append(np.array([np.abs(x[n * win:]).max() < peak]))
    if not flags:
        return []
    f = np.concatenate(flags).astype(np.int8)
    edges = np.diff(np.concatenate([[0], f, [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(s * hop, e * hop) for s, e in zip(starts, ends) if (e - s) * hop >= min_sec]


def picture_motion(video: Path, start: float, end: float) -> float:
    """Средняя разница соседних кадров (0..255) на отрезке, кадр раз в секунду, не больше 30 кадров."""
    dur = end - start
    fps = 1.0 if dur <= 30 else 30.0 / dur
    fr = gray_frames(video, start, dur, fps=fps).astype(np.float32)
    if len(fr) < 2:
        return 0.0
    return float(np.abs(np.diff(fr, axis=0)).mean())


def find_muted(rec: Record) -> None:
    c = rec.cfg["muted"]
    video = rec.proxy if rec.proxy.exists() else rec.source
    muted, static = [], []
    for s, e in silent_runs(rec, c["silence_peak"], c["min_sec"]):
        m = picture_motion(video, s, e)
        (muted if m >= c["frame_diff_min"] else static).append([round(s, 1), round(e, 1)])
    pl = rec.path("muted_playlist.json")
    if pl.exists():  # точная разметка Twitch; тишина по звуку остаётся как справка
        write_json(rec.muted, {"source": "playlist", "muted": read_json(pl),
                               "audio_guess": muted, "silent_static": static})
        muted = read_json(pl)
    else:
        write_json(rec.muted, {"source": "audio", "muted": muted, "silent_static": static})
    total = sum(e - s for s, e in muted)
    if muted:
        log.info("  заглушено Twitch: %d кусков, всего %s", len(muted), fmt_duration(total))
    else:
        log.info("  заглушённых кусков не найдено")


# ---------- шаг целиком ----------

def save_info(rec: Record, info: dict) -> None:
    keep = {k: info.get(k) for k in ("id", "title", "duration", "uploader", "uploader_id", "timestamp",
                                     "upload_date", "webpage_url", "description")}
    keep["vod_id"] = rec.vod_id
    keep["streamer"] = rec.streamer
    # главы записи Twitch — смены игры/категории
    keep["chapters"] = [{"start": c.get("start_time") or 0, "end": c.get("end_time"), "title": c.get("title") or ""}
                        for c in info.get("chapters") or []]
    write_json(rec.info, keep)


def update_streamer_login(rec: Record, info: dict) -> None:
    path = rec.cfg.streamer_dir(rec.streamer) / "streamer.toml"
    login = info.get("uploader_id") or ""
    if not path.exists() or not login:
        return
    text = path.read_text(encoding="utf-8")
    if 'twitch_login = ""' in text:
        path.write_text(text.replace('twitch_login = ""', f'twitch_login = "{login}"', 1), encoding="utf-8")


def run_download(rec: Record, force: bool = False) -> None:
    rec.create()
    if force:
        for p in (rec.source, rec.audio, rec.proxy, rec.chat, rec.muted, rec.info, rec.path("muted_playlist.json"),
                  rec.path("clips.json")):
            p.unlink(missing_ok=True)

    if not rec.info.exists():
        with rec.step("download.info"):
            info = fetch_info(rec.url)
            save_info(rec, info)
            fetch_muted_playlist(rec, info)
            update_streamer_login(rec, info)
            log.info("  «%s», %s, длительность %s", info.get("title"), info.get("uploader"),
                     fmt_time(info.get("duration") or 0))
    else:
        info = read_json(rec.info)

    if rec.source.exists():
        log.info("source.mp4 уже есть, пропускаю скачивание")
    else:
        with rec.step("download.video"):
            fresh = fetch_info(rec.url)
            if not rec.path("muted_playlist.json").exists():
                fetch_muted_playlist(rec, fresh)
            download_video(rec, fresh)

    total = probe_duration(rec.source)
    if not info.get("duration"):
        info["duration"] = total
        write_json(rec.info, info)

    if not rec.audio.exists():
        with rec.step("download.audio"):
            make_audio(rec, total)
    if not rec.proxy.exists():
        with rec.step("download.proxy"):
            make_proxy(rec, total)
    if not rec.muted.exists():
        with rec.step("download.muted"):
            find_muted(rec)
    if not rec.chat.exists():
        with rec.step("download.chat"):
            download_chat(rec)
    if not rec.path("clips.json").exists():
        from .clips import download_clips
        with rec.step("download.clips"):
            download_clips(rec)
