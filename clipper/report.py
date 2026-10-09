"""Страницы: список тем и шортов, сравнение моделей."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path

from .config import ClipperError
from .llm import LLM
from .record import Record, read_json, write_json
from .topics import pass1, topics_file_for
from .usage import read_usage, summarize

log = logging.getLogger("clipper")

TEMPLATE = Path(__file__).parent / "templates" / "topics.html"


def run_entry(label: str, data: dict, usage: dict | None) -> dict:
    return {"label": label, "topics": data["topics"], "shorts": data["shorts"],
            "problems": data.get("seam_problems", []),
            "usage": usage if usage and usage["calls"] else None}


def render_page(rec: Record, out: Path, runs: list[dict], title: str) -> Path:
    info = rec.load_info()
    video = os.path.relpath(rec.proxy, out.parent) if rec.proxy.exists() else ""
    muted = read_json(rec.muted)["muted"] if rec.muted.exists() else []
    data = {"title": title, "streamer": info.get("uploader") or rec.streamer, "url": rec.url,
            "duration": rec.duration(), "video": video, "muted": muted, "runs": runs}
    html = TEMPLATE.read_text(encoding="utf-8")
    page_title = (info.get("title") or rec.vod_id)[:60]
    html = html.replace("__TITLE__", page_title.replace("<", "&lt;"))
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def topics_page(rec: Record) -> Path:
    if not rec.topics.exists():
        raise ClipperError("Нет topics.json — сначала шаг topics.")
    data = read_json(rec.topics)
    usage = summarize(rec.cfg, read_usage(rec), "pass1")
    info = rec.load_info()
    return render_page(rec, rec.path("topics.html"),
                       [run_entry(data.get("model", ""), data, usage)],
                       f"Темы: {info.get('title') or rec.vod_id}")


def compare_models(rec: Record, models: list[str], force: bool = False) -> Path:
    llm = LLM(rec.cfg, rec.usage_log)
    runs = []
    for m in models:
        path = topics_file_for(rec, m)
        path.parent.mkdir(exist_ok=True)
        prefix = f"compare.{m}"
        if force or not path.exists():
            if not force and m == rec.cfg["llm"]["pass1_model"] and rec.topics.exists():
                shutil.copy(rec.topics, path)  # уже посчитано основным проходом — токены не тратим
            else:
                with rec.step(f"compare.{m}"):
                    write_json(path, pass1(rec, llm, m, step=f"compare.{m}.pass1"))
        data = read_json(path)
        rows = read_usage(rec)
        usage = summarize(rec.cfg, rows, prefix)
        if not usage["calls"] and data.get("model") == rec.cfg["llm"]["pass1_model"]:
            usage = summarize(rec.cfg, rows, "pass1")
        runs.append(run_entry(m, data, usage))
    info = rec.load_info()
    return render_page(rec, rec.path("compare") / "compare.html", runs,
                       f"Сравнение моделей: {info.get('title') or rec.vod_id}")


def open_in_browser(path: Path) -> None:
    opener = shutil.which("xdg-open")
    if opener and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
