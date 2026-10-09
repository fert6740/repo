"""Клипы, которые зрители нарезали из этой записи: сильная подсказка, где были яркие моменты."""
from __future__ import annotations

import json
import logging
import time
import urllib.request

from .record import Record, write_json

log = logging.getLogger("clipper")

GQL_URL = "https://gql.twitch.tv/gql"
# Публичные Client-ID веб-клиента Twitch — так же их использует TwitchDownloader
LIST_CLIENT_ID = "kd1unb4b3q4t58fwlpcbzcbnm76a8fp"

CLIPS_QUERY = ('query{user(login:"%s"){clips(first:100%s,criteria:{period:%s}){edges{cursor,node{'
               'id,slug,title,createdAt,curator{displayName},durationSeconds,viewCount,videoOffsetSeconds,'
               'video{id}}},pageInfo{hasNextPage}}}}')


def gql(query: str, client_id: str = LIST_CLIENT_ID) -> dict:
    body = json.dumps({"query": query, "variables": {}}).encode()
    req = urllib.request.Request(GQL_URL, data=body, headers={
        "Client-ID": client_id, "Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    if data.get("errors"):
        raise RuntimeError("; ".join(e.get("message", "?") for e in data["errors"]))
    return data["data"]


def period_for(vod_timestamp: float | None) -> str:
    """Самый узкий период выборки клипов, в который точно попадает запись."""
    if not vod_timestamp:
        return "ALL_TIME"
    days = (time.time() - vod_timestamp) / 86400
    if days < 0.9:
        return "LAST_DAY"
    if days < 6.5:
        return "LAST_WEEK"
    if days < 29:
        return "LAST_MONTH"
    return "ALL_TIME"


def parse_page(data: dict, vod_id: str) -> tuple[list[dict], str | None]:
    clips = ((data.get("user") or {}).get("clips") or {})
    edges = clips.get("edges") or []
    out = []
    for e in edges:
        n = e.get("node") or {}
        if str((n.get("video") or {}).get("id")) != str(vod_id) or n.get("videoOffsetSeconds") is None:
            continue
        start = float(n["videoOffsetSeconds"])
        out.append({"start": start, "end": start + float(n.get("durationSeconds") or 30),
                    "views": int(n.get("viewCount") or 0), "title": n.get("title") or "",
                    "slug": n.get("slug") or "", "curator": (n.get("curator") or {}).get("displayName") or "",
                    "url": f"https://clips.twitch.tv/{n.get('slug')}"})
    more = (clips.get("pageInfo") or {}).get("hasNextPage")
    cursor = edges[-1].get("cursor") if edges and more else None
    return out, cursor


def fetch_vod_clips(login: str, vod_id: str, vod_timestamp: float | None, max_pages: int = 20) -> list[dict]:
    period = period_for(vod_timestamp)
    found, cursor = [], None
    for _ in range(max_pages):
        after = f',after:"{cursor}"' if cursor else ""
        page, cursor = parse_page(gql(CLIPS_QUERY % (login, after, period)), vod_id)
        found += page
        if not cursor:
            break
    found.sort(key=lambda c: c["start"])
    return found


def group_clips(clips: list[dict]) -> list[dict]:
    """Несколько зрителей часто клипают один момент — объединяем пересекающиеся клипы."""
    groups: list[dict] = []
    for c in sorted(clips, key=lambda c: c["start"]):
        g = groups[-1] if groups else None
        if g and c["start"] < g["end"] - 0.5 * (c["end"] - c["start"]):
            g["end"] = max(g["end"], c["end"])
            g["views"] += c["views"]
            g["count"] += 1
            if c["views"] > g["top_views"]:
                g["title"], g["top_views"] = c["title"], c["views"]
        else:
            groups.append({"start": c["start"], "end": c["end"], "views": c["views"], "count": 1,
                           "title": c["title"], "top_views": c["views"]})
    return groups


def download_clips(rec: Record) -> None:
    info = rec.load_info()
    login = info.get("uploader_id") or rec.streamer_cfg.get("twitch_login")
    if not login:
        log.warning("  неизвестен логин стримера на Twitch — клипы зрителей пропущены")
        return
    try:
        clips = fetch_vod_clips(login, rec.vod_id, info.get("timestamp"))
    except Exception as e:  # noqa: BLE001 — клипы необязательны
        log.warning("  клипы зрителей получить не удалось (%s) — продолжаю без них", e)
        return
    write_json(rec.path("clips.json"), {"clips": clips})
    log.info("  клипов зрителей из этой записи: %d", len(clips))
