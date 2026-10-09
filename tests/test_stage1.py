import json
import wave
from pathlib import Path

import numpy as np
import pytest

from clipper import topics as T
from clipper.config import Config, PROJECT_DIR, deep_merge, load_config
from clipper.digest import Line, build_chunks, build_lines, render, split_windows
from clipper.download import parse_chat, pick_format, silent_runs
from clipper.llm import LLM, Result, Usage, parse_cli_output
from clipper.record import Record, fmt_time, parse_time, parse_vod_id, write_json
from clipper.transcribe import filter_segments


@pytest.fixture
def cfg(tmp_path):
    base = load_config()
    data = deep_merge(base.data, {"paths": {"data_dir": str(tmp_path / "data"),
                                            "streamers_dir": str(PROJECT_DIR / "streamers")}})
    return Config(data)


def seg(start, end, text, **kw):
    words = text.split()
    step = (end - start) / max(len(words), 1)
    return {"start": start, "end": end, "text": text, "avg_logprob": kw.get("lp", -0.2),
            "no_speech_prob": kw.get("nsp", 0.05), "compression_ratio": kw.get("cr", 1.3),
            "words": [{"start": start + i * step, "end": start + (i + 0.9) * step, "word": w, "prob": 0.9}
                      for i, w in enumerate(words)]}


def test_time_helpers():
    assert fmt_time(3723) == "1:02:03"
    assert parse_time("1:02:03") == 3723
    assert parse_vod_id("https://www.twitch.tv/videos/2345678901?t=1h") == "2345678901"
    assert parse_vod_id("v123") == "123"
    assert parse_vod_id("hello") is None


def test_filter_segments(cfg):
    c = cfg["transcribe"]
    segs = [seg(0, 3, "привет всем"),
            seg(3, 6, "Субтитры сделал DimaTorzok"),
            seg(6, 9, "тишина", lp=-1.2, nsp=0.8),
            seg(9, 12, "ааа ааа ааа", cr=3.0),
            seg(12, 14, "да"), seg(14, 16, "да"), seg(16, 18, "да"), seg(18, 20, "да"),
            seg(20, 22, "что-то", lp=-2.0)]
    kept, dropped = filter_segments(segs, c)
    assert [s["text"] for s in kept] == ["привет всем", "да", "да"]
    reasons = [d["reason"] for d in dropped]
    assert reasons == ["known_hallucination", "no_speech", "repetitive", "repeat", "repeat", "low_confidence"]


def test_parse_chat():
    raw = {"comments": [
        {"content_offset_seconds": 12.5, "commenter": {"display_name": "Вася"}, "message": {"body": "ахаха KEKW"}},
        {"content_offset_seconds": 3, "commenter": {"name": "petya"},
         "message": {"fragments": [{"text": "при"}, {"text": "вет"}]}},
    ]}
    msgs = parse_chat(raw)
    assert msgs == [{"time": 3.0, "author": "petya", "text": "привет"},
                    {"time": 12.5, "author": "Вася", "text": "ахаха KEKW"}]


def test_pick_format():
    info = {"duration": 3600, "formats": [
        {"format_id": "160p", "height": 160, "vcodec": "avc1", "tbr": 300},
        {"format_id": "1080p60", "height": 1080, "fps": 60, "vcodec": "avc1", "tbr": 6000},
        {"format_id": "1440p", "height": 1440, "vcodec": "avc1", "tbr": 9000},
        {"format_id": "audio", "vcodec": "none", "tbr": 160}]}
    fmt, est = pick_format(info, 1080)
    assert "height<=1080" in fmt
    assert abs(est - 6000 * 1000 / 8 * 3600) < 1


def test_parse_cli_output():
    out = json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "{\"a\":1}",
                      "structured_output": {"a": 1}, "total_cost_usd": 0.01,
                      "usage": {"input_tokens": 5, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 20,
                                "output_tokens": 7, "output_tokens_details": {"thinking_tokens": 2}},
                      "modelUsage": {"claude-opus-5-5": {"outputTokens": 7, "contextWindow": 1000000}}})
    data, u, err = parse_cli_output(out)
    assert data == {"a": 1} and not err
    assert u.input_tokens == 125 and u.output_tokens == 7 and u.thinking_tokens == 2
    assert u.model == "claude-opus-5-5" and u.context_window == 1000000
    data, u, err = parse_cli_output(json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True}))
    assert data is None and err


def test_build_chunks_and_lines():
    segs = [seg(i * 5, i * 5 + 4.5, f"фраза номер {i}") for i in range(20)]  # 0..100 с
    segs += [seg(200, 205, "после паузы")]
    chunks = build_chunks(segs, 20, 30, 20)
    for ch in chunks[:-2]:
        assert 20 <= ch["end"] - ch["start"] <= 30
    lines = build_lines(segs, None, [[150.0, 160.0]], {"chunk_min_sec": 20, "chunk_max_sec": 30,
                                                       "silence_mark_sec": 20, "chat_ratio_mark": 2,
                                                       "laugh_fraction_mark": .3, "laugh_count_mark": 3,
                                                       "shout_z": 2.5}, duration=300)
    text = render(lines)
    assert "⟨звук вырезан 10с⟩" in text
    assert "⟨тишина" in text  # 100..200 минус 10 с вырезанного звука, и хвост 205..300
    assert text.splitlines()[0].startswith("[0:00:00] фраза номер 0")


def test_split_windows():
    lines = [Line(i * 25, i * 25 + 25, "слово " * 40) for i in range(400)]  # ~2.8 часа
    ws = split_windows(lines, 20000, 600, 2.5)
    assert len(ws) > 1
    for a, b in zip(ws, ws[1:]):
        assert b[0].start < a[-1].end  # есть перекрытие
        assert a[-1].end - b[0].start <= 600 + 25
    assert ws[-1][-1] is lines[-1]
    assert split_windows(lines[:5], 20000, 600, 2.5) == [lines[:5]]


def t(start, end, title, score=7, kind="story", cont=False):
    return {"start": start, "end": end, "title": title, "summary": "", "kind": kind, "score": score,
            "why": "", "self_contained": True, "continues": cont}


def test_merge_windows_open_topic():
    # окно 1: 0..4000, окно 2: 3400..8000; тема B не закончена в окне 1
    w1 = {"topics": [t(0, 1500, "A"), t(1500, 3600, "B старт"), t(3600, 4000, "C", cont=True)], "shorts": []}
    w2 = {"topics": [t(3600, 5000, "C"), t(5000, 8000, "D")], "shorts": []}
    topics, _ = T.merge_windows([w1, w2], [(0, 4000), (3400, 8000)])
    assert [(x["title"], x["start"], x["end"]) for x in topics] == [
        ("A", 0, 1500), ("B старт", 1500, 3600), ("C", 3600, 5000), ("D", 5000, 8000)]
    assert T.seam_problems(topics, [(0, 4000), (3400, 8000)], 30) == []


def test_merge_windows_topic_spanning_seam():
    # тема B идёт через начало окна 2; окно 2 видит только её хвост
    w1 = {"topics": [t(0, 3000, "A"), t(3000, 4000, "B", cont=True)], "shorts": [
        {"start": 3500, "end": 3540, "hook": "x", "score": 7, "why": ""}]}
    w2 = {"topics": [t(3400, 4300, "B хвост"), t(4300, 8000, "D")], "shorts": [
        {"start": 3505, "end": 3545, "hook": "y", "score": 9, "why": ""}]}
    topics, shorts = T.merge_windows([w1, w2], [(0, 4000), (3400, 8000)])
    assert [(x["start"], x["end"]) for x in topics] == [(0, 3000), (3000, 4300), (4300, 8000)]
    assert topics[1]["merged_from"] == ["B"]
    assert len(shorts) == 1 and shorts[0]["hook"] == "y"


def test_select(cfg):
    sc = cfg.streamer("_template")
    topics = [t(0, 600, "ok", 8), t(600, 700, "short", 9), t(700, 1300, "low", 4),
              t(1300, 1900, "don", 9, "donations_block"), t(1900, 2500, "muted", 9)]
    T.select(topics, [], [[2000, 2100]], cfg, sc)
    assert [x["selected"] for x in topics] == [True, False, False, False, False]
    assert "звук вырезан Twitch" in topics[4]["reject"]


# ---------- синтетическая запись: сигналы и проход 1 с поддельной нейросетью ----------

def make_record(cfg, dur=1800):
    rec = Record(cfg, "_template", "999")
    rec.dir.mkdir(parents=True)
    sr = 16000
    tt = np.arange(dur * sr) / sr
    x = 0.05 * np.sin(2 * np.pi * 200 * tt)
    x[600 * sr: 603 * sr] *= 15  # крик
    x[1000 * sr: 1020 * sr] = 0  # заглушено
    with wave.open(str(rec.audio), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    segs = [seg(s, s + 4.5, "раз два три четыре пять шесть") for s in range(0, dur - 5, 5) if not 1000 <= s < 1020]
    write_json(rec.transcript, {"segments": segs})
    msgs = [{"time": float(s), "author": "a", "text": "привет"} for s in range(0, dur, 20)]
    msgs += [{"time": 605.0 + i * 0.2, "author": "b", "text": "АХАХАХ KEKW"} for i in range(30)]
    write_json(rec.chat, {"messages": msgs})
    write_json(rec.muted, {"muted": [[1000.0, 1020.0]], "silent_static": []})
    write_json(rec.info, {"title": "тест", "duration": dur})
    return rec


def test_signals_and_digest(cfg):
    from clipper.signals import run_signals
    import pandas as pd
    rec = make_record(cfg)
    run_signals(rec)
    df = pd.read_parquet(rec.signals)
    assert len(df) == 1800
    assert df.loc[601, "loud_db_z"] > 2.5
    assert df.loc[300, "speech"] > 0.5 and df.loc[1010, "speech"] == 0
    assert df["muted"].sum() == 20
    assert df.loc[600, "laugh"] >= 10  # смех в 605..611, сдвиг чата 8 с
    lines = T.load_lines(rec)
    text = render(lines)
    assert "⟨крик⟩" in text and "⟨смех в чате⟩" in text and "⟨чат ×" in text
    assert "⟨звук вырезан 20с⟩" in text
    assert silent_runs(rec, 0.0005, 5) == [(1000.0, 1020.0)]


class FakeBackend:
    name = "fake"

    def __init__(self):
        self.calls = []

    def call(self, model, system, prompt, schema):
        self.calls.append((model, system, prompt, schema))
        lines = [l for l in prompt.split("Расшифровка:\n", 1)[1].splitlines() if l.startswith("[")]
        a, b = lines[0][1:].split("]")[0], lines[-1][1:].split("]")[0]
        mid = fmt_time((parse_time(a) + parse_time(b)) / 2)
        data = {"topics": [
            {"start": a, "end": mid, "title": "Первая", "summary": "s", "kind": "story", "score": 8,
             "why": "w", "self_contained": True},
            {"start": mid, "end": b, "title": "Вторая", "summary": "s", "kind": "discussion", "score": 5,
             "why": "w", "self_contained": True}],
            "shorts": [{"start": a, "end": fmt_time(parse_time(a) + 40), "hook": "h", "score": 9, "why": "w"}]}
        if "recap" in schema["properties"]:
            data["recap"] = "пересказ"
            for x in data["topics"]:
                x["continues"] = False
            data["topics"][-1]["continues"] = True
        return Result(data, Usage(model="claude-" + model, input_tokens=len(prompt) // 3, output_tokens=500,
                                  seconds=1.0))


def test_pass1_single_and_windowed(cfg):
    from clipper.signals import run_signals
    from clipper.usage import report
    from clipper.report import topics_page
    rec = make_record(cfg)
    run_signals(rec)
    llm = LLM(cfg, rec.usage_log)
    fake = FakeBackend()
    llm.backend = fake
    out = T.pass1(rec, llm, "opus")
    assert len(fake.calls) == 1 and "recap" not in fake.calls[0][3]["properties"]
    assert [x["id"] for x in out["topics"]] == [1, 2]
    assert out["topics"][0]["selected"] and not out["topics"][1]["selected"]
    assert isinstance(out["topics"][0]["start"], float)
    assert "Пример" not in fake.calls[0][2]

    cfg.data["llm"]["window_tokens"] = 2000
    fake.calls.clear()
    out = T.pass1(rec, llm, "haiku")
    assert len(fake.calls) > 1
    assert "Пересказ предыдущих частей" in fake.calls[1][2] and "Незакрытая тема" in fake.calls[1][2]
    ts = out["topics"]
    for a, b in zip(ts, ts[1:]):
        assert a["end"] <= b["start"] + 1e-6
    write_json(rec.topics, out)
    page = topics_page(rec)
    html = page.read_text()
    assert "Первая" in html and "__DATA__" not in html
    rep = report(rec)
    assert "Итого" in rep and "haiku" in rep and "opus" in rep


def test_transcribe_resume_and_oom(cfg, monkeypatch):
    from types import SimpleNamespace as NS
    from clipper import transcribe as TR
    rec = make_record(cfg, dur=300)
    rec.transcript.unlink()
    cfg.data["transcribe"]["chunk_minutes"] = 2
    calls = {"n": 0, "batches": []}

    class FakeModel:
        pass

    def fake_run(self, audio):
        calls["n"] += 1
        calls["batches"].append(self.batch)
        if self.batch >= 8:
            raise RuntimeError("CUDA failed with error out of memory")
        dur = len(audio) / 16000
        w = [NS(start=1.0, end=1.5, word=" привет", probability=0.9)]
        return [NS(start=1.0, end=1.5, text=" привет", words=w, avg_logprob=-0.1, no_speech_prob=0.01,
                   compression_ratio=1.0),
                NS(start=dur - 2, end=dur - 1, text=" Продолжение следует...", words=[], avg_logprob=-0.5,
                   no_speech_prob=0.3, compression_ratio=1.0)]

    monkeypatch.setattr(TR, "load_model", lambda c: FakeModel())
    monkeypatch.setattr(TR.Transcriber, "_run", fake_run)
    TR.run_transcribe(rec)
    data = json.loads(rec.transcript.read_text())
    chunks = json.loads((rec.dir / "transcript_parts" / "chunks.json").read_text())
    assert len(chunks) == 3 and chunks[0][0] == 0 and chunks[-1][1] == 300
    assert len(data["segments"]) == 3 and all(s["text"] == "привет" for s in data["segments"])
    assert calls["batches"][:2] == [8, 4]
    # продолжение после прерывания: удалили один кусок и итог — пересчитывается только он
    (rec.dir / "transcript_parts" / "part_001.json").unlink()
    rec.transcript.unlink()
    calls["n"] = 0
    TR.run_transcribe(rec)
    assert calls["n"] == 2  # снова 8 → OOM, затем 4
