"""clipper doctor — проверка, что на этом компьютере всё готово к работе."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

from .config import Config

OK, WARN, FAIL = "✔", "⚠", "✘"


class Report:
    def __init__(self):
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, what: str, advice: str = "") -> None:
        self.rows.append((status, what, advice))
        print(f"{status} {what}" + (f"\n    → {advice}" if advice else ""), flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == FAIL)

    @property
    def warned(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == WARN)


def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def _http(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 20) -> tuple[bool, str]:
    req = urllib.request.Request(url, data=data, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.read(4000).decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def check_programs(r: Report) -> None:
    for b, need, advice in (("ffmpeg", True, "sudo pacman -S ffmpeg"), ("ffprobe", True, "sudo pacman -S ffmpeg"),
                            ("tmux", True, "sudo pacman -S tmux"), ("claude", True, "установите Claude Code"),
                            ("notify-send", False, "уведомлений не будет; можно поставить: sudo pacman -S libnotify")):
        if shutil.which(b):
            r.add(OK, f"программа {b}")
        else:
            r.add(FAIL if need else WARN, f"нет программы {b}", advice)


def check_gpu(r: Report) -> None:
    p = _run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"])
    if not p or p.returncode != 0:
        r.add(FAIL, "видеокарта NVIDIA не отвечает (nvidia-smi)", "проверьте драйвер: sudo pacman -S nvidia-utils")
        return
    name, used, total = [x.strip() for x in p.stdout.strip().splitlines()[0].split(",")]
    free = int(total) - int(used)
    st = OK if free >= 2500 else WARN
    r.add(st, f"видеокарта {name}: свободно {free} МБ из {total} МБ",
          "" if st == OK else "для распознавания речи нужно ~2,5 ГБ — закройте игры/браузер с видео на время работы")


def check_ffmpeg_gpu(r: Report) -> None:
    if not shutil.which("ffmpeg"):
        return
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    p = _run(base + ["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1",
                     "-c:v", "h264_nvenc", "-f", "null", "-"])
    if p and p.returncode == 0:
        r.add(OK, "кодирование видео на видеокарте (h264_nvenc)")
    else:
        r.add(WARN, "кодирование на видеокарте не работает — будет медленнее (на процессоре)",
              (p.stderr.strip().splitlines() or ["?"])[-1][:200] if p else "ffmpeg не запустился")
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.mp4"
        _run(base + ["-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=1", "-c:v", "libx264", str(f)])
        p = _run(base + ["-hwaccel", "cuda", "-i", str(f), "-f", "null", "-"])
        if p and p.returncode == 0 and "Failed" not in p.stderr:
            r.add(OK, "декодирование видео на видеокарте (cuda)")
        else:
            r.add(WARN, "декодирование на видеокарте не работает — будет на процессоре")


def check_whisper(r: Report, cfg: Config) -> None:
    try:
        import ctranslate2
        import faster_whisper
        from .transcribe import preload_cuda_libs
    except ImportError as e:
        r.add(FAIL, f"не установлены библиотеки распознавания: {e}", "запустите ./install.sh")
        return
    c = cfg["transcribe"]
    if c["device"] == "cuda":
        preload_cuda_libs()
        n = ctranslate2.get_cuda_device_count()
        types = ctranslate2.get_supported_compute_types("cuda") if n else set()
        if n and c["compute_type"] in types:
            r.add(OK, f"faster-whisper {faster_whisper.__version__} видит видеокарту ({c['compute_type']})")
        else:
            r.add(FAIL, "faster-whisper не видит видеокарту",
                  "проверьте драйвер NVIDIA; временно можно device = \"cpu\" в config.toml (очень медленно)")
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    cached = any(hub.glob(f"models--*{c['model']}*")) if hub.exists() else False
    if cached:
        r.add(OK, f"модель распознавания {c['model']} уже скачана")
    else:
        ok, err = _http("https://huggingface.co")
        r.add(OK if ok else FAIL, f"модель {c['model']} скачается при первом запуске (~1,6 ГБ)" if ok
              else "нет доступа к huggingface.co — модель распознавания не скачать", "" if ok else err[:200])


def check_network(r: Report, cfg: Config) -> None:
    ok, err = _http("https://www.twitch.tv/")
    r.add(OK if ok else FAIL, "доступ к twitch.tv" if ok else "нет доступа к twitch.tv", "" if ok else err[:200])
    from .clips import GQL_URL, LIST_CLIENT_ID
    body = json.dumps({"query": 'query{user(login:"twitch"){id}}', "variables": {}}).encode()
    ok, text = _http(GQL_URL, body, {"Client-ID": LIST_CLIENT_ID, "Content-Type": "application/json"})
    good = ok and '"id"' in text and '"errors"' not in text
    r.add(OK if good else WARN, "запросы к Twitch для клипов зрителей" if good
          else "запросы к Twitch для клипов зрителей не проходят — клипы будут пропущены", "" if good else text[:200])
    exe = shutil.which("TwitchDownloaderCLI") or (cfg.tools_dir / "TwitchDownloaderCLI").exists()
    if exe:
        r.add(OK, "TwitchDownloaderCLI (чат) установлен")
    else:
        ok, err = _http("https://api.github.com/repos/lay295/TwitchDownloader/releases/latest")
        r.add(OK if ok else WARN, "TwitchDownloaderCLI (чат) скачается сам при первом запуске" if ok
              else "нет доступа к GitHub — чат не скачать", "" if ok else err[:200])


def check_disk(r: Report, cfg: Config) -> None:
    d = cfg.data_dir
    probe = d if d.exists() else d.parent
    free = shutil.disk_usage(probe).free / 1024 ** 3
    st = OK if free >= 50 else (WARN if free >= 25 else FAIL)
    r.add(st, f"свободно на диске с данными: {free:.0f} ГБ ({d})",
          "" if st == OK else "запись на 6 часов занимает 15–25 ГБ, нужно вдвое больше; освободите место или "
                              "смените data_dir в config.toml")


def check_claude(r: Report, cfg: Config) -> None:
    if not shutil.which(cfg["llm"].get("claude_bin", "claude")):
        return
    p = _run([cfg["llm"].get("claude_bin", "claude"), "--version"])
    if p and p.returncode == 0:
        r.add(OK, f"Claude Code {p.stdout.strip()}")
    from .config import ClipperError
    from .llm import LLM
    try:
        res = LLM(cfg, None).call("doctor", "haiku", "Отвечай строго по схеме.", "Ответь ok = true.",
                                  {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]})
        r.add(OK, f"claude -p отвечает по подписке ({res.usage.model}, {res.usage.input_tokens} токенов на запрос)")
    except ClipperError as e:
        r.add(FAIL, "claude -p не отвечает", str(e)[:300])


def run_doctor(cfg: Config, llm: bool = True) -> int:
    r = Report()
    print("Проверяю этот компьютер…\n")
    check_programs(r)
    check_gpu(r)
    check_ffmpeg_gpu(r)
    check_whisper(r, cfg)
    check_network(r, cfg)
    check_disk(r, cfg)
    if llm:
        check_claude(r, cfg)
    print()
    if r.failed:
        print(f"Есть проблемы: {r.failed}. Исправьте пункты со значком {FAIL}.")
    elif r.warned:
        print(f"Работать можно, но есть замечания: {r.warned}.")
    else:
        print("Всё готово к работе.")
    return 1 if r.failed else 0
