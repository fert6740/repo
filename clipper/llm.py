"""Единая точка обращения к нейросети. Два взаимозаменяемых способа: claude_cli (по подписке) и anthropic_api."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import ClipperError, Config

log = logging.getLogger("clipper")


@dataclass
class Usage:
    model: str = ""
    input_tokens: int = 0          # всё, что ушло на вход, включая кэш
    cache_read: int = 0
    cache_creation: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    reported_cost_usd: float = 0.0
    context_window: int = 0
    seconds: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class Result:
    data: dict
    usage: Usage


class LimitReached(ClipperError):
    pass


def _outside_home_tmp() -> Path:
    """Пустая временная папка вне домашней: иначе claude подтянет ~/CLAUDE.md и потратит токены."""
    home = Path.home().resolve()
    for base in (tempfile.gettempdir(), "/tmp", "/var/tmp"):
        b = Path(base).resolve()
        if b.is_dir() and home not in (b, *b.parents) and os.access(b, os.W_OK):
            return Path(tempfile.mkdtemp(prefix="clipper-llm-", dir=b))
    raise ClipperError("Не нашёл временную папку вне домашней (нужна для claude -p).")


def parse_cli_output(stdout: str) -> tuple[dict | None, Usage, str]:
    """Разбор ответа `claude -p --output-format json`: (результат по схеме, расход, текст ошибки)."""
    try:
        d = json.loads(stdout)
    except json.JSONDecodeError:
        return None, Usage(), "ответ не в формате JSON: " + stdout.strip()[-500:]
    if isinstance(d, list):  # на случай потокового вывода — берём итоговое сообщение
        d = next((x for x in reversed(d) if isinstance(x, dict) and x.get("type") == "result"), {})
    u = d.get("usage") or {}
    mu = d.get("modelUsage") or {}
    model = max(mu, key=lambda k: mu[k].get("outputTokens", 0)) if mu else ""
    usage = Usage(
        model=model,
        input_tokens=int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0))
        + int(u.get("cache_creation_input_tokens", 0)),
        cache_read=int(u.get("cache_read_input_tokens", 0)),
        cache_creation=int(u.get("cache_creation_input_tokens", 0)),
        output_tokens=int(u.get("output_tokens", 0)),
        thinking_tokens=int((u.get("output_tokens_details") or {}).get("thinking_tokens", 0)),
        reported_cost_usd=float(d.get("total_cost_usd") or 0.0),
        context_window=int(mu.get(model, {}).get("contextWindow", 0)) if model else 0,
    )
    if d.get("is_error") or d.get("subtype") not in (None, "success"):
        return None, usage, str(d.get("result") or d.get("subtype") or "ошибка")
    data = d.get("structured_output")
    if data is None:
        try:
            data = json.loads(d.get("result") or "")
        except json.JSONDecodeError:
            return None, usage, "в ответе нет результата по схеме"
    return data, usage, ""


LOGIN_WORDS = ("not logged in", "please run /login", "invalid api key", "oauth token has expired")
LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "5-hour limit", "weekly limit", "out of extra usage")


class ClaudeCLI:
    name = "claude_cli"

    def __init__(self, cfg: Config):
        self.c = cfg["llm"]
        self.bin = self.c.get("claude_bin", "claude")
        if not shutil.which(self.bin):
            raise ClipperError(f"Не найдена программа {self.bin} (Claude Code).")

    def call(self, model: str, system: str, prompt: str, schema: dict) -> Result:
        work = _outside_home_tmp()
        cmd = [self.bin, "-p", "--model", model, "--output-format", "json",
               "--json-schema", json.dumps(schema, ensure_ascii=False),
               "--tools", "", "--strict-mcp-config", "--no-session-persistence",
               "--system-prompt", system, *self.c.get("claude_extra_args", [])]
        if self.c.get("effort"):
            cmd += ["--effort", self.c["effort"]]
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("CLAUDECODE") and k not in ("CLAUDE_CODE_ENTRYPOINT", "ANTHROPIC_API_KEY")}
        t0 = time.time()
        try:
            p = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=work, env=env,
                               timeout=self.c["timeout_sec"])
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"claude -p не ответил за {self.c['timeout_sec']} с") from e
        finally:
            shutil.rmtree(work, ignore_errors=True)
        data, usage, err = parse_cli_output(p.stdout)
        usage.seconds = round(time.time() - t0, 1)
        if data is None:
            msg = err or p.stderr.strip()[-1000:] or f"код выхода {p.returncode}"
            if any(w in (msg + p.stderr).lower() for w in LOGIN_WORDS):
                raise ClipperError("Claude Code не залогинен. Запустите в терминале claude, войдите по подписке "
                                   "(/login) и повторите команду.")
            if any(w in (msg + p.stderr).lower() for w in LIMIT_WORDS):
                raise LimitReached("Исчерпан лимит подписки Claude: " + msg + "\nПовторите позже — "
                                   "готовые шаги пересчитываться не будут.")
            raise RuntimeError(msg)
        return Result(data, usage)


class AnthropicAPI:
    """На будущее: тот же интерфейс через официальный пакет anthropic (ключ в ANTHROPIC_API_KEY)."""
    name = "anthropic_api"

    def __init__(self, cfg: Config):
        self.c = cfg["llm"]

    def call(self, model: str, system: str, prompt: str, schema: dict) -> Result:
        raise ClipperError("Способ anthropic_api пока не реализован — поставьте backend = \"claude_cli\" в config.toml.")


BACKENDS = {"claude_cli": ClaudeCLI, "anthropic_api": AnthropicAPI}


class LLM:
    def __init__(self, cfg: Config, usage_log: Path | None):
        name = cfg["llm"]["backend"]
        if name not in BACKENDS:
            raise ClipperError(f"Неизвестный способ обращения к нейросети: {name}")
        self.backend = BACKENDS[name](cfg)
        self.retries = int(cfg["llm"].get("retries", 2))
        self.usage_log = usage_log

    def call(self, step: str, model: str, system: str, prompt: str, schema: dict) -> Result:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            if attempt:
                log.warning("  повтор запроса (%d/%d) после ошибки: %s", attempt, self.retries, last)
                time.sleep(10 * attempt)
            try:
                res = self.backend.call(model, system, prompt, schema)
            except (ClipperError, LimitReached):
                raise
            except Exception as e:  # noqa: BLE001
                last = e
                self._log(step, model, None, len(system) + len(prompt), ok=False, error=str(e)[:300])
                continue
            self._log(step, model, res.usage, len(system) + len(prompt), ok=True)
            log.info("  %s [%s]: вход %d, выход %d токенов, %.0f с", step, res.usage.model or model,
                     res.usage.input_tokens, res.usage.output_tokens, res.usage.seconds)
            return res
        raise ClipperError(f"Нейросеть не ответила после {self.retries + 1} попыток: {last}")

    def _log(self, step: str, model: str, u: Usage | None, chars: int, ok: bool, error: str = "") -> None:
        if not self.usage_log:
            return
        row = {"ts": datetime.now().isoformat(timespec="seconds"), "step": step, "backend": self.backend.name,
               "model_alias": model, "ok": ok, "prompt_chars": chars}
        if u:
            row.update(model=u.model, input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                       cache_read=u.cache_read, cache_creation=u.cache_creation,
                       thinking_tokens=u.thinking_tokens, seconds=u.seconds,
                       reported_cost_usd=u.reported_cost_usd)
        if error:
            row["error"] = error
        with open(self.usage_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
