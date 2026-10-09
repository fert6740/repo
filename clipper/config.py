"""Настройки: config.toml (общие) и streamers/<имя>/streamer.toml."""
from __future__ import annotations

import copy
import shutil
import tomllib
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent


class ClipperError(Exception):
    """Понятная пользователю ошибка: печатается без трейсбэка."""


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ClipperError(f"Ошибка в файле настроек {path}: {e}") from e


def _resolve(p: str) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


class Config:
    def __init__(self, data: dict):
        self.data = data

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def data_dir(self) -> Path:
        return _resolve(self.data["paths"]["data_dir"])

    @property
    def streamers_dir(self) -> Path:
        return _resolve(self.data["paths"]["streamers_dir"])

    @property
    def tools_dir(self) -> Path:
        return _resolve(self.data["paths"]["tools_dir"])

    def streamer_dir(self, name: str) -> Path:
        return self.streamers_dir / name

    def streamer(self, name: str) -> dict:
        """Настройки стримера поверх шаблона (недостающие ключи берутся из шаблона)."""
        base = _load_toml(self.streamers_dir / "_template" / "streamer.toml")
        path = self.streamer_dir(name) / "streamer.toml"
        if path.exists():
            return deep_merge(base, _load_toml(path))
        return base

    def ensure_streamer(self, name: str) -> Path:
        d = self.streamer_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        dst = d / "streamer.toml"
        if not dst.exists():
            shutil.copy(self.streamers_dir / "_template" / "streamer.toml", dst)
        return d


def load_config(path: Path | None = None) -> Config:
    path = path or PROJECT_DIR / "config.toml"
    if not path.exists():
        raise ClipperError(f"Не найден файл настроек {path}")
    return Config(_load_toml(path))
