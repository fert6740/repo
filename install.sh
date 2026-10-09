#!/usr/bin/env bash
# Установка stream-clipper в окружение проекта (без изменений в системе).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
PROJECT="$PWD"

echo "== Проверка программ"
missing=()
for b in ffmpeg ffprobe tmux claude python3; do
  command -v "$b" >/dev/null || missing+=("$b")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Не найдены: ${missing[*]}"
  echo "Для ffmpeg/tmux/python3 выполните: sudo pacman -S --needed ffmpeg tmux python"
  [[ " ${missing[*]} " == *" claude "* ]] && echo "claude — это Claude Code, установите его по инструкции Anthropic."
  exit 1
fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader \
  || echo "Внимание: nvidia-smi не найден — распознавание речи на видеокарте может не заработать."

echo "== uv (менеджер Python-окружений) — ставится внутрь проекта"
if command -v uv >/dev/null; then
  UV=uv
else
  if [ ! -x .bootstrap/bin/uv ]; then
    python3 -m venv .bootstrap
    .bootstrap/bin/pip install -q --upgrade pip uv
  fi
  UV="$PROJECT/.bootstrap/bin/uv"
fi

echo "== Python 3.12 и окружение .venv"
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python 3.12 .venv
fi

echo "== Зависимости (первый раз качается около 2 ГБ, через VPN может занять время)"
"$UV" pip install --python .venv/bin/python -e '.[dev]'

echo "== Команда clipper"
mkdir -p "$HOME/.local/bin"
ln -sf "$PROJECT/bin/clipper" "$HOME/.local/bin/clipper"
case ":$PATH:" in
  *":$HOME/.local/bin:"*) echo "Готово: команда clipper доступна.";;
  *) echo "Готово. Команда: $PROJECT/bin/clipper"
     echo "Чтобы писать просто clipper, выполните один раз:"
     echo "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc && source ~/.bashrc";;
esac

echo "== Проверка"
.venv/bin/python -c "import faster_whisper, ctranslate2, yt_dlp; print('faster-whisper', faster_whisper.__version__, '| ctranslate2', ctranslate2.__version__, '| видеокарт CUDA:', ctranslate2.get_cuda_device_count())"
