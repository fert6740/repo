# stream-clipper

Нарезки из записей стримов Twitch: длинные ролики по темам и шорты. Полное задание — в [TZ.md](TZ.md).

Сейчас готов **этап 1 — «темы без видео»**: скачивание записи и чата, распознавание речи, сигналы
(громкость, речь, чат), поиск тем нейросетью, страница со списком тем и шортов, учёт расхода токенов,
сравнение моделей.

## Установка (один раз)

```bash
git clone <адрес репозитория> ~/stream-clipper
cd ~/stream-clipper
./install.sh
```

Скрипт ничего не ставит в систему: `uv`, Python 3.12 и все библиотеки (yt-dlp, faster-whisper, CUDA-библиотеки
для видеокарты) ложатся в папку проекта (`.bootstrap/`, `.venv/`) и в кэш `uv`. Нужны уже установленные
`ffmpeg`, `tmux`, `python3`, `claude`.

## Порядок работы на этапе 1

```bash
# 1. Какого размера запрос проходит через claude -p (немного тратит лимит, в основном на дешёвой haiku)
clipper llm-probe

# 2. Всё до списка тем. Долго (скачивание + распознавание) — поэтому в фоне, переживёт закрытие окна:
clipper bg run https://www.twitch.tv/videos/<номер> --streamer <имя>
#    посмотреть ход работы: tmux attach -t clipper-...  (выйти, не прерывая: Ctrl+B, затем D)

# 3. Страница со списком тем откроется сама; ещё раз открыть/пересобрать:
clipper page <номер>

# 4. Сколько потрачено токенов и времени:
clipper usage <номер>

# 5. Сравнить модели haiku / sonnet / opus на поиске тем:
clipper compare-models <номер>
```

Отдельные шаги: `clipper download|transcribe|signals|topics <номер>`. Повторный запуск не пересчитывает
готовое (`--force` — пересчитать); прерванное скачивание и распознавание продолжаются с места остановки.

## Где что лежит

```
config.toml                    общие настройки (все пороги, модели, папка данных)
streamers/<имя>/streamer.toml  настройки стримера (создаётся сам из streamers/_template)
data/<имя>/<номер>/            всё по одной записи
  source.mp4 proxy.mp4 audio.wav chat.json info.json muted.json
  transcript.json signals.parquet topics.json topics.html
  llm_usage.jsonl timings.json log.txt
  compare/compare.html         сравнение моделей
```

Папку `data` можно перенести на другой диск: `data_dir` в `config.toml`.
