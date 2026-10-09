"""Команды clipper."""
from __future__ import annotations

import argparse
import logging
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import PROJECT_DIR, ClipperError, Config, load_config
from .record import Record, find_record, fmt_duration, parse_vod_id

log = logging.getLogger("clipper")

LATER = {
    "scenes": 3, "scenes-setup": 3, "plan": 2, "render": 2, "review": 5, "clean": 2,
}


def setup_logging(verbose: bool) -> None:
    root = logging.getLogger("clipper")
    root.setLevel(logging.DEBUG)
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG if verbose else logging.INFO)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)


def get_record(cfg: Config, ref: str, streamer: str | None, create: bool = False) -> Record:
    if create and not Path(ref).expanduser().is_dir():
        vod = parse_vod_id(ref)
        if not vod:
            raise ClipperError(f"Не похоже на ссылку на запись Twitch: {ref}")
        if not streamer:
            try:
                return find_record(cfg, ref)
            except ClipperError:
                raise ClipperError("Укажите имя стримера: --streamer <имя> (так будет называться его папка).")
        rec = Record(cfg, streamer, vod)
        rec.create()
    else:
        rec = find_record(cfg, ref, streamer)
    rec.attach_log()
    return rec


def cmd_step(args, cfg: Config) -> None:
    rec = get_record(cfg, args.record, args.streamer, create=args.cmd == "download")
    if args.cmd == "download":
        from .download import run_download
        run_download(rec, args.force)
        print(f"Готово: {rec.dir}")
    elif args.cmd == "transcribe":
        from .transcribe import run_transcribe
        run_transcribe(rec, args.force)
        print(f"Готово: {rec.transcript}")
    elif args.cmd == "signals":
        from .signals import run_signals
        run_signals(rec, args.force)
        print(f"Готово: {rec.signals}")
    elif args.cmd == "topics":
        from .report import open_in_browser, topics_page
        from .topics import run_topics
        run_topics(rec, args.force, args.model)
        page = topics_page(rec)
        print(f"Готово: {rec.topics}\nСтраница: {page}")
        if not args.no_open:
            open_in_browser(page)


def cmd_run(args, cfg: Config) -> None:
    from .download import run_download
    from .report import open_in_browser, topics_page
    from .signals import run_signals
    from .topics import run_topics
    from .transcribe import run_transcribe
    from .usage import report

    t0 = time.time()
    rec = get_record(cfg, args.url, args.streamer, create=True)
    log.info("Запись %s, стример %s, папка %s", rec.vod_id, rec.streamer, rec.dir)
    run_download(rec, args.force)
    run_transcribe(rec, args.force)
    run_signals(rec, args.force)
    run_topics(rec, args.force, args.model)
    page = topics_page(rec)
    log.info("\n%s", report(rec))
    print(f"\nВсё готово за {fmt_duration(time.time() - t0)}.\nСписок тем: {page}")
    if not args.no_open:
        open_in_browser(page)


def cmd_page(args, cfg: Config) -> None:
    from .report import open_in_browser, topics_page
    rec = get_record(cfg, args.record, args.streamer)
    page = topics_page(rec)
    print(f"Страница: {page}")
    if not args.no_open:
        open_in_browser(page)


def cmd_usage(args, cfg: Config) -> None:
    from .usage import report
    rec = find_record(cfg, args.record, args.streamer)
    print(report(rec))


def cmd_compare(args, cfg: Config) -> None:
    from .report import compare_models, open_in_browser
    rec = get_record(cfg, args.record, args.streamer)
    models = args.models.split(",") if args.models else cfg["llm"]["compare_models"]
    with rec.step("compare-models"):
        page = compare_models(rec, models, args.force)
    print(f"Страница сравнения: {page}\nРасход: clipper usage {rec.vod_id}")
    if not args.no_open:
        open_in_browser(page)


def cmd_probe(args, cfg: Config) -> None:
    from .probe import run_probe
    sizes = [int(x) for x in args.sizes.split(",")]
    models = args.info_models.split(",") if args.info_models else cfg["llm"]["compare_models"]
    run_probe(cfg, args.model, sizes, models)


def cmd_bg(args, cfg: Config) -> None:
    if not shutil.which("tmux"):
        raise ClipperError("Не найден tmux: sudo pacman -S tmux")
    if not args.rest:
        raise ClipperError("Укажите команду, например: clipper bg run <ссылка> --streamer <имя>")
    wrapper = PROJECT_DIR / "bin" / "clipper"
    launcher = [str(wrapper)] if wrapper.exists() else [sys.executable, "-m", "clipper"]
    name = "clipper-" + datetime.now().strftime("%m%d-%H%M%S")
    logf = cfg.data_dir / "bg-logs" / f"{name}.log"
    logf.parent.mkdir(parents=True, exist_ok=True)
    inner = (f"cd {shlex.quote(str(PROJECT_DIR))} && {shlex.join(launcher + args.rest)} 2>&1 | "
             f"tee -a {shlex.quote(str(logf))}; echo; echo 'Закончено. Enter — закрыть окно.'; read")
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "bash", "-lc", inner], check=True)
    print(f"Запущено в фоне (переживёт закрытие окна): {' '.join(args.rest)}\n"
          f"Посмотреть: tmux attach -t {name}   (выйти, не прерывая: Ctrl+B, затем D)\n"
          f"Журнал: {logf}")


def cmd_later(args, cfg: Config) -> None:
    print(f"Команда «{args.cmd}» появится на этапе {LATER[args.cmd]}.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clipper", description="Нарезки из записей стримов Twitch")
    p.add_argument("-v", "--verbose", action="store_true", help="подробный вывод")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="команда")

    r = sub.add_parser("run", help="всё до списка тем: скачать, распознать, найти темы")
    r.add_argument("url")
    r.add_argument("--streamer", required=True, help="имя стримера (папка его настроек)")
    r.add_argument("--model", help="модель для прохода 1 (по умолчанию из config.toml)")
    r.add_argument("--force", action="store_true", help="пересчитать всё заново")
    r.add_argument("--no-open", action="store_true", help="не открывать страницу в браузере")
    r.set_defaults(fn=cmd_run)

    for name, hlp in (("download", "скачать запись, чат, сделать proxy и звук"),
                      ("transcribe", "распознать речь"), ("signals", "посчитать сигналы"),
                      ("topics", "найти темы (проход 1)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("record", help="ссылка, номер записи или путь к папке")
        s.add_argument("--streamer")
        s.add_argument("--force", action="store_true")
        if name == "topics":
            s.add_argument("--model")
            s.add_argument("--no-open", action="store_true")
        s.set_defaults(fn=cmd_step)

    s = sub.add_parser("page", help="пересобрать страницу со списком тем")
    s.add_argument("record"); s.add_argument("--streamer"); s.add_argument("--no-open", action="store_true")
    s.set_defaults(fn=cmd_page)

    s = sub.add_parser("usage", help="расход токенов и времени")
    s.add_argument("record"); s.add_argument("--streamer")
    s.set_defaults(fn=cmd_usage)

    s = sub.add_parser("compare-models", help="сравнить модели на проходе 1")
    s.add_argument("record"); s.add_argument("--streamer")
    s.add_argument("--models", help="через запятую, по умолчанию haiku,sonnet,opus")
    s.add_argument("--force", action="store_true"); s.add_argument("--no-open", action="store_true")
    s.set_defaults(fn=cmd_compare)

    s = sub.add_parser("llm-probe", help="проверить, какого размера запрос проходит через claude -p")
    s.add_argument("--model", default="haiku", help="модель для длинных запросов (по умолчанию haiku — дешевле)")
    s.add_argument("--sizes", default="50000,150000,300000", help="размеры в токенах через запятую")
    s.add_argument("--info-models", help="модели, у которых узнать окно контекста")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("bg", help="запустить команду в фоне (tmux), например: clipper bg run <ссылка> --streamer x")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_bg)

    for name in LATER:
        s = sub.add_parser(name, help=f"(этап {LATER[name]})")
        s.add_argument("rest", nargs=argparse.REMAINDER)
        s.set_defaults(fn=cmd_later)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        cfg = load_config()
        args.fn(args, cfg)
    except ClipperError as e:
        log.error("Ошибка: %s", e)
        return 1
    except KeyboardInterrupt:
        log.error("Прервано. Повторный запуск продолжит с того же места.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
