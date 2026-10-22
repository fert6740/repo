"""Разложить скрины заданий по номерам: Screenshots/<Предмет>/<метка>/NN.<ext>.

Использование:
    python3 -I prepare_screens.py <предмет> <архив.zip | папка | картинка ...> [--repo ПУТЬ] [--label МЕТКА]

Номер задания берётся из последнего числа в имени файла («Задание 07.png» -> 7).
Если номера не у всех файлов или они повторяются — файлы сортируются по имени,
нумеруются подряд и печатается предупреждение (тогда номера надо сверить по картинкам).
В конце печатается JSON: папка назначения и соответствие «номер -> исходный файл».
"""
import argparse
import datetime
import json
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def collect(sources, tmp):
    files = []
    for src in map(Path, sources):
        if src.is_dir():
            files += [p for p in src.rglob("*") if p.is_file()]
        elif src.suffix.lower() == ".zip":
            out = tmp / f"zip{len(list(tmp.iterdir()))}"
            with zipfile.ZipFile(src) as z:
                for info in z.infolist():
                    name = Path(info.filename)
                    # пропускаем папки, мусор macOS и всё, что пытается выйти за пределы папки
                    if info.is_dir() or "__MACOSX" in name.parts or ".." in name.parts or name.is_absolute():
                        continue
                    target = out / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(info))
                    files.append(target)
        elif src.is_file():
            files.append(src)
        else:
            sys.exit(f"Не найдено: {src}")
    return [p for p in files if p.suffix.lower() in IMAGE_EXT and not p.name.startswith(".")]


def number_of(path):
    nums = re.findall(r"\d+", path.stem)
    return int(nums[-1]) if nums else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subject")
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--label", default=datetime.date.today().isoformat())
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as t:
        images = collect(args.sources, Path(t))
        if not images:
            sys.exit("Картинок не найдено")

        nums = [number_of(p) for p in images]
        warning = None
        if None in nums or len(set(nums)) != len(nums):
            images.sort(key=lambda p: p.name)
            nums = list(range(1, len(images) + 1))
            warning = "Номера в именах файлов отсутствуют или повторяются: пронумеровано подряд по имени, сверь номера по самим скринам"
        pairs = sorted(zip(nums, images))

        base = Path(args.repo) / "Screenshots" / args.subject
        dest, i = base / args.label, 2
        while dest.exists():  # вторая контрольная по тому же предмету в тот же день
            dest, i = base / f"{args.label}_{i}", i + 1
        dest.mkdir(parents=True)

        width = max(2, len(str(pairs[-1][0])))
        mapping = {}
        for n, src in pairs:
            name = f"{n:0{width}d}{src.suffix.lower()}"
            shutil.copyfile(src, dest / name)
            mapping[n] = {"file": str((dest / name).relative_to(args.repo)), "source": src.name}

    expected = set(range(1, pairs[-1][0] + 1))
    missing = sorted(expected - set(mapping))
    print(json.dumps({"dest": str(dest.relative_to(args.repo)), "count": len(mapping),
                      "missing_numbers": missing, "warning": warning, "tasks": mapping},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
