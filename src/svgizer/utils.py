import logging
import sys
from pathlib import Path


def setup_logger(level: str, log_file: Path | str | None = None) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)

    if log_file is not None:
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    else:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)
