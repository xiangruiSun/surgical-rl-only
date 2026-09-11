from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    *,
    level: str | int = "INFO",
    log_file: str | os.PathLike[str] | None = None,
    force: bool = True,
) -> None:
    """
    Configure stdlib logging with a consistent format.

    If `log_file` is set, logs are written to both stderr and the file.
    """
    numeric_level = level
    if isinstance(level, str):
        numeric_level = logging.getLevelName(level.upper())
        if not isinstance(numeric_level, int):
            numeric_level = logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        p = Path(log_file).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(p))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=force,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)

