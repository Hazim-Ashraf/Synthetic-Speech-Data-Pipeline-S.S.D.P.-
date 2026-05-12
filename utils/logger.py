"""
Structured logging for the SSDP pipeline.

Provides dual-output logging:
  - Console: colored, human-readable via Rich
  - File: structured with timestamps to logs/pipeline.log
"""

import logging
import os
from pathlib import Path
from datetime import datetime


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Create a structured logger with console + file handlers.

    Args:
        name: Logger name (typically __name__ of calling module).
        log_dir: Directory for log files. Created if it doesn't exist.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── File handler ──────────────────────────────
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "pipeline.log"

    file_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # ── Console handler ───────────────────────────
    console_fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    return logger
