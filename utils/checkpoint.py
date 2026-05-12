"""
Checkpoint utilities for resumable pipeline stages.

All writes are atomic (write to temp file, then rename) to prevent
corruption from crashes mid-write.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from utils.logger import get_logger

log = get_logger(__name__)


def load_checkpoint(path: str) -> Any:
    """
    Load a JSON checkpoint file.

    Args:
        path: Path to the checkpoint JSON file.

    Returns:
        Parsed JSON data, or None if file doesn't exist or is corrupt.
    """
    p = Path(path)
    if not p.exists():
        log.info(f"No checkpoint found at {path}, starting fresh.")
        return None

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        log.info(f"Loaded checkpoint from {path} ({_describe(data)} entries)")
        return data
    except (json.JSONDecodeError, IOError) as e:
        log.warning(f"Corrupt checkpoint at {path}: {e}. Starting fresh.")
        return None


def save_checkpoint(path: str, data: Any) -> None:
    """
    Atomically save data as a JSON checkpoint.

    Writes to a temp file in the same directory, then renames.
    This guarantees the checkpoint is never half-written.

    Args:
        path: Destination path for the checkpoint file.
        data: JSON-serializable data to save.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file in same directory (same filesystem → atomic rename)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(p.parent), suffix=".tmp", prefix=".ckpt_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, str(p))
        log.debug(f"Checkpoint saved to {path} ({_describe(data)} entries)")
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _describe(data: Any) -> str:
    """Return a short description of checkpoint data size."""
    if isinstance(data, list):
        return str(len(data))
    elif isinstance(data, dict):
        return str(len(data))
    return "1"
