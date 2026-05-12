#!/usr/bin/env python3
"""
SSDP — Synthetic Speech Data Pipeline for Egyptian Arabic.

CLI entrypoint for running pipeline stages individually or all together.
Auto-creates a virtual environment and installs dependencies on first run.

Usage:
    python main.py --stage 1          # Generate text prompts via Nile-Chat-4B
    python main.py --stage 2          # Synthesize audio via NAMAA TTS
    python main.py --stage 3          # Auto-score + launch Streamlit review
    python main.py --stage 4          # Export approved samples as HF dataset
    python main.py --all              # Run stages 1→2→auto-score→launch UI
"""

# ══════════════════════════════════════════════════
# AUTO-BOOTSTRAP: venv creation + dependency install
# This runs BEFORE any third-party imports so it works
# even when nothing is installed yet.
# ══════════════════════════════════════════════════

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"


def _in_venv() -> bool:
    """Check if we are currently running inside our project venv."""
    return (
        hasattr(sys, "prefix")
        and sys.prefix != sys.base_prefix
        and Path(sys.prefix).resolve() == VENV_DIR.resolve()
    )


def _venv_python() -> str:
    """Return the path to the Python binary inside the venv."""
    return str(VENV_DIR / "bin" / "python")


def _bootstrap():
    """
    Ensure we are running inside the project venv with all deps installed.

    If not in venv:
      1. Create .venv if it doesn't exist
      2. Install/upgrade requirements.txt into the venv (skipped if already done)
      3. Re-exec the same command inside the venv's Python
      4. Exit the outer process with the inner process's return code

    Uses a marker file (.venv/.deps_installed) to avoid re-checking
    all dependencies on every run. Marker is invalidated when
    requirements.txt changes.
    """
    if _in_venv():
        return  # Already inside our venv, proceed normally

    venv_python = _venv_python()
    marker = VENV_DIR / ".deps_installed"
    needs_install = False

    # ── Step 1: Create venv ───────────────────────
    if not VENV_DIR.exists():
        print("=" * 60)
        print("  SSDP — First-Time Environment Setup")
        print("=" * 60)
        print(f"\n→ Creating virtual environment at {VENV_DIR}...")
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            cwd=str(PROJECT_ROOT),
        )
        print("  ✓ Virtual environment created.")
        needs_install = True
    else:
        # Check if requirements.txt changed since last install
        if REQUIREMENTS.exists() and marker.exists():
            req_mtime = REQUIREMENTS.stat().st_mtime
            marker_mtime = marker.stat().st_mtime
            if req_mtime > marker_mtime:
                print("→ requirements.txt changed, updating dependencies...")
                needs_install = True
        elif not marker.exists():
            needs_install = True

    # ── Step 2: Install dependencies (only if needed) ─
    if needs_install:
        print("=" * 60)
        print("  SSDP — Installing Dependencies")
        print("=" * 60)

        # Upgrade pip
        print("\n→ Upgrading pip inside venv...")
        subprocess.check_call(
            [venv_python, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if REQUIREMENTS.exists():
            print(f"\n→ Installing dependencies from {REQUIREMENTS.name}...")
            print("  (This may take several minutes on first run)\n")
            subprocess.check_call(
                [venv_python, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
                cwd=str(PROJECT_ROOT),
            )
            print("\n  ✓ All dependencies installed into .venv")

            # Write marker so we skip this next time
            marker.write_text("installed")
        else:
            print(f"\n⚠ requirements.txt not found at {REQUIREMENTS}")

    # ── Step 3: Re-exec inside the venv ───────────
    print(f"\n→ Launching inside venv...")
    print("=" * 60 + "\n")

    result = subprocess.run(
        [venv_python] + sys.argv,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "VIRTUAL_ENV": str(VENV_DIR)},
    )
    sys.exit(result.returncode)


# Run bootstrap before anything else
_bootstrap()

# ══════════════════════════════════════════════════
# From here on, we are guaranteed to be inside .venv
# with all dependencies available.
# ══════════════════════════════════════════════════

# ── Load .env file (tokens & secrets) ────────────
# This reads HF_TOKEN, NGROK_AUTHTOKEN etc. from .env
# so you never have to manually export them.
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    with open(_env_path, "r") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                os.environ.setdefault(_key.strip(), _val.strip())

from typing import Optional

import typer
import yaml

sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import get_logger

log = get_logger(__name__)

app = typer.Typer(
    name="ssdp",
    help="Synthetic Speech Data Pipeline for Egyptian Arabic STT fine-tuning.",
    add_completion=False,
)


def _load_config() -> dict:
    """Load pipeline configuration from config.yaml.

    If hf_token is empty in config, injects it from the HF_TOKEN
    environment variable (loaded from .env).
    """
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Missing config: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Inject HF token from environment if not set in config
    if not config.get("hf_token"):
        config["hf_token"] = os.environ.get("HF_TOKEN", "")

    log.info(f"Config loaded from {config_path}")
    return config


def _run_stage1(config: dict) -> None:
    """Stage 1: Generate Egyptian Arabic text prompts via Nile-Chat-4B."""
    log.info("=" * 50)
    log.info("STAGE 1 — Text Generation (Nile-Chat-4B)")
    log.info("=" * 50)
    from stages.stage1_generate import run
    run(config)


def _run_stage2(config: dict) -> None:
    """Stage 2: Synthesize speech audio via NAMAA TTS."""
    log.info("=" * 50)
    log.info("STAGE 2 — TTS Synthesis")
    log.info("=" * 50)
    from stages.stage2_synthesize import run
    run(config)


def _run_stage3(config: dict) -> None:
    """Stage 3: Auto-score then launch Streamlit review app."""
    log.info("=" * 50)
    log.info("STAGE 3 — Review & Scoring")
    log.info("=" * 50)

    # Phase A: auto-scoring
    from stages.stage3_review import run_auto_scoring
    run_auto_scoring(config)

    # Phase B: launch Streamlit UI
    log.info("Launching Streamlit review app... (Press Ctrl+C in terminal when done reviewing)")
    streamlit_path = PROJECT_ROOT / "stages" / "stage3_review.py"
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(streamlit_path)],
            cwd=str(PROJECT_ROOT),
        )
    except KeyboardInterrupt:
        log.info("Streamlit review app closed by user. Proceeding...")


def _run_stage4(config: dict) -> None:
    """Stage 4: Export approved samples as HuggingFace dataset."""
    log.info("=" * 50)
    log.info("STAGE 4 — Dataset Export")
    log.info("=" * 50)
    from stages.stage4_export import run
    run(config)


@app.command()
def main(
    stage: Optional[int] = typer.Option(
        None,
        "--stage",
        "-s",
        help="Run a specific stage (1-4).",
        min=1,
        max=4,
    ),
    all_stages: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Run all stages sequentially (1→2→auto-score→UI).",
    ),
) -> None:
    """
    Run the SSDP pipeline stages.

    Stages:
      1 = Text generation (Nile-Chat-4B)
      2 = TTS synthesis (NAMAA Egyptian TTS)
      3 = Auto-scoring + Streamlit review app
      4 = Dataset export (HuggingFace format)
    """
    if stage is None and not all_stages:
        typer.echo("Error: specify --stage N or --all. Use --help for info.")
        raise typer.Exit(1)

    config = _load_config()

    try:
        if all_stages:
            log.info("Running ALL stages sequentially.")
            _run_stage1(config)
            _run_stage2(config)
            _run_stage3(config)  # includes auto-score + UI
            _run_stage4(config)  # export the dataset after UI is closed
        elif stage == 1:
            _run_stage1(config)
        elif stage == 2:
            _run_stage2(config)
        elif stage == 3:
            _run_stage3(config)
        elif stage == 4:
            _run_stage4(config)

        log.info("Pipeline execution complete.")

    except KeyboardInterrupt:
        log.warning("Pipeline interrupted by user.")
        raise typer.Exit(130)
    except Exception as e:
        log.error(f"Pipeline failed: {e}", exc_info=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
