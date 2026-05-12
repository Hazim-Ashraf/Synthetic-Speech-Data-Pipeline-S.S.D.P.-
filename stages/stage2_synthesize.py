"""
Stage 2 — TTS Synthesis using NAMAA-Space/NAMAA-Egyptian-TTS.

Converts Egyptian Arabic text prompts into speech audio files using the
NAMAA Egyptian TTS model via chatterbox. Features resumable checkpointing
and retry with exponential backoff.

Input:  data/manifests/stage1_prompts.json
Output: data/audio/{id}.wav + data/manifests/stage2_manifest.json
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from tqdm import tqdm

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import get_logger

log = get_logger(__name__)


def _load_model(config: Dict[str, Any]):
    """
    Load the NAMAA Egyptian TTS model.

    Downloads the model checkpoint from HuggingFace and loads it
    onto the specified device.

    Args:
        config: Pipeline config dict.

    Returns:
        Loaded TTS model ready for inference.
    """
    device = config.get("tts_device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
        
    hf_token = config.get("hf_token", "") or os.environ.get("HF_TOKEN", "")

    log.info(f"Loading NAMAA-Egyptian-TTS model on device={device}...")

    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file as load_safetensors
    from chatterbox import mtl_tts

    ckpt_dir = snapshot_download(
        repo_id="NAMAA-Space/NAMAA-Egyptian-TTS",
        token=hf_token if hf_token else None,
    )
    log.info(f"Model checkpoint downloaded to: {ckpt_dir}")

    model = mtl_tts.ChatterboxMultilingualTTS.from_pretrained(device=device)

    t3_state = load_safetensors(
        f"{ckpt_dir}/t3_mtl23ls_v2.safetensors", device=device
    )
    model.t3.load_state_dict(t3_state)
    model.t3.to(device).eval()

    log.info("NAMAA-Egyptian-TTS model loaded successfully.")
    return model


def _synthesize_single(
    model,
    text: str,
    output_path: Path,
    sample_rate: int,
    max_retries: int = 3,
) -> Optional[float]:
    """
    Synthesize a single text sample to audio with retry logic.

    Args:
        model: Loaded TTS model.
        text: Arabic text to synthesize.
        output_path: Path to save the .wav file.
        sample_rate: Audio sample rate.
        max_retries: Maximum retry attempts.

    Returns:
        Duration in seconds if successful, None if all retries failed.
    """
    for attempt in range(1, max_retries + 1):
        try:
            wav = model.generate(text, language_id="ar")

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(str(output_path), wav, model.sr)

            # Calculate duration
            duration = wav.shape[-1] / model.sr
            return round(duration, 2)

        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
            log.warning(
                f"Synthesis attempt {attempt}/{max_retries} failed for "
                f"'{text[:40]}...': {e}. "
                f"{'Retrying in ' + str(wait_time) + 's...' if attempt < max_retries else 'Giving up.'}"
            )
            if attempt < max_retries:
                time.sleep(wait_time)

    return None


def run(config: Dict[str, Any]) -> None:
    """
    Execute Stage 2: Synthesize audio from text prompts using NAMAA TTS.

    Args:
        config: Pipeline configuration dictionary.
    """
    output_dir = Path(config.get("output_dir", "data"))
    device = config.get("tts_device", "cpu")
    batch_size = config.get("batch_size", 10)
    sample_rate = config.get("sample_rate", 22050)
    audio_format = config.get("audio_format", "wav")
    max_retries = config.get("tts_max_retries", 3)

    prompts_path = output_dir / "manifests" / "stage1_prompts.json"
    progress_path = output_dir / "manifests" / "stage2_progress.json"
    manifest_path = output_dir / "manifests" / "stage2_manifest.json"
    audio_dir = output_dir / "audio"

    audio_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prompts from Stage 1 ─────────────────
    prompts_data = load_checkpoint(str(prompts_path))
    if not prompts_data:
        log.error(
            f"No Stage 1 prompts found at {prompts_path}. Run Stage 1 first."
        )
        raise FileNotFoundError(f"Stage 1 manifest not found: {prompts_path}")

    log.info(f"Stage 2: {len(prompts_data)} prompts to synthesize.")

    # ── Load progress checkpoint ──────────────────
    progress = load_checkpoint(str(progress_path))
    
    # Check if Stage 1 prompts are newer than our Stage 2 progress.
    # If so, we need a completely fresh run to avoid mixing old data.
    is_fresh_run = False
    if not isinstance(progress, dict) or not progress:
        is_fresh_run = True
    elif prompts_path.exists() and progress_path.exists():
        if prompts_path.stat().st_mtime > progress_path.stat().st_mtime:
            log.info("Stage 1 prompts are newer than Stage 2 progress. Forcing a fresh run.")
            progress = {}
            is_fresh_run = True

    if is_fresh_run:
        # If progress is empty or forced fresh, clean the audio directory
        # so old files don't mix with the new ones.
        log.info("Fresh run detected. Cleaning old audio files...")
        for old_file in audio_dir.glob(f"*.{audio_format}"):
            try:
                old_file.unlink()
            except Exception as e:
                log.warning(f"Could not delete old file {old_file}: {e}")

    done_ids = {pid for pid, info in progress.items() if info.get("status") == "done"}
    pending = [p for p in prompts_data if p["id"] not in done_ids]

    log.info(
        f"{len(done_ids)} already synthesized, {len(pending)} remaining."
    )

    if not pending:
        log.info("All samples already synthesized. Skipping to manifest export.")
    else:
        # ── Load model ────────────────────────────────
        model = _load_model(config)

        # ── Process in batches ────────────────────────
        for batch_start in tqdm(
            range(0, len(pending), batch_size),
            desc="Synthesizing",
            unit="batch",
        ):
            batch = pending[batch_start : batch_start + batch_size]

            for sample in batch:
                sid = sample["id"]
                text = sample["text"]
                out_path = audio_dir / f"{sid}.{audio_format}"

                log.debug(f"Synthesizing {sid}: {text[:50]}...")

                duration = _synthesize_single(
                    model, text, out_path, sample_rate, max_retries
                )

                if duration is not None:
                    progress[sid] = {
                        "status": "done",
                        "duration": duration,
                        "audio_path": str(out_path),
                    }
                    log.debug(f"  ✓ {sid}: {duration}s")
                else:
                    progress[sid] = {
                        "status": "failed",
                        "duration": 0,
                        "audio_path": "",
                    }
                    log.error(f"  ✗ {sid}: FAILED after {max_retries} retries")

            # Checkpoint after each batch
            save_checkpoint(str(progress_path), progress)

    # ── Build final manifest ──────────────────────
    manifest = []
    failed_ids = []

    for sample in prompts_data:
        sid = sample["id"]
        info = progress.get(sid, {})
        status = info.get("status", "pending")

        if status == "done":
            manifest.append({
                "id": sid,
                "text": sample["text"],
                "audio_path": info.get("audio_path", f"{audio_dir}/{sid}.{audio_format}"),
                "duration_sec": info.get("duration", 0),
                "status": "done",
            })
        else:
            failed_ids.append(sid)
            manifest.append({
                "id": sid,
                "text": sample["text"],
                "audio_path": "",
                "duration_sec": 0,
                "status": "failed",
            })

    save_checkpoint(str(manifest_path), manifest)

    # ── Log summary ───────────────────────────────
    done_count = sum(1 for m in manifest if m["status"] == "done")
    fail_count = len(failed_ids)
    avg_duration = (
        sum(m["duration_sec"] for m in manifest if m["status"] == "done") / done_count
        if done_count > 0
        else 0
    )

    log.info(f"Stage 2 complete: {done_count} done, {fail_count} failed.")
    log.info(f"Average duration: {avg_duration:.2f}s")
    log.info(f"Manifest saved to: {manifest_path}")

    if failed_ids:
        log.warning(f"Failed sample IDs: {failed_ids}")
