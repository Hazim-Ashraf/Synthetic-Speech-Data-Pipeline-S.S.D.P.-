"""
Stage 1b — Track B: Real Speech Harvesting from YouTube.

Downloads Egyptian Arabic speech from YouTube videos, transcribes and
force-aligns with WhisperX, splits into sentence-level segments, and
outputs a manifest identical in format to stage2_manifest.json.

WER scoring is deferred to Stage 3, which handles both Track A and
Track B samples uniformly through the same ASR backend.

All steps are checkpointed — a crash mid-pipeline resumes cleanly.
"""

import re
import sys

import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import get_logger

log = get_logger(__name__)


# ══════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════

def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _is_arabic(char: str) -> bool:
    """Check if a character is Arabic (includes Arabic script blocks)."""
    try:
        name = unicodedata.name(char, "")
        return "ARABIC" in name
    except ValueError:
        return False


def _arabic_ratio(text: str) -> float:
    """Calculate the ratio of Arabic characters in text (ignoring spaces/punct)."""
    chars = [c for c in text if not c.isspace() and unicodedata.category(c) not in ('Po', 'Ps', 'Pe', 'Pi', 'Pf')]
    if not chars:
        return 0.0
    arabic_count = sum(1 for c in chars if _is_arabic(c))
    return arabic_count / len(chars)


# ══════════════════════════════════════════════════
# Step 1 — Download audio via yt-dlp
# ══════════════════════════════════════════════════

def _download_youtube_audio(url: str, output_dir: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Download audio from a YouTube URL.

    Returns:
        Tuple of (video_id, title, audio_path) or (None, None, None) on failure.
    """
    import yt_dlp

    video_id = _extract_video_id(url)
    if not video_id:
        log.error(f"Could not extract video ID from URL: {url}")
        return None, None, None

    output_path = output_dir / f"{video_id}"
    final_wav = output_dir / f"{video_id}.wav"

    # Skip if already downloaded
    if final_wav.exists():
        log.info(f"Audio already exists for {video_id}, skipping download.")
        return video_id, "cached", str(final_wav)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        "quiet": False,
        "no_warnings": False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "unknown")
            duration = info.get("duration", 0)
            log.info(f"Downloaded: {title} ({duration}s) → {final_wav}")
            return video_id, title, str(final_wav)
    except Exception as e:
        log.error(f"Failed to download {url}: {e}")
        return None, None, None


# ══════════════════════════════════════════════════
# Step 2 — Transcribe + force-align via WhisperX
# ══════════════════════════════════════════════════

def _transcribe_and_align(
    audio_path: str,
    device: str = "cpu",
    model_size: str = "large-v3",
    cache_path: Optional[Path] = None,
) -> List[Dict]:
    """
    Transcribe and force-align audio using WhisperX.

    Returns list of segments: [{"text": "...", "start": 0.2, "end": 3.4, "words": [...]}, ...]
    """
    # Check cache first
    if cache_path and cache_path.exists():
        cached = load_checkpoint(str(cache_path))
        if cached:
            log.info(f"Loaded cached alignment from {cache_path} ({len(cached)} segments)")
            return cached

    import whisperx

    compute_type = "float32" if device == "cpu" else "float16"

    log.info(f"Loading WhisperX model ({model_size}) on {device}...")
    log.info("  (Note: If downloading weights or loading into CPU RAM, this may take a few minutes. Progress will be shown below.)")
    model = whisperx.load_model(
        model_size,
        device=device,
        language="ar",
        compute_type=compute_type,
    )

    log.info(f"Transcribing {audio_path}...")
    result = model.transcribe(str(audio_path), language="ar", print_progress=True)
    raw_segments = result.get("segments", [])
    log.info(f"Transcription complete: {len(raw_segments)} raw segments")

    # Force-align to get word-level timestamps
    try:
        log.info("Loading alignment model for Arabic...")
        model_a, metadata = whisperx.load_align_model(
            language_code="ar",
            device=device,
        )
        log.info("Force-aligning segments...")
        aligned = whisperx.align(
            raw_segments,
            model_a,
            metadata,
            str(audio_path),
            device,
            return_char_alignments=False,
            print_progress=True,
        )
        segments = aligned.get("segments", raw_segments)
        log.info(f"Alignment complete: {len(segments)} aligned segments")
    except Exception as e:
        log.warning(f"Force-alignment failed ({e}), using raw segments.")
        segments = raw_segments

    # Serialize segments (strip non-JSON-serializable objects)
    clean_segments = []
    for seg in segments:
        clean_seg = {
            "text": seg.get("text", "").strip(),
            "start": float(seg.get("start", 0)),
            "end": float(seg.get("end", 0)),
        }
        # Keep word-level timing if available
        if "words" in seg:
            clean_seg["words"] = [
                {"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0)}
                for w in seg["words"]
                if isinstance(w, dict) and "word" in w
            ]
        clean_segments.append(clean_seg)

    # Cache alignment results
    if cache_path:
        save_checkpoint(str(cache_path), clean_segments)
        log.info(f"Alignment cached to {cache_path}")

    return clean_segments


# ══════════════════════════════════════════════════
# Step 3 — Split audio into sentence segments
# ══════════════════════════════════════════════════

def _extract_segment(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
    target_sr: int = 22050,
) -> bool:
    """
    Extract a segment from audio, resample to target_sr mono, save as WAV.

    Returns True on success, False on failure.
    """
    from pydub import AudioSegment

    try:
        audio = AudioSegment.from_wav(str(audio_path))
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        segment = audio[start_ms:end_ms]
        segment = segment.set_frame_rate(target_sr)
        segment = segment.set_channels(1)  # mono
        segment.export(str(output_path), format="wav")
        return True
    except Exception as e:
        log.error(f"Failed to extract segment {output_path}: {e}")
        return False


def _filter_segment(
    seg: Dict,
    min_duration: float,
    max_duration: float,
    max_non_arabic: float,
) -> Tuple[bool, str]:
    """
    Check if a segment passes quality filters.

    Returns (passed, reason) tuple.
    """
    text = seg.get("text", "").strip()
    duration = seg.get("end", 0) - seg.get("start", 0)

    if not text:
        return False, "empty_text"
    if duration < min_duration:
        return False, f"too_short ({duration:.1f}s < {min_duration}s)"
    if duration > max_duration:
        return False, f"too_long ({duration:.1f}s > {max_duration}s)"

    ratio = _arabic_ratio(text)
    if ratio < (1.0 - max_non_arabic):
        return False, f"low_arabic ({ratio:.0%} < {1.0 - max_non_arabic:.0%})"

    return True, "ok"




# ══════════════════════════════════════════════════
# Main run function
# ══════════════════════════════════════════════════

def run(config: Dict[str, Any]) -> None:
    """
    Execute Stage 1b: Harvest real Egyptian Arabic speech from YouTube.

    Args:
        config: Pipeline configuration dictionary.
    """
    from tqdm import tqdm

    # ── Config ────────────────────────────────────
    if not config.get("track_b_enabled", False):
        log.info("Track B is disabled in config (track_b_enabled: false). Skipping.")
        return

    urls = config.get("youtube_urls", [])
    if not urls:
        log.info("No YouTube URLs configured. Skipping Track B.")
        return

    output_dir = Path(config.get("output_dir", "data"))
    sample_rate = config.get("sample_rate", 22050)
    min_seg_dur = config.get("min_segment_duration", 2.0)
    max_seg_dur = config.get("max_segment_duration", 15.0)
    max_non_arabic = config.get("max_non_arabic_ratio", 0.30)
    num_real_samples = config.get("num_real_samples", 100)
    whisperx_model = config.get("whisperx_model", "large-v3")
    whisperx_device = config.get("whisperx_device", "cpu")

    # ── Paths ─────────────────────────────────────
    raw_dir = output_dir / "audio" / "real" / "raw"
    seg_dir = output_dir / "audio" / "real"
    manifest_dir = output_dir / "manifests"

    raw_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    dl_progress_path = manifest_dir / "stage1b_download_progress.json"
    manifest_path = manifest_dir / "stage1b_manifest.json"

    # ── Load checkpoints ──────────────────────────
    dl_progress = load_checkpoint(str(dl_progress_path))
    if not isinstance(dl_progress, list):
        dl_progress = []
    downloaded_urls = {entry["url"] for entry in dl_progress if entry.get("status") == "done"}

    # ══════════════════════════════════════════════
    # STEP 1 — Download
    # ══════════════════════════════════════════════
    log.info(f"Track B: {len(urls)} URLs configured, {len(downloaded_urls)} already downloaded.")

    for url in urls:
        url = url.strip()
        if not url or url in downloaded_urls:
            continue

        log.info(f"Downloading: {url}")
        video_id, title, audio_path = _download_youtube_audio(url, raw_dir)

        if video_id and audio_path:
            dl_progress.append({
                "url": url,
                "video_id": video_id,
                "title": title,
                "audio_path": audio_path,
                "status": "done",
            })
            downloaded_urls.add(url)
            save_checkpoint(str(dl_progress_path), dl_progress)
            log.info(f"  ✓ {video_id}: {title}")
        else:
            dl_progress.append({
                "url": url,
                "video_id": None,
                "title": None,
                "audio_path": None,
                "status": "failed",
            })
            save_checkpoint(str(dl_progress_path), dl_progress)
            log.warning(f"  ✗ Failed to download: {url}")

    # Get successfully downloaded videos
    videos = [
        entry for entry in dl_progress
        if entry.get("status") == "done" and entry.get("audio_path")
    ]

    if not videos:
        log.warning("No videos downloaded successfully. Track B has nothing to process.")
        return

    log.info(f"Track B: {len(videos)} videos ready for transcription.")

    # ══════════════════════════════════════════════
    # STEP 2 — Transcribe + align
    # ══════════════════════════════════════════════
    all_segments_raw = []  # (video_entry, segment_dict) tuples

    for video in videos:
        vid = video["video_id"]
        audio_path = video["audio_path"]
        cache_path = manifest_dir / f"stage1b_alignment_{vid}.json"

        log.info(f"Transcribing video: {vid}")
        segments = _transcribe_and_align(
            audio_path,
            device=whisperx_device,
            model_size=whisperx_model,
            cache_path=cache_path,
        )
        log.info(f"  {vid}: {len(segments)} segments from WhisperX")

        for seg in segments:
            all_segments_raw.append((video, seg))

    log.info(f"Track B: {len(all_segments_raw)} total raw segments across all videos.")

    # ══════════════════════════════════════════════
    # STEP 3 — Filter + extract segments
    # ══════════════════════════════════════════════
    kept_segments = []  # (segment_id, video_entry, segment_dict, audio_path)
    filtered_stats = {"total": 0, "kept": 0, "skipped": {}}

    global_idx = 0
    for video, seg in all_segments_raw:
        filtered_stats["total"] += 1
        vid = video["video_id"]

        passed, reason = _filter_segment(seg, min_seg_dur, max_seg_dur, max_non_arabic)
        if not passed:
            filtered_stats["skipped"][reason] = filtered_stats["skipped"].get(reason, 0) + 1
            continue

        segment_id = f"real_{vid}_{global_idx:04d}"
        seg_audio_path = seg_dir / f"{segment_id}.wav"

        # Skip extraction if audio already exists
        if seg_audio_path.exists():
            kept_segments.append((segment_id, video, seg, str(seg_audio_path)))
            filtered_stats["kept"] += 1
            global_idx += 1
            continue

        # Extract segment audio
        ok = _extract_segment(
            video["audio_path"],
            seg["start"],
            seg["end"],
            str(seg_audio_path),
            target_sr=sample_rate,
        )

        if ok:
            kept_segments.append((segment_id, video, seg, str(seg_audio_path)))
            filtered_stats["kept"] += 1
        else:
            filtered_stats["skipped"]["extraction_error"] = filtered_stats["skipped"].get("extraction_error", 0) + 1

        global_idx += 1

        # Stop early if we have enough
        if len(kept_segments) >= num_real_samples:
            log.info(f"Reached target of {num_real_samples} segments. Stopping extraction.")
            break

    log.info(f"Segment filtering: {filtered_stats['kept']} kept out of {filtered_stats['total']} total")
    for reason, count in filtered_stats["skipped"].items():
        log.info(f"  Skipped ({reason}): {count}")

    if not kept_segments:
        log.warning("No segments passed filtering. Track B manifest will be empty.")
        save_checkpoint(str(manifest_path), [])
        return

    # ══════════════════════════════════════════════
    # STEP 4 — Write Track B manifest
    # (WER scoring is deferred to Stage 3)
    # ══════════════════════════════════════════════
    manifest = []

    for sid, video, seg, ap in kept_segments:
        manifest.append({
            "id": sid,
            "text": seg.get("text", "").strip(),
            "audio_path": ap,
            "duration_sec": round(seg.get("end", 0) - seg.get("start", 0), 2),
            "status": "done",
            "source": "real",
            "video_id": video["video_id"],
            "youtube_url": video["url"],
        })

    save_checkpoint(str(manifest_path), manifest)

    # ── Final stats ───────────────────────────────
    log.info("=" * 50)
    log.info("Track B — Harvest Summary")
    log.info("=" * 50)
    log.info(f"  Videos downloaded:   {len(videos)}")
    log.info(f"  Total raw segments:  {len(all_segments_raw)}")
    log.info(f"  Segments after filter: {len(kept_segments)}")
    log.info(f"  Manifest written to: {manifest_path}")
    log.info("  (WER scoring will be handled by Stage 3)")
    log.info("=" * 50)
