"""
Stage 4 — Export approved samples as a HuggingFace Dataset.

Reads stage3_scores.json, filters approved samples (auto_approved with no
human override + human-approved), and builds a typed HuggingFace Dataset.

Output: data/output/egyptian_arabic_ssdp/ + metadata.csv
Optional: push to HuggingFace Hub if hf_repo_id is configured.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from utils.checkpoint import load_checkpoint
from utils.logger import get_logger

log = get_logger(__name__)


def run(config: Dict[str, Any]) -> None:
    """
    Execute Stage 4: Export approved samples as HuggingFace dataset.

    Args:
        config: Pipeline configuration dictionary.
    """
    from datasets import Audio, Dataset, Features, Value

    output_dir = Path(config.get("output_dir", "data"))
    scores_path = output_dir / "manifests" / "stage3_scores.json"
    export_dir = output_dir / "output" / "egyptian_arabic_ssdp"
    sample_rate = config.get("sample_rate", 22050)
    hf_repo_id = config.get("hf_repo_id", "")

    # ── Load scores ───────────────────────────────
    scores = load_checkpoint(str(scores_path))
    if not scores:
        log.error(f"Stage 3 scores not found at {scores_path}. Run Stage 3 first.")
        raise FileNotFoundError(f"Scores not found: {scores_path}")

    # ── Filter approved samples ───────────────────
    # Approved = final_label == "approved" OR
    #            (auto_label == "auto_approved" AND final_label is None)
    approved = []
    for s in scores:
        fl = s.get("final_label")
        al = s.get("auto_label", "")
        if fl == "approved":
            approved.append(s)
        elif al == "auto_approved" and fl is None:
            approved.append(s)

    if not approved:
        log.warning("No approved samples to export.")
        print("No approved samples found. Review samples in Stage 3 first.")
        return

    log.info(f"Exporting {len(approved)} approved samples.")

    # ── Verify audio files exist ──────────────────
    valid = []
    for s in approved:
        ap = s.get("audio_path", "")
        if ap and Path(ap).exists():
            valid.append(s)
        else:
            log.warning(f"Audio missing for {s['id']}: {ap}")

    if not valid:
        log.error("No valid audio files found among approved samples.")
        return

    log.info(f"{len(valid)} samples with valid audio files.")

    # ── Load Stage 1 manifest for domain info ─────
    prompts = load_checkpoint(str(output_dir / "manifests" / "stage1_prompts.json"))
    domain_map = {}
    if prompts:
        domain_map = {p["id"]: p.get("domain", "general") for p in prompts}

    # ── Build dataset dict ────────────────────────
    records = {
        "audio": [],
        "text": [],
        "id": [],
        "domain": [],
        "duration": [],
        "wer": [],
    }

    for s in valid:
        records["audio"].append(s["audio_path"])
        records["text"].append(s["text"])
        records["id"].append(s["id"])
        records["domain"].append(domain_map.get(s["id"], "general"))
        records["duration"].append(float(s.get("duration_sec", 0)))
        records["wer"].append(float(s.get("wer", 0)))

    # ── Create HuggingFace Dataset ────────────────
    features = Features({
        "audio": Audio(sampling_rate=sample_rate),
        "text": Value("string"),
        "id": Value("string"),
        "domain": Value("string"),
        "duration": Value("float32"),
        "wer": Value("float32"),
    })

    ds = Dataset.from_dict(records, features=features)

    # ── Save locally ──────────────────────────────
    export_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(export_dir))
    log.info(f"Dataset saved to {export_dir}")

    # ── Export metadata CSV ───────────────────────
    csv_path = export_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "text", "audio_path", "domain", "duration", "wer"],
        )
        writer.writeheader()
        for s in valid:
            writer.writerow({
                "id": s["id"],
                "text": s["text"],
                "audio_path": s["audio_path"],
                "domain": domain_map.get(s["id"], "general"),
                "duration": s.get("duration_sec", 0),
                "wer": s.get("wer", 0),
            })
    log.info(f"Metadata CSV saved to {csv_path}")

    # ── Optional: push to HuggingFace Hub ─────────
    if hf_repo_id:
        try:
            log.info(f"Pushing dataset to HuggingFace Hub: {hf_repo_id}")
            ds.push_to_hub(hf_repo_id)
            log.info(f"Successfully pushed to {hf_repo_id}")
        except Exception as e:
            log.error(f"Failed to push to Hub: {e}")

    # ── Print final stats ─────────────────────────
    total = len(valid)
    avg_dur = sum(records["duration"]) / total if total else 0
    avg_wer = sum(records["wer"]) / total if total else 0
    domain_dist = Counter(records["domain"])

    print("\n" + "=" * 50)
    print("  SSDP Export — Final Statistics")
    print("=" * 50)
    print(f"  Total samples:     {total}")
    print(f"  Avg duration:      {avg_dur:.2f}s")
    print(f"  Avg WER:           {avg_wer:.3f} ({avg_wer*100:.1f}%)")
    print(f"  Domain distribution:")
    for domain, count in sorted(domain_dist.items()):
        print(f"    {domain:20s}: {count}")
    print("=" * 50)
