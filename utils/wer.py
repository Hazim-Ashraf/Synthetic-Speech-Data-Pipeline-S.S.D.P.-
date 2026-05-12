"""
WER (Word Error Rate) and CER (Character Error Rate) scoring utilities.

Uses the jiwer library for computation. Provides both single-pair and
batch scoring interfaces with Arabic text normalization.
"""

import re
from typing import List, Optional, Tuple

import jiwer

from utils.logger import get_logger

log = get_logger(__name__)


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for fair WER comparison.

    Normalizations applied:
      - Strip diacritics (tashkeel)
      - Normalize alef variants (أ إ آ → ا)
      - Normalize taa marbuta (ة → ه)
      - Remove punctuation
      - Collapse whitespace
    """
    # Remove diacritics (tashkeel): fatha, damma, kasra, sukun, shadda, tanween
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

    # Normalize alef variants
    text = re.sub(r"[أإآ]", "ا", text)

    # Normalize taa marbuta
    text = text.replace("ة", "ه")

    # Remove punctuation (Arabic + Latin)
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate between reference and hypothesis.

    Both texts are Arabic-normalized before comparison.

    Args:
        reference: Ground truth text.
        hypothesis: ASR transcription.

    Returns:
        WER as a float (0.0 = perfect, 1.0 = 100% errors).
        Returns 1.0 if reference is empty.
    """
    ref = normalize_arabic(reference)
    hyp = normalize_arabic(hypothesis)

    if not ref:
        log.warning("Empty reference text, returning WER=1.0")
        return 1.0

    try:
        score = jiwer.wer(ref, hyp)
        return min(score, 1.0)  # Cap at 1.0
    except Exception as e:
        log.error(f"WER computation failed: {e}")
        return 1.0


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate between reference and hypothesis.

    Both texts are Arabic-normalized before comparison.

    Args:
        reference: Ground truth text.
        hypothesis: ASR transcription.

    Returns:
        CER as a float (0.0 = perfect, 1.0 = 100% errors).
        Returns 1.0 if reference is empty.
    """
    ref = normalize_arabic(reference)
    hyp = normalize_arabic(hypothesis)

    if not ref:
        log.warning("Empty reference text, returning CER=1.0")
        return 1.0

    try:
        score = jiwer.cer(ref, hyp)
        return min(score, 1.0)
    except Exception as e:
        log.error(f"CER computation failed: {e}")
        return 1.0


def batch_wer(
    references: List[str], hypotheses: List[str]
) -> List[Tuple[float, float]]:
    """
    Compute WER and CER for batches of reference/hypothesis pairs.

    Args:
        references: List of ground truth texts.
        hypotheses: List of ASR transcriptions.

    Returns:
        List of (wer, cer) tuples.
    """
    assert len(references) == len(hypotheses), (
        f"Mismatched lengths: {len(references)} refs vs {len(hypotheses)} hyps"
    )

    results = []
    for ref, hyp in zip(references, hypotheses):
        w = compute_wer(ref, hyp)
        c = compute_cer(ref, hyp)
        results.append((w, c))

    avg_wer = sum(r[0] for r in results) / len(results) if results else 0.0
    log.info(
        f"Batch WER: {avg_wer:.3f} avg over {len(results)} samples"
    )
    return results
