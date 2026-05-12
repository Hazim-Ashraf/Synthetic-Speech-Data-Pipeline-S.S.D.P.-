"""
Stage 1 — Egyptian Arabic Text Generation via Nile-Chat-4B.

Generates natural, colloquial Egyptian Arabic (عامية مصرية) text prompts
using MBZUAI-Paris/Nile-Chat-4B, a Gemma 3-based model fine-tuned
specifically for Egyptian Arabic text generation.

The model runs locally via HuggingFace Transformers — no API key required.

Output: data/manifests/stage1_prompts.json
"""

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logger import get_logger

log = get_logger(__name__)

# ── System prompt for Nile-Chat-4B ────────────
SYSTEM_PROMPT = """You are an Egyptian Arabic linguist. Generate natural, colloquial Egyptian Arabic \
sentences (عامية مصرية). 
Rules:
- Arabic script ONLY. No romanized Arabic (Arabizi). No English.
- Pure Egyptian dialect. NO Modern Standard Arabic (فصحى).
- Use authentic Egyptian vocabulary: عايز، مش، دلوقتي، إيه، بتاع، ازيك، يلا، معلش
- Vary sentence length: short (3-5 words), medium (6-10 words), long (11-15 words)
- Ensure EXACTLY even coverage of these 6 domains to prevent repetition (approx. 8-9 samples per domain):
  1. Sales / Shopping (e.g., "القميص ده عليه خصم حلو أوي")
  2. Customer Service (e.g., "يا فندم أقدر أساعد حضرتك إزاي؟")
  3. Teacher talking to students (e.g., "يا ولاد ركزوا معايا في الدرس ده")
  4. Daily greetings / casual talk (e.g., "صباح الفل يا باشا عامل إيه؟")
  5. Food ordering / delivery (e.g., "الأوردر اتأخر جدا يا جماعة")
  6. General / Miscellaneous (street directions, family, etc.)
- Output ONLY a JSON array of strings. No explanation. No numbering.
  Example: ["يا فندم رقم الطلب كام؟", "افتحوا الكتاب صفحة عشرة"]"""

# ── Domain keywords for classification ────────
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "daily_greetings": [
        "صباح", "مساء", "ازيك", "أخبارك", "سلام", "عامل", "تصبح",
        "مع السلامه", "أهلا", "يا باشا", "الحمد لله", "يا معلم",
    ],
    "shopping": [
        "سوق", "اشتري", "فلوس", "سعر", "بكام", "غالي", "رخيص",
        "محل", "تمن", "حساب", "خصم", "عروض", "منتج", "تخفيض", "مقاس", "ألوان"
    ],
    "family": [
        "ماما", "بابا", "أخو", "أخت", "عيلة", "بيت", "ولاد",
        "جوز", "مرات", "جدو", "تيته",
    ],
    "street_directions": [
        "شارع", "يمين", "شمال", "دوغري", "ناصيه", "محطه", "روح",
        "فين", "طريق", "ميدان", "كوبري","مواصلات",
    ],
    "phone_calls": [
        "تليفون", "كلم", "اتصل", "رن", "رسال", "موبايل", "خط",
        "مكالمه", "سمع", "رد",
    ],
    "food_ordering": [
        "أكل", "هتطلب", "مطعم", "فول", "كشري", "شاي", "قهوه",
        "سندوتش", "فطار", "غدا", "عشا", "دليفري",
    ],
    "complaints": [
        "زهق", "زعلان", "مشكله", "وحش", "مش كويس", "ضايق",
        "تعبان", "مقرف", "خراب", "بايظ",
    ],
    "surprise_emotion": [
        "يا نهار", "ربنا", "معقول", "بجد", "مصدق", "عجبي",
        "يا لهوي", "ده إيه", "ياه", "خضيت",
    ],
    "work_talk": [
        "شغل", "مدير", "مرتب", "اجتماع", "مكتب", "مشروع",
        "اجازه", "زميل", "شركه", "ترقيه",
    ],
    "customer_service": [
        "يا فندم", "خدمة", "عملاء", "أساعد", "حضرتك", "مشكلة",
        "رقم", "طلب", "نعتذر", "تأكيد", "شكوى", "حسابك", "نحلها"
    ],
    "teacher_students": [
        "يا ولاد", "درس", "امتحان", "واجب", "كتاب", "كراسة", "فهمتوا",
        "سؤال", "جواب", "سبورة", "مدرسة", "فصل", "ركزوا", "صفحة", "أستاذ"
    ],
}


def _classify_domain(text: str) -> str:
    """Classify a sentence into a domain based on keyword matching."""
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[domain] = score

    if scores:
        return max(scores, key=scores.get)
    return "general"


def _classify_length(text: str) -> str:
    """Classify sentence length: short (3-5), medium (6-10), long (11+)."""
    word_count = len(text.split())
    if word_count <= 5:
        return "short"
    elif word_count <= 10:
        return "medium"
    else:
        return "long"


def _parse_model_response(content: str) -> List[str]:
    """
    Parse model response, extracting JSON array of Arabic strings.

    Handles cases where the model wraps JSON in markdown code blocks
    or produces extra text around the array.
    """
    content = content.strip()

    # Strip markdown code fences if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    content = content.strip()

    # Try direct JSON parse first
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [s.strip() for s in data if isinstance(s, str) and s.strip()]
    except json.JSONDecodeError:
        pass

    # Try to extract JSON array from surrounding text
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [s.strip() for s in data if isinstance(s, str) and s.strip()]
        except json.JSONDecodeError:
            pass

    log.warning(f"Failed to parse model response as JSON: {content[:200]}...")
    return []


def _load_model(config: Dict[str, Any]):
    """
    Load Nile-Chat-4B GGUF model via llama-cpp-python.

    Args:
        config: Pipeline config dict.

    Returns:
        Tuple of (llama_model, model_name).
    """
    from huggingface_hub import snapshot_download
    from llama_cpp import Llama

    repo_id = config.get("text_gen_gguf_repo", "tensorblock/MBZUAI-Paris_Nile-Chat-4B-GGUF")
    filename_pattern = config.get("text_gen_gguf_file", "*Q4_K_M.gguf")
    hf_token = config.get("hf_token", "") or os.environ.get("HF_TOKEN", "")

    log.info(f"Downloading/Locating GGUF model: {repo_id} ({filename_pattern})...")
    
    # Download or use cached model (snapshot_download handles wildcards)
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=filename_pattern,
        token=hf_token if hf_token else None,
    )

    # Find the specific .gguf file inside local_dir
    gguf_files = list(Path(local_dir).glob(filename_pattern))
    if not gguf_files:
        raise FileNotFoundError(f"Could not find GGUF file matching {filename_pattern} in {local_dir}")
    model_path = str(gguf_files[0])

    # Determine GPU layers based on OS version and config
    device = config.get("text_gen_device", config.get("tts_device", "mps"))
    n_gpu_layers = -1  # Attempt full GPU offloading by default
    
    import platform
    if platform.system() == "Darwin":
        mac_ver = platform.mac_ver()[0]
        try:
            major_ver = int(mac_ver.split('.')[0])
            if major_ver < 13:
                log.warning(
                    f"macOS {mac_ver} detected. Apple Metal requires macOS 13.0+. "
                    f"Falling back to CPU inference for llama.cpp to avoid crashes."
                )
                n_gpu_layers = 0
        except Exception:
            pass

    if device == "cpu":
        n_gpu_layers = 0

    log.info(f"Loading GGUF model from {model_path} into memory...")
    
    # Load model with llama.cpp
    # n_ctx=2048 handles our context window
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=n_gpu_layers, 
        verbose=False,
    )

    log.info(f"GGUF model loaded successfully.")
    return llm, repo_id


def run(config: Dict[str, Any]) -> None:
    """
    Execute Stage 1: Generate Egyptian Arabic text prompts via Nile-Chat-4B (GGUF).

    Args:
        config: Pipeline configuration dictionary.
    """
    num_samples = config.get("num_samples", 200)
    batch_size = config.get("generation_batch_size", 20)
    output_dir = Path(config.get("output_dir", "data"))
    manifest_path = output_dir / "manifests" / "stage1_prompts.json"
    checkpoint_path = output_dir / "manifests" / "stage1_checkpoint.json"

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────
    existing = load_checkpoint(str(checkpoint_path))
    all_texts: List[str] = existing if isinstance(existing, list) else []
    seen = set(all_texts)

    log.info(
        f"Stage 1: Targeting {num_samples} samples, "
        f"{len(all_texts)} already generated, "
        f"batch_size={batch_size}"
    )

    if len(all_texts) >= num_samples:
        log.info("Target already reached from checkpoint. Skipping generation.")
    else:
        # ── Load model (only if we need to generate) ──
        llm, model_name = _load_model(config)
        log.info(f"Using model: {model_name}")

        # ── Batch generation loop ─────────────────────
        batch_num = 0
        while len(all_texts) < num_samples:
            remaining = num_samples - len(all_texts)
            request_count = min(batch_size, remaining)
            batch_num += 1

            user_prompt = (
                f"Generate exactly {request_count} unique Egyptian Arabic sentences. "
                f"Return ONLY a JSON array of strings."
            )

            log.info(
                f"Batch {batch_num}: requesting {request_count} prompts "
                f"({len(all_texts)}/{num_samples} done)"
            )

            try:
                # llama-cpp-python chat completion API
                outputs = llm.create_chat_completion(
                    messages=[
                        {"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{user_prompt}"},
                    ],
                    max_tokens=2048,
                    temperature=0.9,
                )

                # Extract generated text
                content = outputs["choices"][0]["message"]["content"].strip()
                new_texts = _parse_model_response(content)

                # Deduplicate
                added = 0
                for text in new_texts:
                    if text not in seen and len(all_texts) < num_samples:
                        all_texts.append(text)
                        seen.add(text)
                        added += 1

                log.info(
                    f"Batch {batch_num}: parsed {len(new_texts)} sentences, "
                    f"{added} new (after dedup). Total: {len(all_texts)}/{num_samples}"
                )

                # Checkpoint after each batch
                save_checkpoint(str(checkpoint_path), all_texts)

            except Exception as e:
                log.error(f"Batch {batch_num}: Generation error: {e}")
                continue

    # ── Build manifest with metadata ──────────────
    manifest = []
    for idx, text in enumerate(all_texts[:num_samples]):
        manifest.append({
            "id": f"{idx + 1:04d}",
            "text": text,
            "domain": _classify_domain(text),
            "length_class": _classify_length(text),
            "word_count": len(text.split()),
        })

    save_checkpoint(str(manifest_path), manifest)
    log.info(f"Stage 1 complete: {len(manifest)} prompts saved to {manifest_path}")

    # ── Log statistics ────────────────────────────
    domain_dist = Counter(m["domain"] for m in manifest)
    length_dist = Counter(m["length_class"] for m in manifest)
    avg_words = sum(m["word_count"] for m in manifest) / len(manifest) if manifest else 0

    log.info(f"Domain distribution: {dict(domain_dist)}")
    log.info(f"Length distribution: {dict(length_dist)}")
    log.info(f"Average word count: {avg_words:.1f}")
