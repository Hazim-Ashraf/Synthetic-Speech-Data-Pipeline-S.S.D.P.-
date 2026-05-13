"""
Stage 3 — Auto-Scoring + Streamlit Review App.

Phase A: ASR transcription (NeMo primary, Whisper fallback) + WER scoring.
Phase B: Streamlit UI for human review of (text, audio) pairs.

Scoring is checkpointed per-sample via stage3_scores_progress.json.
Auto-labeling uses WER thresholds from config.yaml.
"""
import json, os, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def _load_config() -> Dict[str, Any]:
    with open(Path(__file__).parent.parent / "config.yaml", "r") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════
# ASR loading
# ══════════════════════════════════════════════════

def _load_asr(config):
    """Load ASR: try NeMo locally, HF API, or fall back to Whisper."""
    backend = config.get("asr_backend", "auto")
    
    if backend == "colab_api":
        colab_url = config.get("colab_api_url", "").strip()
        if not colab_url or colab_url == "PASTE_YOUR_COLAB_URL_HERE":
            print("[ASR] Warning: colab_api_url not set. Falling back to Whisper...")
        else:
            print(f"[ASR] Using EgypTalk-ASR-v2 via Colab API at {colab_url}...")
            import requests
            import re
            
            def _clean_hypothesis(text: str) -> str:
                if text.startswith("Hypothesis(") and ("text='" in text or 'text="' in text):
                    match = re.search(r"text='([^']*)'", text)
                    if match: return match.group(1)
                    match = re.search(r'text="([^"]*)"', text)
                    if match: return match.group(1)
                return text

            def fn(p):
                with open(p, "rb") as f:
                    files = {"file": f}
                    for attempt in range(3):
                        try:
                            response = requests.post(f"{colab_url}/transcribe", files=files, timeout=30)
                            if response.status_code == 200:
                                raw_text = response.json().get("text", "")
                                return _clean_hypothesis(raw_text)
                            else:
                                raise RuntimeError(f"Colab API Error: {response.status_code} {response.text}")
                        except Exception as e:
                            if attempt == 2:
                                raise RuntimeError(f"Colab API failed after 3 attempts: {e}")
                            import time
                            time.sleep(2)
            return fn, "colab_api"

    if backend == "hf_api":
        hf_token = config.get("hf_token", "") or os.environ.get("HF_TOKEN", "")
        if not hf_token:
            print("[ASR] Warning: hf_token missing. Falling back to Whisper...")
        else:
            print("[ASR] Using EgypTalk-ASR-v2 via Hugging Face Inference API...")
            import requests
            import time
            def fn(p):
                API_URL = "https://api-inference.huggingface.co/models/NAMAA-Space/EgypTalk-ASR-v2"
                headers = {"Authorization": f"Bearer {hf_token}"}
                with open(p, "rb") as f:
                    data = f.read()
                for attempt in range(3):
                    response = requests.post(API_URL, headers=headers, data=data)
                    if response.status_code == 200:
                        res = response.json()
                        return res.get("text", "").strip() if isinstance(res, dict) else str(res)
                    elif response.status_code == 503 and "loading" in response.text.lower():
                        time.sleep(15)  # Wait for model to load into memory on HF servers
                        continue
                    else:
                        raise RuntimeError(f"HF API Error: {response.status_code} {response.text}")
                raise RuntimeError("HF API failed to load the model after 3 attempts.")
            return fn, "hf_api"

    if backend in ("nemo", "auto"):
        try:
            from nemo.collections.asr.models import ASRModel
            print("[ASR] Loading EgypTalk-ASR-v2 via NeMo...")
            asr = ASRModel.from_pretrained("NAMAA-Space/EgypTalk-ASR-v2")
            def fn(p):
                r = asr.transcribe([p])
                return r[0] if isinstance(r[0], str) else str(r[0])
            print("[ASR] NeMo loaded.")
            return fn, "nemo"
        except (ImportError, Exception) as e:
            if backend == "nemo":
                raise
            print(f"[ASR] NeMo unavailable ({e}), falling back to Whisper...")
            
    import whisper
    sz = config.get("whisper_model_size", "large-v3")
    print(f"[ASR] Loading Whisper {sz}...")
    print(f"      (Note: If downloading weights or loading into CPU RAM, this may take a few minutes. Progress will be shown below.)")
    mdl = whisper.load_model(sz)
    def fn(p):
        return mdl.transcribe(p, language="ar").get("text", "").strip()
    print(f"[ASR] Whisper {sz} loaded.")
    return fn, "whisper"


def _auto_label(wer, cfg):
    """Assign an auto-label based on WER thresholds from config."""
    wa = cfg.get("wer_auto_approve_threshold", 0.15)
    wr = cfg.get("wer_auto_reject_threshold", 0.40)
    if wer < wa: return "auto_approved"
    if wer > wr: return "auto_rejected"
    return "needs_review"


# ══════════════════════════════════════════════════
# Phase A — Auto-scoring (checkpointed)
# ══════════════════════════════════════════════════

def run_auto_scoring(config):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from utils.wer import compute_wer
    from utils.checkpoint import load_checkpoint, save_checkpoint
    from tqdm import tqdm

    od = Path(config.get("output_dir", "data"))

    # ── Merge Track A + Track B manifests ─────────
    manifest = []

    # Track A — synthetic
    track_a_path = od / "manifests/stage2_manifest.json"
    if track_a_path.exists():
        track_a = load_checkpoint(str(track_a_path))
        if track_a:
            for s in track_a:
                s.setdefault("source", "synthetic")
            manifest.extend(track_a)

    # Track B — real speech
    track_b_path = od / "manifests/stage1b_manifest.json"
    if track_b_path.exists():
        track_b = load_checkpoint(str(track_b_path))
        if track_b:
            # source field already set to "real" in stage1b
            manifest.extend(track_b)

    if not manifest:
        print("Error: No manifests found. Run Stage 2 and/or Stage 1b first.")
        return
    manifest = [m for m in manifest if m.get("status") == "done"]

    scores_path = od / "manifests/stage3_scores.json"
    prog_path = od / "manifests/stage3_scores_progress.json"

    progress = load_checkpoint(str(prog_path))
    if not isinstance(progress, dict): progress = {}
    
    # Force fresh run if either manifest is newer than Stage 3 progress
    if prog_path.exists():
        prog_mtime = prog_path.stat().st_mtime
        for mpath in [track_a_path, track_b_path]:
            if mpath.exists() and mpath.stat().st_mtime > prog_mtime:
                print(f"{mpath.name} is newer than Stage 3 progress. Forcing fresh scoring run.")
                progress = {}
                break

    pending = [m for m in manifest if m["id"] not in progress]
    print(f"[Scoring] {len(progress)} done, {len(pending)} remaining.")

    if pending:
        transcribe, asr_name = _load_asr(config)
        for s in tqdm(pending, desc="Auto-scoring"):
            sid, text, ap = s["id"], s["text"], s["audio_path"]
            try:
                hyp = transcribe(ap)
                wer = compute_wer(text, hyp)
                progress[sid] = {"wer": round(wer, 4),
                    "asr_hypothesis": hyp, "auto_label": _auto_label(wer, config),
                    "final_label": None, "reviewer_notes": ""}
            except Exception as e:
                progress[sid] = {"wer": 1.0, "asr_hypothesis": "",
                    "auto_label": "auto_rejected", "final_label": None,
                    "reviewer_notes": f"Error: {e}"}
            save_checkpoint(str(prog_path), progress)
        print(f"[Scoring] ASR backend: {asr_name}")

    # Build scores file (preserve source field)
    scores = []
    for m in manifest:
        info = progress.get(m["id"], {})
        scores.append({"id": m["id"], "text": m["text"],
            "audio_path": m["audio_path"], "duration_sec": m.get("duration_sec", 0),
            "source": m.get("source", "synthetic"),
            "wer": info.get("wer", 1.0),
            "asr_hypothesis": info.get("asr_hypothesis", ""),
            "auto_label": info.get("auto_label", "auto_rejected"),
            "final_label": info.get("final_label"),
            "reviewer_notes": info.get("reviewer_notes", "")})
    save_checkpoint(str(scores_path), scores)

    labels = [v.get("auto_label", "") for v in progress.values()]
    print(f"\n  Approved: {labels.count('auto_approved')}, "
          f"Rejected: {labels.count('auto_rejected')}, "
          f"Review: {labels.count('needs_review')}")


# ══════════════════════════════════════════════════
# Phase B — Streamlit UI
# ══════════════════════════════════════════════════

def _eff_label(s):
    return s.get("final_label") or s.get("auto_label", "needs_review")

def _save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_streamlit_app():
    import streamlit as st
    cfg = _load_config()
    sp = Path(cfg.get("output_dir", "data")) / "manifests/stage3_scores.json"

    st.set_page_config(page_title="SSDP Review", page_icon="🎧", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)

    if not sp.exists():
        st.error("Scores file not found. Run auto-scoring first.")
        return
    with open(sp, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        st.error("No scored samples.")
        return

    if "idx" not in st.session_state: st.session_state.idx = 0

    # Sidebar
    with st.sidebar:
        st.title("🎧 SSDP Review")
        st.markdown("---")
        filt = st.selectbox("Filter", ["all","needs_review","auto_approved","auto_rejected","synthetic_only","real_only"])
        if filt == "synthetic_only":
            filtered = [s for s in data if s.get("source", "synthetic") == "synthetic"]
        elif filt == "real_only":
            filtered = [s for s in data if s.get("source", "synthetic") == "real"]
        elif filt == "all":
            filtered = data
        else:
            filtered = [s for s in data if _eff_label(s) == filt]

        st.markdown("---")
        labels = [_eff_label(s) for s in data]
        ap = sum(1 for l in labels if l in ("auto_approved","approved"))
        rj = sum(1 for l in labels if l in ("auto_rejected","rejected"))
        pn = sum(1 for l in labels if l == "needs_review")
        for val, lbl, clr in [(ap,"Approved","#a6e3a1"),(rj,"Rejected","#f38ba8"),(pn,"Pending","#f9e2af")]:
            st.markdown(f'<div class="stat-card"><div class="stat-num" style="color:{clr}">{val}</div>'
                        f'<div class="stat-lbl">{lbl}</div></div>', unsafe_allow_html=True)

        # Source breakdown
        synth_count = sum(1 for s in data if s.get("source", "synthetic") == "synthetic")
        real_count = sum(1 for s in data if s.get("source", "synthetic") == "real")
        if real_count > 0:
            st.markdown("---")
            st.caption(f"🤖 Synthetic: {synth_count}    🎙️ Real: {real_count}")

        st.markdown("---")
        st.progress((ap+rj)/len(data) if data else 0)
        st.caption(f"{ap+rj}/{len(data)} reviewed")

        st.markdown("---")
        st.markdown("### Finish Review")
        
        all_reviewed = (ap + rj) == len(data)
        
        if not all_reviewed:
            approve_unreviewed = st.checkbox("Approve all unreviewed samples (Export as they are)")
        else:
            approve_unreviewed = False
            st.success("All samples reviewed! Ready to export.")
            
        can_proceed = all_reviewed or approve_unreviewed
        
        if st.button("Proceed to Stage 4 ➡️", type="primary", use_container_width=True, disabled=not can_proceed):
            if approve_unreviewed:
                for x in data:
                    if x.get("final_label") is None:
                        x["final_label"] = "approved"
                _save_json(data, sp)
            os._exit(0)

    if not filtered:
        st.info("No samples match filter.")
        return

    st.session_state.idx = max(0, min(st.session_state.idx, len(filtered)-1))
    s = filtered[st.session_state.idx]

    st.markdown(f"### Sample `{s['id']}` — {st.session_state.idx+1}/{len(filtered)}")

    # Source badge
    src = s.get("source", "synthetic")
    if src == "real":
        st.markdown('<span class="bb">🎙️ Real Speech</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="bb">🤖 Synthetic</span>', unsafe_allow_html=True)

    # Navigation
    c1, _, c3 = st.columns([1,3,1])
    with c1:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.idx = max(0, st.session_state.idx-1); st.rerun()
    with c3:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.idx = min(len(filtered)-1, st.session_state.idx+1); st.rerun()

    st.markdown("---")
    st.markdown("**Original Text:**")
    st.markdown(f'<div class="ar-text">{s["text"]}</div>', unsafe_allow_html=True)

    ap_path = s.get("audio_path", "")
    if ap_path and os.path.exists(ap_path): st.audio(ap_path, format="audio/wav")
    else: st.warning("Audio file not found.")

    # Badges
    b1, b2 = st.columns(2)
    wer = s.get("wer", 1.0)
    bc = "bg" if wer < 0.15 else ("by" if wer < 0.40 else "br")
    b1.markdown(f'<span class="{bc}">WER: {wer*100:.1f}%</span>', unsafe_allow_html=True)
    el = _eff_label(s)
    lc = {"auto_approved":"bg","approved":"bg","auto_rejected":"br","rejected":"br"}.get(el,"by")
    b2.markdown(f'<span class="{lc}">{el}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**ASR Hypothesis:**")
    st.markdown(f'<div class="ar-hyp">{s.get("asr_hypothesis","—")}</div>', unsafe_allow_html=True)
    st.caption(f"Duration: {s.get('duration_sec',0):.2f}s")

    st.markdown("---")
    a1, a2 = st.columns(2)
    with a1:
        if st.button("✅ Approve", use_container_width=True, type="primary"):
            for x in data:
                if x["id"] == s["id"]: x["final_label"] = "approved"; break
            _save_json(data, sp)
            st.session_state.idx = min(len(filtered)-1, st.session_state.idx+1); st.rerun()
    with a2:
        if st.button("❌ Reject", use_container_width=True):
            for x in data:
                if x["id"] == s["id"]: x["final_label"] = "rejected"; break
            _save_json(data, sp)
            st.session_state.idx = min(len(filtered)-1, st.session_state.idx+1); st.rerun()

    notes = st.text_area("Reviewer notes", value=s.get("reviewer_notes",""), key=f"n_{s['id']}")
    if notes != s.get("reviewer_notes",""):
        for x in data:
            if x["id"] == s["id"]: x["reviewer_notes"] = notes; break
        _save_json(data, sp)


_CSS = """<style>
.ar-text{direction:rtl;text-align:right;font-size:1.8rem;font-family:'Noto Sans Arabic',sans-serif;
line-height:2;padding:1rem;background:#1e1e2e;border-radius:.5rem;border-left:4px solid #89b4fa;
color:#cdd6f4;margin-bottom:1rem}
.ar-hyp{direction:rtl;text-align:right;font-size:1.3rem;font-family:'Noto Sans Arabic',sans-serif;
line-height:1.8;padding:.8rem;background:#313244;border-radius:.5rem;color:#bac2de;margin-bottom:1rem}
.bg{background:#a6e3a1;color:#1e1e2e;padding:.3rem .8rem;border-radius:1rem;font-weight:bold;
font-size:.9rem;display:inline-block}
.by{background:#f9e2af;color:#1e1e2e;padding:.3rem .8rem;border-radius:1rem;font-weight:bold;
font-size:.9rem;display:inline-block}
.br{background:#f38ba8;color:#1e1e2e;padding:.3rem .8rem;border-radius:1rem;font-weight:bold;
font-size:.9rem;display:inline-block}
.bb{background:#89b4fa;color:#1e1e2e;padding:.3rem .8rem;border-radius:1rem;font-weight:bold;
font-size:.9rem;display:inline-block}
.stat-card{background:#313244;padding:1rem;border-radius:.5rem;text-align:center;margin-bottom:.5rem}
.stat-num{font-size:2rem;font-weight:bold}
.stat-lbl{font-size:.85rem;color:#a6adc8}
</style>"""

if __name__ == "__main__":
    run_streamlit_app()
