#!/usr/bin/env python3
"""
NeMo ASR Remote Inference Server — EgypTalk-ASR-v2.

Runs the NAMAA-Space/EgypTalk-ASR-v2 NeMo model on a GPU-equipped machine
and exposes a /transcribe endpoint via Ngrok for remote access.

Usage (on any machine with an NVIDIA GPU):
    1. pip install nemo_toolkit[asr] flask pyngrok huggingface_hub
    2. Set your Ngrok auth token below or as NGROK_AUTHTOKEN env var
    3. python nemagpu.py
    4. Copy the printed Ngrok URL into config.yaml → colab_api_url

This also works on Google Colab — just run the cells in order.

Requirements:
    - NVIDIA GPU with CUDA support (will NOT work on Apple Silicon / Mac M1)
    - nemo_toolkit[asr], flask, pyngrok, huggingface_hub
"""

import os
import sys
import threading

from flask import Flask, request, jsonify
from pyngrok import ngrok
from nemo.collections.asr.models import ASRModel
from huggingface_hub import hf_hub_download

# ── Configuration ─────────────────────────────────
# Set your Ngrok auth token via the NGROK_AUTHTOKEN environment variable
# or in the .env file. Get a free token at:
# https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTHTOKEN", "")
if not NGROK_AUTH_TOKEN:
    print("ERROR: NGROK_AUTHTOKEN not set. Add it to your .env file or export it.")
    sys.exit(1)
os.environ["NGROK_AUTHTOKEN"] = NGROK_AUTH_TOKEN

PORT = int(os.environ.get("NEMA_PORT", 5000))

# ── Flask App ─────────────────────────────────────
app = Flask(__name__)

# ── Load the NeMo ASR model ──────────────────────
print("=" * 60)
print("  NeMo ASR Server — EgypTalk-ASR-v2")
print("=" * 60)

print("\n→ Downloading NeMo model from Hugging Face...")
nemo_file = hf_hub_download(
    repo_id="NAMAA-Space/EgypTalk-ASR-v2",
    filename="asr-egyptian-nemo-v2.0.nemo"
)

print("→ Loading NeMo model into memory (this takes ~1 minute)...")
asr_model = ASRModel.restore_from(nemo_file)
print("✓ Model loaded successfully!\n")


@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcribe an uploaded .wav audio file using EgypTalk-ASR-v2."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the uploaded file temporarily
    temp_path = "temp_audio.wav"
    file.save(temp_path)

    try:
        result = asr_model.transcribe([temp_path])
        text = result[0] if isinstance(result[0], str) else str(result[0])
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model": "EgypTalk-ASR-v2"})


# ── Expose via Ngrok & start server ──────────────
if __name__ == "__main__":
    public_url = ngrok.connect(PORT).public_url
    print("=" * 60)
    print(f"  ✅ YOUR ASR SERVER URL IS: {public_url}")
    print(f"  Paste this into config.yaml → colab_api_url")
    print("=" * 60)
    print(f"\n  Listening on port {PORT}...")
    print("  Press Ctrl+C to stop.\n")

    app.run(host="0.0.0.0", port=PORT)