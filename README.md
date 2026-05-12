# SSDP — Synthetic Speech Data Pipeline for Egyptian Arabic

A production-grade pipeline for generating synthetic speech training data targeting fine-tuning of Speech-to-Text (STT) models on Egyptian Arabic (عامية مصرية).

## Pipeline Architecture

```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   STAGE 1           │     │   STAGE 2           │     │   STAGE 3           │     │   STAGE 4           │
│   Text Generation   │────▶│   TTS Synthesis     │────▶│   Review & Score    │────▶│   Dataset Export    │
│                     │     │                     │     │                     │     │                     │
│  Nile-Chat-4B →     │     │  NAMAA-Egyptian-TTS │     │  A: Auto-scoring    │     │  HuggingFace        │
│  Egyptian prompts   │     │  → .wav audio files │     │  B: Streamlit UI    │     │  Dataset + CSV      │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘     └─────────────────────┘
         │                           │                           │                           │
         ▼                           ▼                           ▼                           ▼
  stage1_prompts.json      stage2_manifest.json        stage3_scores.json         output/dataset + CSV
  stage1_checkpoint.json   stage2_progress.json        stage3_scores_progress.json
```

**Every stage writes a manifest.** Any stage can be re-run independently. Stages 1, 2, and 3's scoring phase are fully resumable via checkpoints — a crash at sample 150 restarts from 151, not from zero.

---

## Model Choices & Rationale

### Text Generation: MBZUAI-Paris/Nile-Chat-4B
- **Why**: A 4B-parameter model from the JAIS initiative, based on Gemma 3 and fine-tuned specifically for Egyptian Arabic (عامية مصرية). Unlike general-purpose LLMs that treat Egyptian dialect as a secondary capability, Nile-Chat was continually pre-trained on 3.3B tokens of Egyptian web text and instruction-tuned on 1.9M Egyptian Arabic instructions. It natively understands the dialect's vocabulary, grammar, and orthographic conventions.
- **Trade-off**: Being a 4B model, it requires ~8GB RAM (bfloat16) and runs locally rather than via API. Generation is slower than a cloud API but eliminates API costs and external dependencies. The model runs on CPU, CUDA, or MPS (Apple Silicon).
- **Why not GPT-4o**: Nile-Chat-4B is purpose-built for Egyptian Arabic, whereas GPT-4o treats it as one of many languages. Nile-Chat produces more authentic dialectal text with fewer MSA contamination artifacts. It also runs fully offline — no API key or internet required during generation.

### TTS: NAMAA-Space/NAMAA-Egyptian-TTS (via chatterbox)
- **Why**: One of the very few TTS models specifically trained on Egyptian Arabic dialect. Most Arabic TTS models target Modern Standard Arabic (فصحى), which sounds fundamentally different from Egyptian colloquial speech. NAMAA's model captures Egyptian phonology, intonation, and vocabulary.
- **Trade-off**: Synthetic speech will never match natural recordings in prosody variation. The pipeline mitigates this through WER-based quality scoring to filter low-quality outputs.

### ASR Judge (Primary): NAMAA-Space/EgypTalk-ASR-v2 (via NeMo)
- **Why**: Purpose-built for Egyptian Arabic transcription. Trained specifically on Egyptian dialect data, giving it significantly better accuracy than general Arabic ASR models.
- **Trade-off**: Requires `nemo_toolkit[asr]`, one of the heaviest ML installs in Python (PyTorch Lightning, Hydra, Apex, etc.). Known to have installation issues on Mac M1/ARM systems. The pipeline solves this via a **Remote GPU Server** (see below).

### ASR Judge (Fallback): OpenAI Whisper large-v3
- **Why**: Universal multilingual ASR with reasonable Arabic coverage. Easy to install, works on all platforms including Mac M1. Selected as a pragmatic fallback when NeMo installation fails.
- **Trade-off**: Whisper was not specifically trained on Egyptian dialect. It may normalize dialectal speech toward MSA, producing slightly higher WER scores on authentic Egyptian text. This means Whisper-scored data will have a conservative bias — some good samples may be sent to manual review rather than auto-approved.
- **How fallback works**: Set `asr_backend: auto` in config.yaml (default). The pipeline tries NeMo first; if the import fails, it automatically falls back to Whisper and logs which backend is active.

---

## Remote ASR Server (nemagpu.py)

The primary ASR model (EgypTalk-ASR-v2) requires NVIDIA's `nemo_toolkit`, which **cannot compile natively on Apple Silicon**. To bypass this, the pipeline includes a **Remote GPU Inference Server**.

### How It Works
1. `nemagpu.py` runs on any machine with an NVIDIA GPU (e.g., Google Colab, a Linux server, cloud VM)
2. It loads the NeMo model and exposes a `/transcribe` HTTP endpoint via Ngrok
3. Your Mac sends audio files over the network to the GPU server for transcription
4. Results are returned as JSON and used for WER scoring

### Running on Google Colab
1. Upload `nemagpu.py` to Google Colab or open the linked notebook
2. Run the script — it will install dependencies, load the model, and print an Ngrok URL
3. Copy the URL into `config.yaml` → `colab_api_url`
4. Set `asr_backend: "colab_api"` in `config.yaml`

### Running on Any NVIDIA GPU Machine
```bash
# Install dependencies
pip install nemo_toolkit[asr] flask pyngrok huggingface_hub

# Set your Ngrok auth token (get one at https://ngrok.com)
export NGROK_AUTHTOKEN="your_token_here"

# Start the server
python nemagpu.py
```

> **Note**: This script **requires CUDA (NVIDIA GPU)**. It will not work on Mac M1/M2/M3 (Apple Silicon) because NeMo depends on CUDA for model inference. The whole purpose of this server is to offload GPU-bound work to a CUDA-capable machine while your Mac runs the rest of the pipeline.

---

## Egyptian Arabic Challenges

### Dialect vs. MSA
Egyptian Arabic (عامية مصرية) differs substantially from Modern Standard Arabic (فصحى) in vocabulary, grammar, and phonology. Nile-Chat-4B was specifically trained on Egyptian dialect data, and the system prompt explicitly enforces dialectal output and prohibits MSA.

### Orthographic Variation
Egyptian Arabic has no standardized orthography. The same word may be written multiple ways (e.g., عاوز vs عايز). The WER scorer normalizes common variants (alef forms, taa marbuta) to reduce false mismatches.

### Code-Switching Risk
Even dialect-specialized models may occasionally mix English or MSA into generated text. The system prompt strictly prohibits this, but generated data should be spot-checked. Nile-Chat-4B is less prone to this than general-purpose models since it was trained predominantly on Egyptian Arabic data.

### Limited Training Data
Egyptian Arabic is low-resource compared to MSA. This pipeline exists precisely to address this gap through controlled synthetic data generation.

---

## Quality Assurance Design

### WER-Based Scoring
Every synthesized sample is scored by Word Error Rate (WER): how well the TTS output can be transcribed back to the original text. Low WER = the speech clearly represents the intended text.

### Auto-Label Thresholds
| Condition | Label |
|-----------|-------|
| WER < 15% | `auto_approved` |
| WER > 40% | `auto_rejected` |
| Everything else | `needs_review` |

### Human-in-the-Loop
- Auto-approved samples are **still shown** in the review UI for spot-checking
- Human decisions override auto-labels
- All decisions are persisted immediately (crash-safe)

---

## Observed Limitations & Trade-offs

1. **TTS naturalness**: Synthetic speech lacks the prosodic variation, hesitations, and emotional range of natural recordings. Models fine-tuned on this data should be supplemented with real speech data.

2. **Domain bias**: Nile-Chat-4B generates text based on its training data (3.3B tokens of Egyptian web text). The generated text may over-represent certain patterns or vocabulary common in the training corpus, while missing niche or regional expressions.

3. **NeMo complexity**: The full NeMo toolkit install pulls ~2GB of dependencies and requires CUDA. On Mac M1 systems, use the Colab API bridge or Whisper fallback.

4. **WER on synthetic speech**: WER is computed by transcribing TTS output back through ASR. This creates a circular dependency — both models may share similar failure modes, potentially over-approving certain error patterns.

---

## Setup & Usage

### Prerequisites
- Python 3.9+
- ~16GB RAM (for loading Nile-Chat-4B in bfloat16)
- ~15GB disk space (for models and audio)
- Mac M1/M2/M3 (MPS GPU used by default), or NVIDIA GPU (CUDA), or CPU

### Installation — Fully Automatic

**No manual setup required.** Just run `main.py` with any system Python 3.9+:

```bash
cd ssdp/
python main.py --all
```

On first run, `main.py` will automatically:
1. Create a `.venv/` virtual environment inside the project directory
2. Install all dependencies from `requirements.txt` **into `.venv`** (never globally)
3. Re-launch itself inside the venv and proceed with the pipeline

Subsequent runs detect the existing `.venv` and skip straight to execution.

> **Optional heavy dependency** (install manually inside the venv if needed):
> ```bash
> source .venv/bin/activate
> pip install nemo_toolkit[asr]   # Primary ASR (may fail on Mac M1)
> ```

### Configuration

The defaults in `config.yaml` are pre-set for **Mac M1** (MPS GPU):
```yaml
text_gen_device: "mps"      # Use "mps" on Mac M1, "cuda" on NVIDIA, "cpu" otherwise
tts_device: "mps"           # Same device options for TTS
asr_backend: "colab_api"    # "colab_api" for remote GPU, "auto" tries NeMo → Whisper
```

### Running the Pipeline

```bash
# Run individual stages (venv auto-created on first run)
python main.py --stage 1          # Generate text prompts
python main.py --stage 2          # Synthesize audio
python main.py --stage 3          # Auto-score + launch review UI
python main.py --stage 4          # Export approved samples

# Run everything sequentially
python main.py --all
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| NeMo install fails on Mac M1 | Use `asr_backend: "colab_api"` or `"whisper"` in config.yaml |
| Nile-Chat-4B OOM | Use `text_gen_device: "cpu"` (slower but lower memory) |
| TTS fails on some samples | Pipeline retries 3x with exponential backoff, marks failures |
| Streamlit won't launch | Run directly: `streamlit run stages/stage3_review.py` |
| Colab API returns `Hypothesis(...)` | Pipeline auto-cleans NeMo Hypothesis objects via regex |

---

## Project Structure

```
ssdp/
├── config.yaml                  # All externalized config
├── main.py                      # CLI entrypoint (run stages individually or all)
├── nemagpu.py                   # Remote ASR server (run on NVIDIA GPU machine)
├── stages/
│   ├── stage1_generate.py       # Text generation via Nile-Chat-4B
│   ├── stage2_synthesize.py     # TTS synthesis via NAMAA
│   ├── stage3_review.py         # Auto-scoring + Streamlit review app
│   └── stage4_export.py         # HuggingFace dataset export
├── utils/
│   ├── checkpoint.py            # Resume/checkpoint logic (atomic writes)
│   ├── wer.py                   # WER/CER scoring with Arabic normalization
│   └── logger.py                # Structured dual-output logging
├── data/
│   ├── manifests/               # JSON manifests per stage
│   ├── audio/                   # Synthesized .wav files
│   └── reviewed/                # Approved samples only
├── logs/
│   └── pipeline.log             # Structured log with timestamps
├── requirements.txt
└── README.md
```

---

## License

This pipeline is provided as-is for research and educational purposes.
The generated data inherits the licenses of the underlying models (Nile-Chat-4B, NAMAA TTS, EgypTalk ASR, Whisper).
