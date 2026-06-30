# Medical Voice Assistant

> Locally-run open-source medical voice agent that treats silence and prosody as clinical signals, real-time prosody-gated empathy, entirely on Apple Silicon.

**Not a chatbot wearing an audio costume.** This agent listens like a clinician — reading pauses, voice tremor, and what isn’t said — then adapts its own pace, vocabulary, and emotional tone in response. Everything runs locally on a MacBook Air M4.

---

## Why this exists

Most medical voice agents treat silence as dead air, disfluencies as noise, and emotion as a post-hoc tag. They jump in after 1.5 seconds of quiet and spit out text from an LLM that’s never heard the tremor in a patient’s voice.

**Silence is data.**
A pause after “when did the pain start?” is clinical information — memory retrieval, emotional processing, disclosure hesitancy, or confusion. The agent that fills that pause hasn’t listened; it’s just been waiting for you to stop.

This project builds a pipeline where:

- Silence budgets are per-question-type, not global.
- Prosody (pitch, pace, shimmer, tremor) gates the LLM’s system prompt *before* it speaks.
- Vocabulary mirrors the patient (“tummy” stays “tummy”).
- Deliberate 300–400ms pauses signal thoughtfulness.
- Oblique disclosures (“trouble with… you know… the drinking”) are met with open space, not structured probes.

---

## Core Architecture

```
Microphone → Silero VAD → Audio chunks
                           ↓
                   Whisper (local) + medical vocab boost
                           ↓
                  Transcript + disfluencies preserved
                           ↓
             openSMILE / SpeechBrain prosody stream
                           ↓
          Real-time prosodic feature vector extraction
                           ↓
     ┌─────── Silence Budget Controller (per question type)
     │
     ▼
 LLM (BioMistral / Llama-3.1 8B) with prosody-injected prompt
     │
     ▼
 Post-LLM deliberate pause (300–400ms)
     │
     ▼
 TTS (Piper / StyleTTS2) with adapted pacing & emotion
     │
     ▼
 Speaker output
```

All components run concurrently via a [Pipecat](https://github.com/pipecat-ai/pipecat) pipeline, orchestrated entirely on-device.

---

## What it actually does

- **Listens in real-time** with per-utterance silence budgets (configurable per question type: factual recall 4s, emotionally loaded questions up to 8s, suicide screening near-infinite patience).
- **Reads prosody** (pitch slope, energy contour, speech rate, jitter/shimmer) from the last few seconds of audio and translates it into a clinical qualifier.
- **Gates the LLM prompt**: if the patient’s voice shows distress markers, the system prepends instructions to slow down, validate, and not problem-solve.
- **Mirrors vocabulary**: “tummy” → “tummy”, “the bad one” → tracked as a named entity for later reference.
- **Handles oblique disclosure**: identifies hedging and under-disclosure, responds with minimal acknowledgment + open space (“Tell me more about that”) rather than jumping to a diagnostic probe.
- **Pauses before replying**: a deliberate 300–400ms post-LLM silence that reads as thoughtfulness, not lag.
- **Runs entirely locally** – no audio leaves the machine. Suitable for clinical simulations, patient-facing prototypes, and research in environments that demand data sovereignty.

---

## Tech Stack (All Local, Open-Source)

| Component | Implementation | Notes |
|-----------|---------------|-------|
| **VAD** | Silero VAD (ONNX) | Detects speech, silence, hesitation breaths. |
| **STT** | Whisper large-v3 (faster-whisper or whisper.cpp with CoreML) | Medical vocabulary injection. Preserves disfluencies via `whisper-timestamped`. |
| **Prosody extraction** | openSMILE (GeMAPS features) or SpeechBrain emotion models | Running on 2–3s sliding window, outputs arousal/valence/dominance + tremor flag. |
| **LLM** | BioMistral-7B, Llama-3.1-8B, or Meditron-7B (GGUF Q4_K_M) | llama.cpp with Metal. TTFT <500ms. Safety classifier in parallel. |
| **Orchestration** | Pipecat | Python, local transport. Services for each pipeline stage, plus a custom silence budget controller. |
| **TTS** | Piper (speed) or StyleTTS2 (emotion control) | Lightweight voices, adjustable pacing. Vocabulary mirroring via phoneme dictionary. |

Target device: **MacBook Air M4 with 16+ GB RAM**. (24 GB recommended for simultaneous large model + TTS.)

---

## Getting Started

### Prerequisites

- macOS 15+ on Apple Silicon (M1–M4).
- Python 3.11+.
- Homebrew (for whisper.cpp / llama.cpp dependencies).

### Installation

```bash
git clone https://github.com/ronit22203/medical-voice-assistant.git
cd medical-voice-assistant

# Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt

# Download models (automatic on first run, or scripted)
python scripts/download_models.py
```

### Quick start

```bash
python run_agent.py --profile clinician_empathetic
```

This launches the voice agent with a default silence budget config and prosody gating. Speak into your MacBook mic. The terminal will show real-time transcript, prosodic features, and the enriched LLM prompt.

---

## Silence Budget Configuration

The behaviour that makes this clinically distinct is **per-question-type silence tolerance**. Defined in `config/silence_budgets.yaml`:

```yaml
question_types:
  factual_recall:         # "When did the pain start?"
    max_silence_ms: 4500
    early_cutoff_on_semantic_completion: true

  emotional_disclosure:   # "How has your mood been?"
    max_silence_ms: 7000
    early_cutoff_on_semantic_completion: false  # let them fill the space

  suicide_screening:      # "Have you had thoughts of harming yourself?"
    max_silence_ms: 12000
    early_cutoff_on_semantic_completion: false
    suppress_fillers: true
    minimal_acknowledgment_only: true
```

The `silence_budget_controller` service inside Pipecat reads the type of the last agent question (tagged by the LLM) and enforces these timings.

---

## Prosody-Gated Empathy

The hardest open problem this project tackles: **emotional state tracking that actually gates the LLM prompt in real time**.

1. As the patient speaks, openSMILE (or SpeechBrain) computes prosodic features every 500ms from the trailing audio window.
2. When VAD marks silence, a rule engine maps the vector to a clinical qualifier:
   - `tremor > 0.6` + `speech rate slowdown > 30%` → `[DISTRESS]`
   - `pitch flat` + `low energy` → `[FLAT_AFFECT]`
   - `long pause with breathiness` → `[DISCLOSURE_HESITANCY]`
3. That qualifier is injected into the LLM system prompt **before the transcript is attached**.
4. Example prompt transformation:

```text
BASE: You are a clinical voice assistant. Use warm, simple language.

AFTER GATING:
[CLINICAL CONTEXT: Patient's voice shows DISTRESS markers — increased tremor, slowing pace, shallow breath. Respond with short sentences, mirror their vocabulary, validate affect before any information gathering. Do not offer solutions yet.]
You are a clinical voice assistant...
```

This is how the agent becomes *actually* empathetic, not just playing an empathetic script.

---

## Deliberate Latency

After the LLM returns a response, the agent does not speak immediately. It inserts a **configurable post-generation pause** (default 350ms). This is not pipeline latency — it’s intentional silence. It tells the patient, “what you said was heavy enough that I’m pausing.”

TTFT junkies will hate this. It is clinically necessary.

---

## Limitations (and what’s next)

- **Not FDA-cleared.** This is a research framework and clinical simulation tool, not a medical device.
- The emotional gating rules are heuristic; future work will replace them with a learned model trained on annotated clinical conversations.
- Vocabulary mirroring is currently pattern-based; a small fine-tuned model would make it robust to more dialects and medical literacy levels.
- Multi-turn state (tracking “the bad one” across a conversation) is in progress using a local key-value store.
- Accent/dialect prosody mapping is currently English-centric. We need per-language baseline models.

---

## Repo Structure

### medical-voice-assistant

```
medical-voice-assistant/
├── README.md
├── pyproject.toml
├── .env.example
│
├── config/
│   ├── agents/
│   │   ├── empathetic.yaml
│   │   └── triage.yaml
│   ├── silence_budgets.yaml
│   ├── prosody_rules.yaml
│   ├── prompts.yaml
│   └── topology.yaml
│
├── scripts/
│   ├── fetch_models.py               # Download + quantize Whisper, Llama, Piper, prosody
│   ├── benchmark.py                  # Latency + memory profiling on Apple Silicon
│   └── annotate_silence.py           # Annotate silence events for budget tuning
│
├── models/                           # gitignored
│   ├── whisper/
│   ├── llama/
│   ├── piper/
│   └── prosody/
│
├── src/
│   └── mva/
│       ├── __init__.py
│       ├── main.py                   # CLI entry point, run_agent orchestration
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── orchestrator.py       # Pipecat assembly, transport, thread wiring
│       │   └── state.py              # Multi-turn context, patient term memory
│       │
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── capture.py            # Mic input, device selection
│       │   ├── player.py             # Speaker output, buffer management
│       │   ├── vad.py                # Silero; emits speech/silence/hesitation events
│       │   └── utils.py              # Resampling, chunking, format helpers
│       │
│       ├── stt/
│       │   ├── __init__.py
│       │   ├── engine.py             # faster-whisper / whisper.cpp, disfluency preservation
│       │   └── vocab.py              # Runtime medical term weight injection
│       │
│       ├── signal/
│       │   ├── __init__.py
│       │   ├── prosody.py            # openSMILE GeMAPS streaming feature extraction
│       │   ├── emotion.py            # Arousal/valence/dominance via SpeechBrain
│       │   └── tremor.py             # Jitter/shimmer, micro-tremor detection
│       │
│       ├── clinical/
│       │   ├── __init__.py
│       │   ├── silence.py            # Budget state machine + question-type classifier
│       │   ├── prosody_gate.py       # Feature vector → clinical qualifier mapping
│       │   ├── disclosure.py         # Oblique disclosure detection, minimal ack logic
│       │   ├── vocab_mirror.py       # Register adaptation, patient term tracking
│       │   └── safety.py             # Parallel suicidal ideation classifier
│       │
│       ├── llm/
│       │   ├── __init__.py
│       │   ├── engine.py             # llama.cpp, Metal offload, context window mgmt
│       │   └── prompt.py             # System prompt assembly with injected clinical ctx
│       │
│       ├── tts/
│       │   ├── __init__.py
│       │   ├── piper.py              # Lightweight local TTS
│       │   ├── styletss.py           # Emotion-controllable alternative
│       │   └── pause.py              # Post-LLM deliberate silence insertion
│       │
│       └── infra/
│           ├── __init__.py
│           ├── logging.py
│           ├── metrics.py            # Latency, silence hit rates, prosody timestamps
│           └── device.py             # ANE/Metal telemetry, memory pressure monitoring
│
├── tests/
│   ├── conftest.py
│   ├── audio/
│   │   └── test_vad.py
│   ├── stt/
│   │   └── test_vocab.py
│   ├── signal/
│   │   └── test_prosody.py
│   ├── clinical/
│   │   ├── test_silence.py
│   │   ├── test_prosody_gate.py
│   │   ├── test_disclosure.py
│   │   └── test_safety.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── fixtures/
│       ├── audio/
│       └── mocks/
│
├── docs/
│   ├── architecture.md
│   ├── clinical_design.md
│   ├── local_setup.md
│   └── api.md
│
└── examples/
    ├── basic_agent.py
    ├── silence_demo.py
    └── prosody_plot.py
```

## Roadmap

- [ ] Real-time visual feedback (prosody trace, silence timer) for clinicians observing.
- [ ] Multi-party mode (clinician + patient + agent).
- [ ] Integration with local EHR sandbox for simulated encounter note generation.
- [ ] Quantized StyleTTS2 pipeline for controllable emotional expression.
- [ ] Learned silence budget policy (RL from expert clinician feedback).

---

## Contributing

This project needs clinicians, voice researchers, and MLOps engineers who believe listening is more than latency. Areas where help is especially wanted:

- Clinical validation of prosody-to-empathy mapping rules.
- Low-resource language prosody models.
- Psychiatric safety classifiers for edge cases.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and first-issue suggestions.

---

## License

MIT. Build with it. Deploy it where patients are.
