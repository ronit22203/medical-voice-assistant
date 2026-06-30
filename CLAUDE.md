# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Locally-run medical voice assistant for Apple Silicon (M1–M4). All inference runs on-device — no audio or text leaves the machine. The design treats silence and prosody as clinical signals, not noise.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires macOS 15+ on Apple Silicon and Python 3.11+. LLM inference depends on a running [LM Studio](https://lmstudio.ai) instance at `http://localhost:1234/v1` serving a compatible model (default config targets `meta-llama-3.1-8b-instruct`).

## Running

```bash
# Minimal working pipeline (example.py)
python example.py

# Full agent with silence budgets and prosody gating (target entrypoint, not yet built)
python run_agent.py --profile clinician_empathetic
```

## Architecture

The pipeline is a sequential Pipecat chain:

```
Mic → Silero VAD → Whisper STT → LLM (via OpenAI-compat API) → Piper TTS → Speaker
```

`example.py` is the working baseline. The planned production layout lives under `src/mva/` (described in `README.md`) and is not yet implemented.

Key architectural decisions:
- **Silence budgets** are per-question-type (factual: 4.5s, emotional: 7s, suicide screening: 12s), not global. Defined in `config/silence_budgets.yaml` (planned).
- **Prosody gating** runs openSMILE/SpeechBrain on a 500ms sliding window and injects a clinical qualifier (`[DISTRESS]`, `[FLAT_AFFECT]`, `[DISCLOSURE_HESITANCY]`) into the LLM system prompt before each response.
- **Deliberate post-LLM pause** (default 350ms) is intentional, not pipeline lag — it signals the patient their words registered.
- **LLM is accessed via OpenAI-compatible API** pointed at localhost, so swapping the local model (BioMistral, Llama-3.1-8B, Meditron) requires only changing the model ID in the config.
- The Piper voice model (`en_US-lessac-medium.onnx`) is checked into the repo.

## Planned Module Map (`src/mva/`)

| Module | Role |
|--------|------|
| `pipeline/orchestrator.py` | Pipecat assembly, transport, thread wiring |
| `audio/vad.py` | Silero VAD; emits speech/silence/hesitation events |
| `stt/engine.py` | faster-whisper with disfluency preservation |
| `signal/prosody.py` | openSMILE GeMAPS streaming features |
| `clinical/silence.py` | Budget state machine + question-type classifier |
| `clinical/prosody_gate.py` | Feature vector → clinical qualifier |
| `clinical/disclosure.py` | Oblique disclosure detection |
| `llm/prompt.py` | System prompt assembly with injected clinical context |
| `tts/pause.py` | Post-LLM deliberate silence insertion |
