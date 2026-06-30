# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Locally-run medical voice assistant for Apple Silicon (M1–M4). All inference runs on-device — no audio or text leaves the machine. The design treats silence and prosody as clinical signals, not noise.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires macOS 15+ on Apple Silicon and Python 3.11+. LLM inference requires a running [LM Studio](https://lmstudio.ai) instance at `http://localhost:1234/v1` with a model loaded. The Piper voice model (`.onnx`) is **not** checked in — download `en_US-lessac-medium.onnx` and its `.onnx.json` sidecar and place them in the repo root (or wherever Piper resolves the voice name).

## Running

```bash
python example.py
```

All tunable parameters live in `config/settings.yaml`. Edit that file to swap models, adjust VAD thresholds, or change the system prompt — no Python changes needed. The `meta.name` / `meta.notes` fields are printed at startup to self-document runs.

## Architecture

`example.py` is the single working entrypoint. It builds a sequential Pipecat 1.4.0 pipeline:

```
Mic → EchoGate → MicMonitor → VADProcessor → WhisperSTT → LLMContextAggregator
    → OpenAILLM → ThinkStripper → PiperTTS → Speaker → LLMContextAggregator
```

**Non-obvious wiring decisions learned the hard way:**

- **VAD is a standalone `VADProcessor`**, not a parameter of `LocalAudioTransport`. `LocalAudioTransportParams` has no `vad_analyzer` field — passing one there is silently ignored by Pydantic. The STT service listens for `VADUserStartedSpeakingFrame` / `VADUserStoppedSpeakingFrame`, which only `VADProcessor` emits.
- **`FrameProcessor` subclasses must call `await super().process_frame(frame, direction)` before `await self.push_frame(...)`**. The base class sets an internal `__started` flag on `StartFrame`; without it, `push_frame` silently drops every frame.
- **Sample rates must be set explicitly**: Whisper expects 16 kHz in, Piper `en_US-lessac-medium` outputs 22050 Hz. Mismatching produces sped-up/garbled audio.
- **`EchoGate`** drops `InputAudioRawFrame` while `BotStartedSpeakingFrame` is active. Without it, the speaker output is picked up by the mic, VAD fires, and `InterruptionFrame` cuts TTS mid-sentence.
- **`ThinkStripper`** sits between LLM and TTS. Reasoning models (Qwen, DeepSeek-R1) stream `<think>...</think>` tokens before the answer; this processor buffers until `</think>` then forwards only the post-block text. Non-reasoning models pass through immediately on the first token.
- **`LLMContextAggregatorPair`** returns two processors: `user_aggregator` goes before the LLM, `assistant_aggregator` goes after `transport.output()` at the end of the pipeline.

## Configuration (`config/settings.yaml`)

All fields map 1:1 to constructor arguments in `example.py`. The `vad` block is unpacked directly with `VADParams(**cfg["vad"])`, so adding or removing fields there must stay in sync with `VADParams`'s field names.

## Planned Module Map (`src/mva/`)

Not yet implemented. Described in `README.md`. The planned production layout includes:

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
