# Medical Voice Pipeline — Linux GPU Benchmark
## Hardware: NVIDIA RTX 4000 Ada (20GB) @ $0.26/hr

| Component | Latency | Backend |
|-----------|---------|---------|
| STT (Whisper base) | 0.54s | CUDA |
| LLM (qwen2.5:7b) | ~1.0s | Ollama/CUDA |
| TTS (Piper lessac-medium) | ~3-4s (12 words) | CPU ONNX |
| **Total** | **~5s** | |

## Open items
- F-068: cuDNN not available for onnxruntime-gpu CUDA TTS
- Fix: install CUDA 12 toolkit or switch to Coqui XTTS
