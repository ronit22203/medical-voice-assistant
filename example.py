import asyncio
import sys
from loguru import logger
from pipecat.audio.utils import calculate_audio_volume
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

logger.remove()
logger.add(sys.stderr, level="WARNING", filter=lambda r: r["name"].startswith("pipecat"))
logger.add(sys.stderr, level="DEBUG",   filter=lambda r: not r["name"].startswith("pipecat"))

_SKIP_TYPES = (InputAudioRawFrame, AudioRawFrame, TTSAudioRawFrame)


class MicMonitor(FrameProcessor):
    """Prints mic volume every N audio frames."""

    def __init__(self, every: int = 50):
        super().__init__()
        self._count = 0
        self._every = every

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self._count += 1
            if self._count % self._every == 0:
                vol = calculate_audio_volume(frame.audio, frame.sample_rate)
                bar = "#" * int(vol * 40)
                print(f"[MIC-LEVEL] vol={vol:.3f}  |{bar:<40}|", flush=True)
        await self.push_frame(frame, direction)


class FrameTracer(FrameProcessor):
    """Prints every non-audio frame that passes through, labelled by stage."""

    def __init__(self, stage: str):
        super().__init__()
        self._stage = stage

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not isinstance(frame, _SKIP_TYPES):
            print(f"[{self._stage}] {type(frame).__name__}", flush=True)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            print("[VAD]   >>> speech detected <<<", flush=True)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            print("[VAD]   >>> speech ended, sending to Whisper <<<", flush=True)
        elif isinstance(frame, TranscriptionFrame):
            print(f'[STT]   >>> transcript: "{frame.text}" <<<', flush=True)
        elif isinstance(frame, TTSStartedFrame):
            print("[AUDIO] >>> playing response <<<", flush=True)
        elif isinstance(frame, TTSStoppedFrame):
            print("[AUDIO] >>> playback done <<<", flush=True)

        await self.push_frame(frame, direction)


async def main():
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(
                confidence=0.5,
                min_volume=0.3,
                start_secs=0.2,
                stop_secs=0.8,
            )
        )
    )

    stt = WhisperSTTService(
        device="cpu",
        compute_type="int8",
        settings=WhisperSTTService.Settings(model="medium")
    )

    llm = OpenAILLMService(
        api_key="lm-studio",
        base_url="http://localhost:1234/v1",
        settings=OpenAILLMService.Settings(model="meta-llama-3.1-8b-instruct")
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice="en_US-lessac-medium")
    )

    sys_prompt = (
        "You are a helpful, empathetic medical AI assistant. "
        "Keep responses brief, conversational, and clear for voice interaction."
    )
    context = LLMContext(messages=[{"role": "system", "content": sys_prompt}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        MicMonitor(every=50),
        vad,
        FrameTracer("AFTER-VAD"),
        stt,
        FrameTracer("AFTER-STT"),
        user_aggregator,
        FrameTracer("AFTER-AGGREGATOR"),
        llm,
        FrameTracer("AFTER-LLM"),
        tts,
        FrameTracer("AFTER-TTS"),
        transport.output(),
        assistant_aggregator,
    ])

    runner = WorkerRunner()
    worker = PipelineWorker(pipeline)
    await runner.add_workers(worker)

    print("\n[READY] Mic is ON — speak now. Watch for [VAD] events.\n", flush=True)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
