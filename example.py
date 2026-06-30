import asyncio
import sys
from pathlib import Path
import yaml
from loguru import logger
from pipecat.audio.utils import calculate_audio_volume
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
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

def load_config(path: str = "config/settings.yaml") -> dict:
    return yaml.safe_load((Path(__file__).parent / path).read_text())


logger.remove()
logger.add(sys.stderr, level="WARNING", filter=lambda r: r["name"].startswith("pipecat"))
logger.add(sys.stderr, level="DEBUG",   filter=lambda r: not r["name"].startswith("pipecat"))

_SKIP_TYPES = (InputAudioRawFrame, AudioRawFrame, TTSAudioRawFrame)


class EchoGate(FrameProcessor):
    """Drops mic audio while the bot is speaking to prevent echo-triggered interruptions."""

    def __init__(self):
        super().__init__()
        self._bot_speaking = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            print("[GATE]  bot speaking — mic gated", flush=True)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            print("[GATE]  bot done — mic open", flush=True)

        if self._bot_speaking and isinstance(frame, InputAudioRawFrame):
            return  # discard mic audio while bot is talking

        await self.push_frame(frame, direction)


class ThinkStripper(FrameProcessor):
    """Buffers streamed LLM tokens and discards <think>...</think> blocks before TTS."""

    def __init__(self):
        super().__init__()
        self._buffer = ""
        self._streaming = False  # True once think block is fully consumed

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._buffer = ""
            self._streaming = False
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            if self._streaming:
                await self.push_frame(frame, direction)
                return

            self._buffer += frame.text

            if "</think>" in self._buffer:
                after = self._buffer.split("</think>", 1)[1].lstrip("\n ")
                self._buffer = ""
                self._streaming = True
                if after:
                    await self.push_frame(LLMTextFrame(after), direction)

            elif "<think>" not in self._buffer:
                # No think block — non-reasoning response, stream immediately
                text = self._buffer
                self._buffer = ""
                self._streaming = True
                if text:
                    await self.push_frame(LLMTextFrame(text), direction)

            # else: inside <think> block, keep buffering silently
            return

        await self.push_frame(frame, direction)


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
    cfg = load_config()
    meta = cfg.get("meta", {})
    if meta:
        print(f"[CONFIG] {meta.get('name', '—')} — {meta.get('description', '')}", flush=True)

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=cfg["audio"]["input"]["sample_rate"],
            audio_out_sample_rate=cfg["audio"]["output"]["sample_rate"],
        )
    )

    _vad_keys = {"confidence", "min_volume", "start_secs", "stop_secs"}
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(**{k: v for k, v in cfg["vad"].items() if k in _vad_keys})
        )
    )

    stt = WhisperSTTService(
        device=cfg["stt"]["device"],
        compute_type=cfg["stt"]["compute_type"],
        settings=WhisperSTTService.Settings(model=cfg["stt"]["model"])
    )

    llm = OpenAILLMService(
        api_key=cfg["llm"]["api_key"],
        base_url=cfg["llm"]["base_url"],
        settings=OpenAILLMService.Settings(model=cfg["llm"]["model"])
    )

    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=cfg["tts"]["voice"])
    )

    sys_prompt = cfg["system_prompt"].strip()
    context = LLMContext(messages=[{"role": "system", "content": sys_prompt}])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline([
        transport.input(),
        EchoGate(),
        MicMonitor(every=50),
        vad,
        FrameTracer("AFTER-VAD"),
        stt,
        FrameTracer("AFTER-STT"),
        user_aggregator,
        FrameTracer("AFTER-AGGREGATOR"),
        llm,
        FrameTracer("AFTER-LLM"),
        ThinkStripper(),
        tts,
        FrameTracer("AFTER-TTS"),
        transport.output(),
        assistant_aggregator,
    ])

    runner = WorkerRunner()
    worker = PipelineWorker(pipeline)
    await runner.add_workers(worker)

    print("\n[READY] Mic is ON — speak now.\n", flush=True)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
