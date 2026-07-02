"""CLI entrypoint for the Cohere Wyoming server."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Optional

import torch

from .handler import CohereWyomingEventHandler
from .speaker_id import SpeakerRegistry
from .transcriber import CohereTranscriber, SUPPORTED_LANGUAGES
from .vad import VadConfig
from .wyoming_protocol import (
    AsrModel,
    AsrProgram,
    AsyncServer,
    Attribution,
    Info,
    WYOMING_AVAILABLE,
)


LOGGER = logging.getLogger("cohere-wyoming")


def build_info(transcriber: CohereTranscriber) -> Info:
    model = AsrModel(
        name=transcriber.model_name,
        description="Cohere Transcribe speech-to-text model with speaker diarization",
        attribution=Attribution(
            name="syvai / Cohere",
            url="https://huggingface.co/syvai/cohere-transcribe-diarize",
        ),
        installed=True,
        languages=sorted(SUPPORTED_LANGUAGES),
        version="0.1.0",
    )
    program = AsrProgram(
        name="cohere-transcribe",
        description="Wyoming protocol server backed by Cohere Transcribe",
        attribution=Attribution(
            name="Cohere Wyoming",
            url="https://github.com/rhasspy/wyoming-faster-whisper",
        ),
        installed=True,
        models=[model],
        version="0.1.0",
    )
    return Info(asr=[program])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cohere Wyoming server for Home Assistant",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--uri",
        default="tcp://0.0.0.0:10300",
        help="Wyoming server URI",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="syvai/cohere-transcribe-diarize",
        help="Hugging Face model ID or local path",
    )
    parser.add_argument(
        "-l",
        "--language",
        default="en",
        help="Default spoken language (ISO 639-1 code)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force runtime device, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Number of torch CPU threads",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--disable-vad",
        action="store_true",
        default=False,
        help="Disable Silero VAD and use only the fallback silence detector",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="Override Silero VAD detection threshold",
    )
    return parser.parse_args(argv)


async def serve(args: argparse.Namespace) -> None:
    if not WYOMING_AVAILABLE:
        raise RuntimeError(
            "The 'wyoming' package is not installed. Run 'uv sync' before starting the Wyoming server."
        )

    default_vad_config = VadConfig.from_env()
    transcriber = CohereTranscriber(
        model_name=args.model,
        default_language=args.language,
        prefer_device=args.device,
        vad_config=VadConfig.from_env(
            enabled=False if args.disable_vad else default_vad_config.enabled,
            threshold=args.vad_threshold if args.vad_threshold is not None else default_vad_config.threshold,
        ),
        speaker_registry=SpeakerRegistry.from_env(device=args.device),
    )
    transcriber.load()

    # Warm the speaker registry at startup (loads ECAPA + builds profiles) so the
    # first transcription does not block the asyncio event loop on model download.
    if transcriber.speaker_registry is not None and transcriber.speaker_registry.enabled:
        try:
            transcriber.speaker_registry.reload_if_changed()
            LOGGER.info("Speaker identification ready: %s", transcriber.speaker_registry.status_payload())
        except Exception as error:
            LOGGER.warning("Speaker registry warm-up failed: %s", error)

    # Warm the ASR/diarize model's generate path so the FIRST real transcription is
    # fast (CUDA kernels compiled at startup). Calls _generate_diarized directly to
    # bypass VAD; the dummy audio content is irrelevant.
    try:
        import time as _time

        import numpy as _np

        from .audio import pcm16le_to_float32

        # Warm the full request path with realistic input so the first real
        # request does not pay for lazy loads and shape-dependent CUDA work:
        # PCM conversion + 48 kHz stereo resample (soxr init), Silero VAD load,
        # and a full-length (30 s) generate pass.
        _t = _time.time()
        _pcm = (_np.random.randn(48000 * 2 * 2) * 300).astype("<i2").tobytes()
        pcm16le_to_float32(_pcm, sample_rate=48000, channels=2, width=2)
        LOGGER.info("Audio conversion warmup done in %.1fs", _time.time() - _t)

        _t = _time.time()
        _dummy = (_np.random.randn(16000).astype("float32") * 0.02)
        transcriber.vad_detector.detect_speech(_dummy, sample_rate=16000)
        LOGGER.info("VAD warmup done in %.1fs", _time.time() - _t)

        _t = _time.time()
        _dummy_full = (_np.random.randn(30 * 16000).astype("float32") * 0.02)
        transcriber._generate_diarized(_dummy_full, 16000, transcriber.default_language, 0.0)
        LOGGER.info("ASR model warmup done in %.1fs", _time.time() - _t)
    except Exception as error:  # warmup must never block serving
        LOGGER.warning("ASR model warmup failed (continuing): %s", error)

    info_event = build_info(transcriber).event()
    server = AsyncServer.from_uri(args.uri)

    LOGGER.info("Model: %s", transcriber.model_name)
    LOGGER.info("Language: %s", transcriber.default_language)
    LOGGER.info("Device: %s", transcriber.device)
    LOGGER.info("URI: %s", args.uri)
    LOGGER.info("Wyoming server ready")
    await server.run(
        lambda *handler_args, **handler_kwargs: CohereWyomingEventHandler(
            transcriber,
            info_event,
            *handler_args,
            **handler_kwargs,
        )
    )


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.set_num_threads(args.threads)
    asyncio.run(serve(args))


if __name__ == "__main__":
    main()
