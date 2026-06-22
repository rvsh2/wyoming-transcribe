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
