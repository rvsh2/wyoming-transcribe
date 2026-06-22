#!/usr/bin/env python3
"""
Cohere Transcribe HTTP debug server backed by the shared Cohere runtime.
"""

from __future__ import annotations

import argparse
import html
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse

from cohere_wyoming.audio import is_effectively_silent, read_audio_to_numpy
from cohere_wyoming.enrollment import EnrollmentError, EnrollmentStore
from cohere_wyoming.transcriber import (
    CohereTranscriber,
    LANGUAGE_ALIASES,
    SUPPORTED_LANGUAGES,
)
from cohere_wyoming.speaker_id import SpeakerRegistry
from cohere_wyoming.vad import VadConfig


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cohere-transcribe-server")

service = CohereTranscriber(
    vad_config=VadConfig.from_env(),
    speaker_registry=SpeakerRegistry.from_env(),
)
INDEX_TEMPLATE_PATH = Path(__file__).resolve().parent / "cohere_wyoming" / "templates" / "index.html"
IGNORED_WHISPER_PARAMS = ("temperature_inc", "prompt", "encode", "no_timestamps")


def sync_legacy_globals() -> None:
    """Keep legacy module globals available for compatibility and tests."""
    global model, processor, device, model_id, default_language, model_backend
    model = service.model
    processor = service.processor
    device = service.device
    model_id = service.model_name
    default_language = service.default_language
    model_backend = service.backend


sync_legacy_globals()


def load_model(model_name: str):
    service.load(model_name)
    sync_legacy_globals()


async def read_upload_audio(file: UploadFile) -> tuple[np.ndarray, int]:
    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        return read_audio_to_numpy(file_bytes, file.filename or "audio")
    except HTTPException:
        raise
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Error reading audio: {err}") from err


def resolve_language(language: Optional[str]) -> str:
    if language is None:
        return service.default_language

    resolved = language.strip().lower()
    if resolved == "auto":
        return service.default_language

    resolved = LANGUAGE_ALIASES.get(resolved, resolved)
    if resolved not in SUPPORTED_LANGUAGES:
        logger.warning(
            "Language '%s' not supported by Cohere Transcribe. Falling back to '%s'.",
            resolved,
            service.default_language,
        )
        return service.default_language

    return resolved


def ensure_model_loaded() -> None:
    if not service.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")


def log_ignored_whisper_options(
    *,
    temperature_inc: float,
    prompt: Optional[str],
    encode: Optional[bool],
    no_timestamps: Optional[bool],
    translate: Optional[bool],
) -> None:
    ignored: list[str] = []
    if temperature_inc != 0.2:
        ignored.append("temperature_inc")
    if prompt:
        ignored.append("prompt")
    if encode is not None and encode is not True:
        ignored.append("encode")
    if no_timestamps:
        ignored.append("no_timestamps")
    if translate:
        logger.warning("Translate mode is not supported by Cohere Transcribe - ignoring")
    if ignored:
        logger.info(
            "Accepted whisper.cpp compatibility parameters without applying them: %s",
            ", ".join(ignored),
        )


def build_segments(result: dict) -> list[dict]:
    """Expose diarized speaker segments; fall back to one whole-utterance segment."""
    segments = result.get("segments") or []
    if segments:
        return [
            {
                "id": index,
                "speaker": segment.get("speaker"),
                "start": segment.get("start", 0.0),
                "end": segment.get("end", result["duration"]),
                "text": segment.get("text", ""),
            }
            for index, segment in enumerate(segments)
        ]
    return [{"id": 0, "start": 0.0, "end": result["duration"], "text": result["text"]}]


def format_timestamp(duration: float, *, srt: bool) -> str:
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    millis = int((duration % 1) * 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def format_whisper_response(response_format: str, result: dict):
    text = result["text"]
    duration = result["duration"]
    if response_format == "text":
        return PlainTextResponse(text + "\n")
    if response_format == "srt":
        content = (
            "1\n"
            f"00:00:00,000 --> {format_timestamp(duration, srt=True)}\n"
            f"{text}\n\n"
        )
        return PlainTextResponse(content, media_type="text/plain")
    if response_format == "vtt":
        content = (
            "WEBVTT\n\n"
            f"00:00:00.000 --> {format_timestamp(duration, srt=False)}\n"
            f"{text}\n\n"
        )
        return PlainTextResponse(content, media_type="text/vtt")
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result["language"],
                "duration": duration,
                "text": text,
                "segments": build_segments(result),
            }
        )
    return JSONResponse({"text": text})


def format_openai_response(response_format: str, result: dict):
    text = result["text"]
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result["language"],
                "duration": result["duration"],
                "text": text,
                "segments": build_segments(result),
            }
        )
    return JSONResponse({"text": text})


def health_payload() -> dict:
    sync_legacy_globals()
    return service.health_payload()


def render_supported_language_badges() -> str:
    return "".join(
        f'<span class="badge">{html.escape(lang)}</span>'
        for lang in sorted(SUPPORTED_LANGUAGES)
    )


def render_language_options() -> str:
    options: list[str] = []
    for lang in sorted(SUPPORTED_LANGUAGES):
        selected = " selected" if lang == service.default_language else ""
        options.append(
            f'<option value="{html.escape(lang)}"{selected}>{html.escape(lang)}</option>'
        )
    return "".join(options)


def render_compatibility_notes() -> str:
    notes = [
        (
            "<strong>Full request/response compatibility:</strong> basic whisper.cpp "
            "multipart requests and JSON/text/subtitle response shapes"
        ),
        (
            "<strong>Compatibility-only parameters:</strong> "
            "<code>temperature_inc</code>, <code>prompt</code>, <code>encode</code>, "
            "<code>no_timestamps</code>"
        ),
        (
            "<strong>Diarization:</strong> per-speaker segments with timestamps "
            "(<code>verbose_json</code>); transcript text is prefixed per speaker"
        ),
        (
            "<strong>Not supported:</strong> translation, auto language detection"
        ),
    ]
    return "".join(f"<li>{note}</li>" for note in notes)


def render_index_page() -> str:
    template = INDEX_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "__MODEL_ID__": html.escape(service.model_name or "not loaded"),
        "__DEVICE__": html.escape(str(service.device) if service.device is not None else "unknown"),
        "__MODEL_BACKEND__": html.escape(service.backend),
        "__SUPPORTED_LANGUAGE_BADGES__": render_supported_language_badges(),
        "__LANGUAGE_OPTIONS__": render_language_options(),
        "__COMPATIBILITY_NOTES__": render_compatibility_notes(),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def transcribe_audio(
    audio_data: np.ndarray,
    sr: int = 16000,
    language: str = "en",
    temperature: float = 0.0,
) -> dict:
    return service.transcribe_pcm(
        audio_data,
        sample_rate=sr,
        language=language,
        temperature=temperature,
    ).asdict()


def run_transcription_request(
    *,
    audio_data: np.ndarray,
    sr: int,
    language: Optional[str],
    temperature: float,
) -> dict:
    resolved_language = resolve_language(language)
    duration = round(len(audio_data) / sr, 2) if sr else 0.0

    if is_effectively_silent(audio_data):
        logger.info("No speech detected above silence threshold; returning empty transcription")
        return {
            "text": "",
            "language": resolved_language,
            "duration": duration,
            "processing_time": 0.0,
        }

    try:
        return transcribe_audio(
            audio_data,
            sr=sr,
            language=resolved_language,
            temperature=temperature,
        )
    except Exception as err:
        logger.error("Transcription error: %s", err, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {err}") from err


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(
    title="Cohere Transcribe Server",
    description="whisper.cpp-compatible API powered by syvai/cohere-transcribe-diarize",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return render_index_page()


@app.get("/health")
async def health():
    return JSONResponse(health_payload())


@app.post("/inference")
async def inference(
    file: UploadFile = File(...),
    temperature: float = Form(0.0),
    temperature_inc: float = Form(0.2),
    response_format: str = Form("json"),
    language: Optional[str] = Form(None),
    encode: Optional[bool] = Form(True),
    no_timestamps: Optional[bool] = Form(False),
    prompt: Optional[str] = Form(None),
    translate: Optional[bool] = Form(False),
):
    ensure_model_loaded()
    log_ignored_whisper_options(
        temperature_inc=temperature_inc,
        prompt=prompt,
        encode=encode,
        no_timestamps=no_timestamps,
        translate=translate,
    )
    audio_data, sr = await read_upload_audio(file)
    result = run_transcription_request(
        audio_data=audio_data,
        sr=sr,
        language=language,
        temperature=temperature,
    )
    return format_whisper_response(response_format, result)


@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None, alias="model"),
    language: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: float = Form(0.0),
    prompt: Optional[str] = Form(None),
):
    del model_name, prompt
    ensure_model_loaded()
    audio_data, sr = await read_upload_audio(file)
    result = run_transcription_request(
        audio_data=audio_data,
        sr=sr,
        language=language,
        temperature=temperature,
    )
    return format_openai_response(response_format, result)


@app.post("/load")
async def load(model_path: Optional[str] = Form(None, alias="model")):
    if model_path is None or model_path.strip() == "":
        raise HTTPException(status_code=400, detail="No model path provided")

    try:
        load_model(model_path.strip())
        return JSONResponse({"status": "ok", "model": model_path.strip()})
    except Exception as err:
        logger.error("Failed to load model: %s", err, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {err}") from err


def enrollment_store() -> EnrollmentStore:
    return EnrollmentStore(service.speaker_registry.enrollment_dir)


@app.get("/speakers")
async def list_speakers():
    return JSONResponse(
        {
            "speakers": enrollment_store().list_speakers(),
            "speaker_id": service.speaker_registry.status_payload(),
        }
    )


@app.post("/speakers")
async def create_speaker(name: str = Form(...)):
    try:
        created = enrollment_store().create_speaker(name)
    except EnrollmentError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return JSONResponse({"status": "ok", "name": created})


@app.delete("/speakers/{name}")
async def delete_speaker(name: str):
    try:
        enrollment_store().delete_speaker(name)
    except EnrollmentError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return JSONResponse({"status": "ok"})


@app.post("/speakers/{name}/samples")
async def add_speaker_sample(name: str, file: UploadFile = File(...)):
    data = await file.read()
    try:
        sample = enrollment_store().add_sample(name, data, file.filename or "audio")
    except EnrollmentError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return JSONResponse({"status": "ok", "sample": sample})


@app.get("/speakers/{name}/samples/{sample_id}")
async def get_speaker_sample(name: str, sample_id: str):
    try:
        path = enrollment_store().sample_path(name, sample_id)
    except EnrollmentError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return FileResponse(str(path), media_type="audio/wav", filename=sample_id)


@app.delete("/speakers/{name}/samples/{sample_id}")
async def delete_speaker_sample(name: str, sample_id: str):
    try:
        enrollment_store().delete_sample(name, sample_id)
    except EnrollmentError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return JSONResponse({"status": "ok"})


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cohere Transcribe Server - whisper.cpp compatible API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Hostname/IP address for the server")
    parser.add_argument("--port", type=int, default=8580, help="Port number for the server")
    parser.add_argument(
        "--no-load-model",
        action="store_true",
        default=False,
        help="Skip loading the ASR model (serve the enrollment UI / API only)",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="syvai/cohere-transcribe-diarize",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="en",
        help="Default spoken language (ISO 639-1 code)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=4,
        help="Number of threads (sets torch threads)",
    )
    parser.add_argument(
        "-ng",
        "--no-gpu",
        action="store_true",
        default=False,
        help="Disable GPU, use CPU only",
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
    return parser.parse_args()


def main():
    args = parse_args()
    default_vad_config = VadConfig.from_env()
    service.set_default_language(args.language)
    service.prefer_device = "cpu" if args.no_gpu else None
    service.set_model_name(args.model)
    service.set_vad_config(
        VadConfig.from_env(
            enabled=False if args.disable_vad else default_vad_config.enabled,
            threshold=args.vad_threshold if args.vad_threshold is not None else default_vad_config.threshold,
        )
    )
    sync_legacy_globals()
    torch.set_num_threads(args.threads)

    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    print(f"  Model:    {service.model_name}")
    print(f"  Language: {service.default_language}")
    print(f"  Host:     {args.host}:{args.port}")
    print(f"  Threads:  {args.threads}")
    print(f"  GPU:      {'disabled' if args.no_gpu else 'auto'}")

    if args.no_load_model:
        logger.info("Skipping ASR model load (enrollment UI / API only mode)")
    else:
        load_model(args.model)

    logger.info("Starting server on %s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
