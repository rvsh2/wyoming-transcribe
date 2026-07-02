#!/usr/bin/env python3
"""
Cohere Transcribe HTTP debug server backed by the shared Cohere runtime.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import io
import logging
import os
import secrets
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
)

from cohere_wyoming.audio import is_effectively_silent, read_audio_to_numpy
from cohere_wyoming.enrollment import SPEAKER_ROLES, EnrollmentError, EnrollmentStore
from cohere_wyoming.history import RecognitionLog
from cohere_wyoming.pending import PendingError, PendingStore
from cohere_wyoming.transcriber import (
    CohereTranscriber,
    LANGUAGE_ALIASES,
    SPEAKER_LABEL,
    SUPPORTED_LANGUAGES,
)
from cohere_wyoming.settings import SPEAKER_TEXT_MODES
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
        built = []
        for index, segment in enumerate(segments):
            start = segment.get("start", 0.0)
            end = segment.get("end", result["duration"])
            built.append(
                {
                    "id": index,
                    "speaker": segment.get("speaker"),
                    "name": segment.get("name"),
                    "score": segment.get("score"),
                    "start": start,
                    # The model can emit end < start; clamp so consumers (SRT/VTT)
                    # never get a negative-duration cue.
                    "end": max(start, end),
                    "text": segment.get("text", ""),
                }
            )
        return built
    return [{"id": 0, "start": 0.0, "end": result["duration"], "text": result["text"]}]


def format_timestamp(duration: float, *, srt: bool) -> str:
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)
    millis = int((duration % 1) * 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def render_cue_text(segment: dict) -> str:
    """Cue text with the enrolled name or anonymous speaker label prefixed."""
    text = segment.get("text", "")
    name = segment.get("name")
    if name:
        return f"{name}: {text}"
    speaker = segment.get("speaker")
    if speaker is not None:
        return f"{SPEAKER_LABEL} {speaker}: {text}"
    return text


def format_subtitles(result: dict, *, srt: bool) -> str:
    """Render one cue per diarized segment (falls back to one whole-file cue)."""
    cues = []
    for index, segment in enumerate(build_segments(result), start=1):
        start = format_timestamp(segment["start"], srt=srt)
        end = format_timestamp(segment["end"], srt=srt)
        cues.append(f"{index}\n{start} --> {end}\n{render_cue_text(segment)}\n\n")
    header = "" if srt else "WEBVTT\n\n"
    return header + "".join(cues)


def format_whisper_response(response_format: str, result: dict):
    text = result["text"]
    duration = result["duration"]
    if response_format == "text":
        return PlainTextResponse(text + "\n")
    if response_format == "srt":
        return PlainTextResponse(format_subtitles(result, srt=True), media_type="text/plain")
    if response_format == "vtt":
        return PlainTextResponse(format_subtitles(result, srt=False), media_type="text/vtt")
    if response_format == "verbose_json":
        return JSONResponse(
            {
                "task": "transcribe",
                "language": result["language"],
                "duration": duration,
                "text": text,
                "speaker": result.get("speaker"),
                "speaker_score": result.get("speaker_score"),
                "speaker_role": result.get("speaker_role"),
                "utterance_id": result.get("utterance_id"),
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
                "speaker": result.get("speaker"),
                "speaker_score": result.get("speaker_score"),
                "speaker_role": result.get("speaker_role"),
                "utterance_id": result.get("utterance_id"),
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
        "__ASR_AVAILABLE__": "true" if service.is_loaded() else "false",
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

# Optional bearer/X-API-Token auth for the management API. Off when API_TOKEN
# is unset (loopback-only publishing is then the safety boundary). "/" stays
# open as a static info page and "/health" must stay open for the container
# healthcheck.
API_TOKEN = os.environ.get("API_TOKEN", "").strip() or None
OPEN_PATHS = {"/", "/health"}


@app.middleware("http")
async def require_api_token(request: Request, call_next):
    if API_TOKEN is not None and request.url.path not in OPEN_PATHS:
        provided = request.headers.get("x-api-token", "")
        if not provided:
            authorization = request.headers.get("authorization", "")
            if authorization.lower().startswith("bearer "):
                provided = authorization[7:]
        if not secrets.compare_digest(provided, API_TOKEN):
            return JSONResponse(
                {"detail": "Invalid or missing API token"}, status_code=401
            )
    return await call_next(request)


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
    # Inference takes seconds; run it off the event loop so /health, the
    # enrollment UI and concurrent uploads stay responsive.
    result = await asyncio.to_thread(
        run_transcription_request,
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
    result = await asyncio.to_thread(
        run_transcription_request,
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
        await asyncio.to_thread(load_model, model_path.strip())
        return JSONResponse({"status": "ok", "model": model_path.strip()})
    except Exception as err:
        logger.error("Failed to load model: %s", err, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load model: {err}") from err


@app.get("/settings")
async def get_settings():
    settings = service.settings_store.load()
    return JSONResponse(
        {
            "speaker_text_mode": settings.speaker_text_mode,
            "speaker_text_modes": list(SPEAKER_TEXT_MODES),
        }
    )


@app.post("/settings")
async def update_settings(speaker_text_mode: str = Form(...)):
    try:
        settings = service.settings_store.save(speaker_text_mode=speaker_text_mode)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return JSONResponse({"status": "ok", "speaker_text_mode": settings.speaker_text_mode})


def enrollment_store() -> EnrollmentStore:
    return EnrollmentStore(service.speaker_registry.enrollment_dir)


def pending_store() -> PendingStore:
    return PendingStore.from_env(service.speaker_registry.enrollment_dir)


@app.get("/history")
async def recognition_history(limit: int = 50):
    """Recent transcription decisions (who was recognized, or utterance_id)."""
    log = RecognitionLog.from_env(service.speaker_registry.enrollment_dir)
    entries = log.recent(limit=max(1, min(limit, 500))) if log is not None else []
    return JSONResponse({"entries": entries})


# Transient/operational files that do not belong in an enrollment backup:
# pending clips are a ring buffer, and restoring an old recognition log would
# overwrite a newer one.
_EXPORT_EXCLUDED = (".pending", ".history.jsonl", ".history.jsonl.tmp")


def _build_export_archive() -> bytes:
    """tar.gz of the enrollment dir (samples, roles, settings)."""
    root = Path(service.speaker_registry.enrollment_dir)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if relative.parts and relative.parts[0] in _EXPORT_EXCLUDED:
                    continue
                if path.is_file():
                    archive.add(str(path), arcname=str(relative))
    return buffer.getvalue()


def _restore_export_archive(data: bytes) -> int:
    """Extract a backup into the enrollment dir; rejects unsafe paths."""
    root = Path(service.speaker_registry.enrollment_dir)
    root.mkdir(parents=True, exist_ok=True)
    restored = 0
    try:
        archive = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
    except tarfile.TarError as error:
        raise ValueError(f"Not a valid backup archive: {error}") from error
    with archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue  # directories are created below; links are ignored
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path in archive: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                continue
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read())
            restored += 1
    if restored == 0:
        raise ValueError("Archive contains no files")
    return restored


@app.get("/export")
async def export_enrollment():
    data = await asyncio.to_thread(_build_export_archive)
    return Response(
        content=data,
        media_type="application/gzip",
        headers={"Content-Disposition": 'attachment; filename="speakers-backup.tar.gz"'},
    )


@app.post("/import")
async def import_enrollment(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty archive upload")
    try:
        restored = await asyncio.to_thread(_restore_export_archive, data)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return JSONResponse({"status": "ok", "files": restored})


def public_clip(clip: dict) -> dict:
    """Pending-clip metadata without the embedding vector."""
    return {key: value for key, value in clip.items() if key != "embedding"}


@app.get("/pending")
async def list_pending():
    """Unrecognized-voice clips grouped by voice (cluster = same speaker)."""
    clusters = pending_store().clusters()
    return JSONResponse(
        {
            "count": sum(len(cluster) for cluster in clusters),
            "clusters": [
                {"cluster": index, "clips": [public_clip(clip) for clip in cluster]}
                for index, cluster in enumerate(clusters)
            ],
        }
    )


@app.get("/pending/{utterance_id}/audio")
async def get_pending_audio(utterance_id: str):
    try:
        path = pending_store().audio_path(utterance_id)
    except PendingError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return FileResponse(str(path), media_type="audio/wav", filename=f"{utterance_id}.wav")


@app.delete("/pending/{utterance_id}")
async def delete_pending(utterance_id: str):
    try:
        pending_store().delete(utterance_id)
    except PendingError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    return JSONResponse({"status": "ok"})


@app.post("/speakers/{name}/samples/from-utterance/{utterance_id}")
async def claim_pending(name: str, utterance_id: str, include_cluster: bool = Form(True)):
    """Enroll pending clip(s) as samples of a (possibly new) person.

    With include_cluster (default), all pending clips of the same voice are
    claimed together, so the person's profile immediately has several samples.
    This is the endpoint an LLM pipeline calls after asking "who is speaking?".
    """
    store = pending_store()
    enrollment = enrollment_store()
    try:
        ids = store.cluster_members(utterance_id) if include_cluster else [utterance_id]
        samples = []
        for clip_id in ids:
            audio_bytes = store.audio_path(clip_id).read_bytes()
            samples.append(enrollment.add_sample(name, audio_bytes, f"{clip_id}.wav"))
            store.delete(clip_id)
    except PendingError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err
    except EnrollmentError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    return JSONResponse(
        {"status": "ok", "name": name, "claimed": ids, "samples": samples}
    )


@app.post("/speakers/{name}/role")
async def set_speaker_role(name: str, role: str = Form(...)):
    try:
        saved = enrollment_store().set_role(name, role)
    except EnrollmentError as err:
        status = 404 if "does not exist" in str(err) else 400
        raise HTTPException(status_code=status, detail=str(err)) from err
    return JSONResponse({"status": "ok", "name": name, "role": saved})


@app.get("/speakers")
async def list_speakers():
    speakers = enrollment_store().list_speakers()
    # Report enrolled people from disk, not the live registry profiles: this
    # (UI) process never loads embeddings, so registry._profiles is always empty.
    status = service.speaker_registry.status_payload()
    status["enrolled"] = [s["name"] for s in speakers if s["samples"]]
    return JSONResponse(
        {"speakers": speakers, "speaker_id": status, "roles": list(SPEAKER_ROLES)}
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
