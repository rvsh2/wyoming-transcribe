"""Cohere transcription backend independent from HTTP/Wyoming transport."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .audio import is_effectively_silent, read_audio_to_numpy
from .speaker_id import SpeakerRegistry
from .vad import SileroVoiceActivityDetector, VadConfig


LOGGER = logging.getLogger("cohere-wyoming.transcriber")

# Diarize decoder prompt. The processor still needs language=, but its own
# decoder_input_ids are overridden with this sequence so the model emits speaker
# and timestamp tokens. "{lang}" is filled with the resolved ISO 639-1 code.
DIARIZE_PROMPT_TEMPLATE = (
    "<|startofcontext|>",
    "<|startoftranscript|>",
    "<|emo:undefined|>",
    "<|{lang}|>",
    "<|{lang}|>",
    "<|pnc|>",
    "<|noitn|>",
    "<|timestamp|>",
    "<|diarize|>",
)

# Hard per-pass limit of the diarize model; longer audio is split into windows.
MAX_CHUNK_SECONDS = 30

# Prefix used when rendering anonymous diarized speakers into a single transcript.
SPEAKER_LABEL = "Mówca"

_SEGMENT_RE = re.compile(
    r"<\|spltoken(\d+)\|>\s*<\|t:([0-9]+(?:\.[0-9]+)?)\|>(.*?)"
    r"(?=<\|t:[0-9]|<\|spltoken\d|<\|endoftext\|>|$)",
    re.DOTALL,
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


def chunk_audio(audio_data: np.ndarray, sample_rate: int, max_seconds: int = MAX_CHUNK_SECONDS):
    """Split audio into windows no longer than max_seconds, with second offsets."""
    max_samples = int(max_seconds * sample_rate)
    if max_samples <= 0 or len(audio_data) <= max_samples:
        return [(audio_data, 0.0)]

    chunks = []
    for start in range(0, len(audio_data), max_samples):
        chunk = audio_data[start : start + max_samples]
        if len(chunk) == 0:
            continue
        chunks.append((chunk, start / sample_rate))
    return chunks


def parse_diarized_output(raw: str, offset: float = 0.0) -> list[dict]:
    """Parse the diarize token stream into ordered speaker segments.

    Format (confirmed via spike): ``<|spltokenN|><|t:START|> text <|t:END|>`` repeated,
    terminated by ``<|endoftext|>``. Timestamps are offset by ``offset`` seconds so
    segments from later chunks land on the global timeline.
    """
    diarize_split = raw.split("<|diarize|>", 1)
    body = diarize_split[1] if len(diarize_split) > 1 else raw

    segments: list[dict] = []
    matches = list(_SEGMENT_RE.finditer(body))
    for index, match in enumerate(matches):
        speaker = int(match.group(1))
        start = float(match.group(2))
        text = _SPECIAL_TOKEN_RE.sub("", match.group(3)).strip()

        end = None
        end_match = re.match(r"\s*<\|t:([0-9]+(?:\.[0-9]+)?)\|>", body[match.end() :])
        if end_match:
            end = float(end_match.group(1))
        elif index + 1 < len(matches):
            end = float(matches[index + 1].group(2))

        if not text:
            continue
        segments.append(
            {
                "speaker": speaker,
                "start": round(start + offset, 2),
                "end": round((end if end is not None else start) + offset, 2),
                "text": text,
            }
        )

    if not segments:
        # No diarize tokens (e.g. empty/garbled generation) - fall back to plain text.
        plain = _SPECIAL_TOKEN_RE.sub("", body).strip()
        if plain:
            segments.append({"speaker": 0, "start": offset, "end": offset, "text": plain})

    return segments


def segment_label(segment: dict) -> str:
    """Display label for a segment: enrolled name when known, else 'Mówca N'."""
    name = segment.get("name")
    if name:
        return name
    return f"{SPEAKER_LABEL} {segment['speaker']}"


def render_speaker_text(segments: list[dict]) -> str:
    """Render diarized segments into a single transcript with speaker prefixes."""
    lines: list[tuple[str, str]] = []
    for segment in segments:
        text = segment["text"].strip()
        if not text:
            continue
        label = segment_label(segment)
        if lines and lines[-1][0] == label:
            lines[-1] = (label, f"{lines[-1][1]} {text}")
        else:
            lines.append((label, text))

    return "\n".join(f"{label}: {text}" for label, text in lines)

SUPPORTED_LANGUAGES = {
    "ar",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "vi",
    "zh",
}

LANGUAGE_ALIASES = {
    "arabic": "ar",
    "chinese": "zh",
    "dutch": "nl",
    "english": "en",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "polish": "pl",
    "portuguese": "pt",
    "spanish": "es",
    "vietnamese": "vi",
}


@dataclass
class TranscriptionResult:
    text: str
    language: str
    duration: float
    processing_time: float
    segments: list[dict] = field(default_factory=list)

    def asdict(self) -> dict:
        return asdict(self)


class CohereTranscriber:
    """Lazy-loading wrapper around Cohere Transcribe."""

    def __init__(
        self,
        *,
        model_name: str = "syvai/cohere-transcribe-diarize",
        default_language: str = "en",
        prefer_device: Optional[str] = None,
        vad_config: Optional[VadConfig] = None,
        speaker_registry: Optional[SpeakerRegistry] = None,
    ) -> None:
        self.model_name = model_name
        self.default_language = self.resolve_language(default_language)
        self.prefer_device = prefer_device
        self.backend = "native"
        self.model = None
        self.processor = None
        self.device = None
        self.vad_detector = SileroVoiceActivityDetector(vad_config or VadConfig.from_env())
        self.speaker_registry = speaker_registry

    def resolve_language(self, language: Optional[str]) -> str:
        """Resolve a requested language or fall back to the configured default."""
        if language is None:
            return self.default_language

        resolved = language.strip().lower()
        if resolved == "auto":
            return self.default_language

        resolved = LANGUAGE_ALIASES.get(resolved, resolved)
        if resolved not in SUPPORTED_LANGUAGES:
            LOGGER.warning(
                "Language '%s' not supported by Cohere Transcribe. Falling back to '%s'.",
                resolved,
                self.default_language,
            )
            return self.default_language

        return resolved

    def set_default_language(self, language: str) -> None:
        self.default_language = self.resolve_language(language)

    def set_model_name(self, model_name: str) -> None:
        self.model_name = model_name

    def set_vad_config(self, vad_config: VadConfig) -> None:
        self.vad_detector.update_config(vad_config)

    def _select_device(self) -> torch.device:
        if self.prefer_device == "cpu":
            LOGGER.info("Using forced CPU device")
            return torch.device("cpu")

        if self.prefer_device and self.prefer_device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA device requested but CUDA is not available")
            LOGGER.info("Using requested CUDA device: %s", self.prefer_device)
            return torch.device(self.prefer_device)

        if torch.cuda.is_available():
            LOGGER.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
            return torch.device("cuda:0")

        LOGGER.info("Using CPU device")
        return torch.device("cpu")

    def _model_dtype(self) -> torch.dtype:
        """bfloat16 on CUDA (as recommended for the diarize model), fp32 on CPU."""
        if self.prefer_device == "cpu":
            return torch.float32
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    def load_model_artifacts(self, model_name: str):
        """Load processor/model preferring local Hugging Face cache first."""
        dtype = self._model_dtype()
        processor_local = {"trust_remote_code": False, "local_files_only": True}
        model_local = {**processor_local, "dtype": dtype}
        processor_remote = {"trust_remote_code": False}
        model_remote = {**processor_remote, "dtype": dtype}

        try:
            LOGGER.info("Trying to load model artifacts from local cache first")
            processor = AutoProcessor.from_pretrained(model_name, **processor_local)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_local)
            LOGGER.info("Loaded model artifacts from local cache (dtype=%s)", dtype)
            return processor, model
        except Exception as local_error:
            LOGGER.info(
                "Local cache load failed, retrying with network access: %s",
                local_error,
            )

        processor = AutoProcessor.from_pretrained(model_name, **processor_remote)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, **model_remote)
        LOGGER.info("Loaded model artifacts with network access (dtype=%s)", dtype)
        return processor, model

    def load(self, model_name: Optional[str] = None) -> None:
        """Load the model and only swap instance state after success."""
        if model_name:
            self.model_name = model_name

        LOGGER.info("Loading model: %s (backend=%s)", self.model_name, self.backend)
        start_time = time.time()
        next_processor, next_model = self.load_model_artifacts(self.model_name)

        if torch.cuda.is_available() and self.prefer_device != "cpu":
            gpu_count = torch.cuda.device_count()
            primary_gpu_name = torch.cuda.get_device_name(0)
            LOGGER.info(
                "CUDA is available (%s visible GPU%s). Using primary device cuda:0 (%s).",
                gpu_count,
                "" if gpu_count == 1 else "s",
                primary_gpu_name,
            )
            try:
                next_device = self._select_device()
                next_model = next_model.to(next_device)
            except (RuntimeError, torch.OutOfMemoryError) as error:
                LOGGER.warning(
                    "Falling back to CPU because loading the model on %s failed: %s",
                    next_device,
                    error,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                next_device = torch.device("cpu")
                next_model = next_model.to(next_device)
        else:
            next_device = self._select_device()
            next_model = next_model.to(next_device)

        next_model.eval()

        self.processor = next_processor
        self.model = next_model
        self.device = next_device

        elapsed = time.time() - start_time
        LOGGER.info("Model loaded in %.1fs using backend=%s", elapsed, self.backend)

    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def health_payload(self) -> dict:
        return {
            "status": "ok" if self.is_loaded() else "loading",
            "ready": self.is_loaded(),
            "model": self.model_name or None,
            "device": str(self.device) if self.device is not None else None,
            "backend": self.backend,
            "vad": self.vad_detector.status_payload(),
            "speaker_id": (
                self.speaker_registry.status_payload()
                if self.speaker_registry is not None
                else {"enabled": False}
            ),
        }

    def transcribe_pcm(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Transcribe normalized PCM audio."""
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        resolved_language = self.resolve_language(language)
        duration_s = len(audio_data) / sample_rate

        if is_effectively_silent(audio_data):
            LOGGER.info("No speech detected above silence threshold; returning empty transcription")
            return TranscriptionResult(
                text="",
                language=resolved_language,
                duration=round(duration_s, 2),
                processing_time=0.0,
            )

        vad_decision = self.vad_detector.detect_speech(audio_data, sample_rate=sample_rate)
        if not vad_decision.has_speech:
            LOGGER.info(
                "Silero VAD rejected audio as non-speech (reason=%s, segments=%s, total_ms=%s, max_ms=%s, speech_rms=%.6f, noise_rms=%.6f, snr=%.3f)",
                vad_decision.reason,
                vad_decision.speech_segments,
                vad_decision.total_speech_ms,
                vad_decision.max_segment_ms,
                vad_decision.speech_rms,
                vad_decision.noise_rms,
                vad_decision.speech_to_noise_ratio,
            )
            return TranscriptionResult(
                text="",
                language=resolved_language,
                duration=round(duration_s, 2),
                processing_time=0.0,
            )

        start_time = time.time()

        segments: list[dict] = []
        for chunk, offset in chunk_audio(audio_data, sample_rate):
            raw = self._generate_diarized(chunk, sample_rate, resolved_language, temperature)
            segments.extend(parse_diarized_output(raw, offset=offset))

        self._identify_speakers(segments, audio_data, sample_rate)
        text = render_speaker_text(segments)

        elapsed = time.time() - start_time
        rtfx = duration_s / elapsed if elapsed > 0 else 0
        speaker_count = len({segment["speaker"] for segment in segments})
        LOGGER.info(
            "Transcribed %.1fs audio in %.1fs (RTFx: %.1fx) lang=%s speakers=%d backend=%s",
            duration_s,
            elapsed,
            rtfx,
            resolved_language,
            speaker_count,
            self.backend,
        )

        return TranscriptionResult(
            text=text,
            language=resolved_language,
            duration=round(duration_s, 2),
            processing_time=round(elapsed, 2),
            segments=segments,
        )

    def _build_prompt_ids(self, language: str) -> torch.Tensor:
        """Build the diarize decoder prompt token ids for the given language."""
        tokenizer = self.processor.tokenizer
        unk_id = tokenizer.unk_token_id
        ids: list[int] = []
        for template in DIARIZE_PROMPT_TEMPLATE:
            token = template.format(lang=language)
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id is None or token_id == unk_id:
                LOGGER.warning(
                    "Diarize prompt token '%s' missing from tokenizer; prompt may be malformed",
                    token,
                )
            ids.append(token_id)
        return torch.tensor([ids], device=self.model.device)

    def _generate_diarized(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: str,
        temperature: float,
    ) -> str:
        """Run a single diarize generation pass and return the raw token stream."""
        inputs = self.processor(
            audio_data,
            sampling_rate=sample_rate,
            return_tensors="pt",
            language=language,
        )

        model_inputs: dict = {}
        for key, value in inputs.items():
            if not isinstance(value, torch.Tensor):
                continue  # e.g. audio_chunk_index is a plain list
            if value.is_floating_point():
                model_inputs[key] = value.to(self.model.device, dtype=self.model.dtype)
            else:
                model_inputs[key] = value.to(self.model.device)

        # Override the processor's decoder prompt to force diarize + timestamps.
        model_inputs["decoder_input_ids"] = self._build_prompt_ids(language)
        if "attention_mask" not in model_inputs:
            features = model_inputs["input_features"]
            model_inputs["attention_mask"] = torch.ones(
                features.shape[:2], device=self.model.device
            )

        generate_kwargs = {
            "max_new_tokens": 400,
            "do_sample": False,
            "repetition_penalty": 1.2,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **generate_kwargs)

        return self.processor.tokenizer.decode(outputs[0], skip_special_tokens=False)

    def _identify_speakers(
        self, segments: list[dict], audio_data: np.ndarray, sample_rate: int
    ) -> None:
        """Annotate each segment with an enrolled speaker name when one matches."""
        registry = self.speaker_registry
        if registry is None or not registry.enabled or not segments:
            return

        try:
            registry.reload_if_changed()
        except Exception as error:
            LOGGER.warning("Speaker enrollment reload failed: %s", error)
            return

        if not registry.has_profiles():
            return

        total_samples = len(audio_data)
        for segment in segments:
            start = max(0, int(segment["start"] * sample_rate))
            end = min(total_samples, int(segment["end"] * sample_rate))
            if end <= start:
                continue
            try:
                match = registry.identify(audio_data[start:end])
            except Exception as error:
                LOGGER.warning("Speaker identification failed: %s", error)
                return
            if match.name:
                segment["name"] = match.name
                segment["score"] = match.score

    def transcribe_file_bytes(
        self,
        file_bytes: bytes,
        *,
        filename: str = "audio",
        language: Optional[str] = None,
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        audio_data, sample_rate = read_audio_to_numpy(file_bytes, filename)
        return self.transcribe_pcm(
            audio_data,
            sample_rate=sample_rate,
            language=language,
            temperature=temperature,
        )
