"""Cohere transcription backend independent from HTTP/Wyoming transport."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from .audio import is_effectively_silent, read_audio_to_numpy
from .enrollment import read_role
from .history import RecognitionLog
from .pending import PendingStore
from .settings import DEFAULT_SPEAKER_TEXT_MODE, SettingsStore
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

# Padding kept around the VAD speech span when cropping non-speech audio before
# generation. Leading/trailing noise makes the diarize model hallucinate
# (observed on Polish far-field clips: 1.2 s of speech in a 8.9 s clip produced
# looping gibberish; the cropped clip transcribed correctly). Kept small: the
# VAD span already includes VAD_SPEECH_PAD_MS, and every extra bit of trailing
# noise invites end-of-clip babble.
CROP_PAD_SECONDS = 0.1

# Generation budget per window. When a window's output hits this cap the window
# is re-transcribed as two shorter ones (down to MIN_SPLIT_SECONDS) so dense
# speech is not silently truncated.
MAX_NEW_TOKENS = 400
MIN_SPLIT_SECONDS = 8

# Minimum ECAPA cosine similarity to treat a speaker from a later window as the
# same person as one heard in an earlier window (see _merge_diarized_windows).
# Override with the SPEAKER_CHAIN_THRESHOLD environment variable.
DEFAULT_CHAIN_THRESHOLD = 0.40

# Prefix used when rendering anonymous diarized speakers into a single transcript.
SPEAKER_LABEL = "Mówca"

_SEGMENT_RE = re.compile(
    r"<\|spltoken(\d+)\|>\s*<\|t:([0-9]+(?:\.[0-9]+)?)\|>(.*?)"
    r"(?=<\|t:[0-9]|<\|spltoken\d|<\|endoftext\|>|$)",
    re.DOTALL,
)
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]*\|>")


# When long audio must be windowed, the cut is placed on the quietest moment
# within the last SPLIT_SEARCH_SECONDS of the window instead of a hard cut at
# exactly max_seconds, so words are not sliced mid-syllable.
SPLIT_SEARCH_SECONDS = 10
_SPLIT_FRAME_MS = 30


def _quietest_split(
    audio_data: np.ndarray, search_start: int, search_end: int, sample_rate: int
) -> int:
    """Sample index of the quietest moment in [search_start, search_end)."""
    frame = max(1, int(sample_rate * _SPLIT_FRAME_MS / 1000))
    segment = np.asarray(audio_data[search_start:search_end], dtype=np.float32)
    frame_count = len(segment) // frame
    if frame_count < 2:
        return search_end
    frames = segment[: frame_count * frame].reshape(frame_count, frame)
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    # Prefer the LAST near-quietest frame: on uniform audio this keeps full
    # windows instead of always cutting at the search-region start.
    threshold = float(rms.min()) * 1.25 + 1e-9
    index = int(np.nonzero(rms <= threshold)[0][-1])
    split = search_start + index * frame + frame // 2
    if search_end - split <= frame * 2:
        return search_end
    return split


def chunk_audio(audio_data: np.ndarray, sample_rate: int, max_seconds: int = MAX_CHUNK_SECONDS):
    """Split audio into windows no longer than max_seconds, with second offsets.

    Split points land on the quietest moment near each window's end (see
    _quietest_split) so speech is not cut mid-word at the window boundary.
    """
    max_samples = int(max_seconds * sample_rate)
    if max_samples <= 0 or len(audio_data) <= max_samples:
        return [(audio_data, 0.0)]

    search_samples = min(int(SPLIT_SEARCH_SECONDS * sample_rate), max_samples // 2)
    chunks = []
    start = 0
    while len(audio_data) - start > max_samples:
        window_end = start + max_samples
        split = _quietest_split(
            audio_data, window_end - search_samples, window_end, sample_rate
        )
        chunks.append((audio_data[start:split], start / sample_rate))
        start = split
    chunks.append((audio_data[start:], start / sample_rate))
    return chunks


def parse_diarized_output(raw: str, offset: float = 0.0, duration: Optional[float] = None) -> list[dict]:
    """Parse the diarize token stream into ordered speaker segments.

    Format (confirmed via spike): ``<|spltokenN|><|t:START|> text <|t:END|>`` repeated,
    terminated by ``<|endoftext|>``. Timestamps are offset by ``offset`` seconds so
    segments from later chunks land on the global timeline.

    The model often omits the closing timestamp of the last segment; ``duration``
    (the window length in seconds) is used as that segment's end so it never
    collapses to a zero-length span (which would break speaker identification,
    pending-voice clips and subtitle cues).
    """
    diarize_split = raw.split("<|diarize|>", 1)
    body = diarize_split[1] if len(diarize_split) > 1 else raw

    segments: list[dict] = []
    matches = list(_SEGMENT_RE.finditer(body))

    # The model sometimes emits transcript text BEFORE the first
    # <|spltokenN|><|t:...|> header (observed on Polish multi-window audio);
    # dropping it would lose whole sentences. Attribute it to the first
    # headed speaker, spanning from the window start to their first timestamp.
    if matches:
        leading = _SPECIAL_TOKEN_RE.sub("", body[: matches[0].start()]).strip()
        if leading:
            segments.append(
                {
                    "speaker": int(matches[0].group(1)),
                    "start": round(offset, 2),
                    "end": round(float(matches[0].group(2)) + offset, 2),
                    "text": leading,
                }
            )

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
        elif duration is not None:
            end = max(start, duration)

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
            segments.append(
                {
                    "speaker": 0,
                    "start": offset,
                    "end": round(offset + (duration or 0.0), 2),
                    "text": plain,
                }
            )

    return segments


def segment_label(segment: dict) -> str:
    """Display label for a segment: enrolled name when known, else 'Mówca N'."""
    name = segment.get("name")
    if name:
        return name
    return f"{SPEAKER_LABEL} {segment['speaker']}"


def render_speaker_text(segments: list[dict], mode: str = DEFAULT_SPEAKER_TEXT_MODE) -> str:
    """Render diarized segments into a single transcript.

    Modes "prefix" and "both" prefix each turn with the speaker label; mode
    "field" keeps the turn structure but omits labels (identity is delivered
    via the Transcript event's "speaker" field instead).
    """
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

    if mode == "field":
        return "\n".join(text for _, text in lines)
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
    # Enrolled name of the dominant speaker (most speech time), if recognized.
    speaker: Optional[str] = None
    speaker_score: Optional[float] = None
    # Role of the recognized dominant speaker (admin/user/guest).
    speaker_role: Optional[str] = None
    # Pending-clip id saved for an unrecognized dominant speaker (see pending.py).
    utterance_id: Optional[str] = None
    # Speaker text mode the transcript text was rendered with (see settings.py).
    text_mode: str = DEFAULT_SPEAKER_TEXT_MODE

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
        settings_store: Optional[SettingsStore] = None,
        pending_store: Optional[PendingStore] = None,
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
        self.settings_store = settings_store or SettingsStore.from_env()
        # Unrecognized-voice buffer; only useful with speaker ID active.
        if pending_store is not None:
            self.pending_store = pending_store
        elif speaker_registry is not None and speaker_registry.enabled:
            self.pending_store = PendingStore.from_env(speaker_registry.enrollment_dir)
        else:
            self.pending_store = None
        self.recognition_log = RecognitionLog.from_env(
            speaker_registry.enrollment_dir if speaker_registry is not None else None
        )
        self.speaker_chain_threshold = float(
            os.environ.get("SPEAKER_CHAIN_THRESHOLD", DEFAULT_CHAIN_THRESHOLD)
        )
        self._prompt_id_cache: dict[str, torch.Tensor] = {}
        # Serializes inference (and model swaps in load()) so async transports
        # can safely offload transcribe_pcm to worker threads.
        self._inference_lock = threading.Lock()
        # Consecutive transcription failures, maintained by the Wyoming handler
        # to make persistent breakage visible despite empty-transcript replies.
        self.failure_streak = 0

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
            next_device = self._select_device()
            try:
                next_model = next_model.to(next_device)
            except (RuntimeError, torch.OutOfMemoryError) as error:
                LOGGER.warning(
                    "Falling back to CPU (float32) because loading the model on %s failed: %s",
                    next_device,
                    error,
                )
                torch.cuda.empty_cache()
                next_device = torch.device("cpu")
                # The dtype was chosen for CUDA (bfloat16); on CPU bf16 is slow
                # and some ops are unsupported, so the fallback runs in float32.
                next_model = next_model.to(next_device, dtype=torch.float32)
        else:
            next_device = self._select_device()
            next_model = next_model.to(next_device)

        next_model.eval()

        # Swap under the inference lock so an in-flight transcription never
        # sees a half-updated processor/model/device triple.
        with self._inference_lock:
            self.processor = next_processor
            self.model = next_model
            self.device = next_device
            self._prompt_id_cache.clear()
            # Fail fast here rather than crashing the first request mid-transcription.
            self._validate_structural_prompt_tokens()

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

        # Async transports call this via worker threads; serialize all torch
        # work (VAD, generate, ECAPA) and block model swaps mid-transcription.
        with self._inference_lock:
            return self._transcribe_pcm_locked(
                audio_data,
                sample_rate=sample_rate,
                resolved_language=resolved_language,
                duration_s=duration_s,
                temperature=temperature,
            )

    def _transcribe_pcm_locked(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int,
        resolved_language: str,
        duration_s: float,
        temperature: float,
    ) -> TranscriptionResult:
        text_mode = self.settings_store.load().speaker_text_mode

        if is_effectively_silent(audio_data):
            LOGGER.info("No speech detected above silence threshold; returning empty transcription")
            return TranscriptionResult(
                text="",
                language=resolved_language,
                duration=round(duration_s, 2),
                processing_time=0.0,
                text_mode=text_mode,
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
                text_mode=text_mode,
            )

        start_time = time.time()

        # Only the speech span (plus padding) is sent to generation; segment
        # timestamps are mapped back to the full-clip timeline via crop_offset,
        # so speaker clips/pending audio keep using the original audio.
        crop_start, crop_end = self._speech_bounds(vad_decision, len(audio_data), sample_rate)
        trimmed_s = (len(audio_data) - (crop_end - crop_start)) / sample_rate
        if trimmed_s >= 1.0:
            LOGGER.info(
                "Trimmed %.1fs of non-speech padding before transcription "
                "(speech span %.2f-%.2fs of %.2fs)",
                trimmed_s,
                crop_start / sample_rate,
                crop_end / sample_rate,
                duration_s,
            )
        crop_offset = crop_start / sample_rate

        windows: list[list[dict]] = []
        for chunk, offset in chunk_audio(audio_data[crop_start:crop_end], sample_rate):
            windows.extend(
                self._transcribe_window(
                    chunk, crop_offset + offset, sample_rate, resolved_language, temperature
                )
            )
        segments, merge_embeddings = self._merge_diarized_windows(
            windows, audio_data, sample_rate
        )

        speaker_embeddings = self._identify_speakers(
            segments, audio_data, sample_rate, merge_embeddings
        )
        dominant_name, dominant_score = self._dominant_speaker(segments)
        text = render_speaker_text(segments, mode=text_mode)

        speaker_role = None
        utterance_id = None
        if dominant_name and self.speaker_registry is not None:
            speaker_role = read_role(self.speaker_registry.enrollment_dir, dominant_name)
        elif text.strip():
            utterance_id = self._save_pending_utterance(
                segments, audio_data, sample_rate, text, speaker_embeddings
            )

        if self.recognition_log is not None and text.strip():
            self.recognition_log.append(
                text=text,
                language=resolved_language,
                duration=round(duration_s, 2),
                speaker=dominant_name,
                score=dominant_score,
                role=speaker_role,
                utterance_id=utterance_id,
            )

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
            speaker=dominant_name,
            speaker_score=dominant_score,
            speaker_role=speaker_role,
            utterance_id=utterance_id,
            text_mode=text_mode,
        )

    @staticmethod
    def _speech_bounds(vad_decision, total_samples: int, sample_rate: int) -> tuple[int, int]:
        """Sample range to transcribe: the VAD speech span padded by CROP_PAD_SECONDS.

        Falls back to the full clip when the decision carries no span (VAD
        disabled, fallback mode, or an inconsistent range).
        """
        start = getattr(vad_decision, "speech_start_sample", None)
        end = getattr(vad_decision, "speech_end_sample", None)
        if start is None or end is None or not 0 <= start < end <= total_samples:
            return 0, total_samples
        pad = int(CROP_PAD_SECONDS * sample_rate)
        return max(0, start - pad), min(total_samples, end + pad)

    def _resolve_prompt_token(self, token: str) -> Optional[int]:
        """Return the token id, or None if the tokenizer lacks it (unk)."""
        tokenizer = self.processor.tokenizer
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == tokenizer.unk_token_id:
            return None
        return token_id

    def _validate_structural_prompt_tokens(self) -> None:
        """Fail fast at load time if the model lacks the diarize structural tokens."""
        missing = [
            template
            for template in DIARIZE_PROMPT_TEMPLATE
            if "{lang}" not in template and self._resolve_prompt_token(template) is None
        ]
        if missing:
            raise RuntimeError(
                f"Model {self.model_name} is not diarize-capable: tokenizer is missing "
                f"prompt tokens {missing}."
            )

    def _language_token_id(self, language: str) -> int:
        """Resolve the <|lang|> token, falling back to the default language then English."""
        seen: list[str] = []
        for candidate in (language, self.default_language, "en"):
            if candidate in seen:
                continue
            seen.append(candidate)
            token_id = self._resolve_prompt_token(f"<|{candidate}|>")
            if token_id is not None:
                if candidate != language:
                    LOGGER.warning(
                        "Language token '<|%s|>' missing; falling back to '<|%s|>'",
                        language,
                        candidate,
                    )
                return token_id
        raise RuntimeError(
            f"No usable language prompt token for '{language}' in model {self.model_name}"
        )

    def _build_prompt_ids(self, language: str) -> torch.Tensor:
        """Build (and cache per language) the diarize decoder prompt token ids.

        Structural tokens are validated once at load(); the language token falls
        back to the default language (then English) if absent.
        """
        cached = self._prompt_id_cache.get(language)
        if cached is not None:
            return cached

        ids: list[int] = []
        for template in DIARIZE_PROMPT_TEMPLATE:
            if "{lang}" in template:
                ids.append(self._language_token_id(language))
            else:
                token_id = self._resolve_prompt_token(template)
                if token_id is None:
                    raise RuntimeError(
                        f"Diarize prompt token '{template}' missing from tokenizer for "
                        f"model {self.model_name}"
                    )
                ids.append(token_id)

        prompt = torch.tensor([ids], device=self.model.device)
        self._prompt_id_cache[language] = prompt
        return prompt

    def _generate_diarized(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        language: str,
        temperature: float,
    ) -> tuple[str, bool]:
        """Run a single diarize generation pass.

        Returns the raw token stream and whether generation was cut off by the
        MAX_NEW_TOKENS cap (i.e. the window's tail may be missing).
        """
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
            # Fallback only; this model's processor always returns attention_mask.
            # input_features is (batch, frames, features) for cohere_asr, so
            # shape[:2] = (batch, frames) is the correct mask shape (matches the
            # model card's torch.ones(input_features.shape[:2])).
            features = model_inputs["input_features"]
            model_inputs["attention_mask"] = torch.ones(
                features.shape[:2], device=self.model.device
            )

        generate_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "repetition_penalty": 1.2,
        }
        if temperature > 0:
            generate_kwargs["do_sample"] = True
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **generate_kwargs)

        generated_tokens = outputs[0].shape[-1] - model_inputs["decoder_input_ids"].shape[-1]
        end_token_id = self._resolve_prompt_token("<|endoftext|>")
        if end_token_id is None:
            end_token_id = self.processor.tokenizer.eos_token_id
        truncated = generated_tokens >= MAX_NEW_TOKENS and (
            end_token_id is None or int(outputs[0][-1]) != end_token_id
        )

        return self.processor.tokenizer.decode(outputs[0], skip_special_tokens=False), truncated

    def _transcribe_window(
        self,
        chunk: np.ndarray,
        offset: float,
        sample_rate: int,
        language: str,
        temperature: float,
    ) -> list[list[dict]]:
        """Transcribe one window, splitting it in half when generation is truncated.

        Returns one parsed-segment list per generation pass; each pass has its
        own window-local speaker indices, so callers must merge them via
        _merge_diarized_windows.
        """
        raw, truncated = self._generate_diarized(chunk, sample_rate, language, temperature)
        duration = len(chunk) / sample_rate

        if truncated and duration >= 2 * MIN_SPLIT_SECONDS:
            LOGGER.warning(
                "Diarize generation hit the %d-token cap on a %.1fs window; "
                "retrying as two shorter windows",
                MAX_NEW_TOKENS,
                duration,
            )
            half = len(chunk) // 2
            return self._transcribe_window(
                chunk[:half], offset, sample_rate, language, temperature
            ) + self._transcribe_window(
                chunk[half:], offset + half / sample_rate, sample_rate, language, temperature
            )

        if truncated:
            LOGGER.warning(
                "Diarize generation hit the %d-token cap on a %.1fs window; "
                "the end of this window may be missing from the transcript",
                MAX_NEW_TOKENS,
                duration,
            )

        return [parse_diarized_output(raw, offset=offset, duration=duration)]

    def _merge_diarized_windows(
        self,
        windows: list[list[dict]],
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> tuple[list[dict], dict[int, np.ndarray]]:
        """Merge per-window diarized segments onto one global speaker space.

        Diarize speaker indices restart at 0 in every generation window, so the
        same index in two windows usually names two different people. Each
        window-local speaker is matched to speakers from earlier windows by
        ECAPA voiceprint similarity; without a confident match it gets a fresh
        global index — over-splitting is recoverable by enrollment naming,
        silently merging two people is not.

        Without the embedding backend (speaker ID disabled, or embedding
        failure before any window was merged) the model's own indices are kept
        as-is across windows: a single speaker then stays one "Mówca 0" instead
        of being shredded into per-window speakers, at the cost of the
        (pre-existing) risk that two different people sharing index 0 in
        different windows get merged. When embeddings fail mid-merge, later
        windows get fresh indices instead — their raw indices would collide
        with already-assigned global ones and silently merge two people.

        Returns the merged segments plus one L2-normalized ECAPA voiceprint per
        global speaker (empty when embeddings were unavailable), so speaker
        identification does not have to re-embed the same audio.
        """
        windows = [window for window in windows if window]
        if not windows:
            return [], {}
        if len(windows) == 1:
            return windows[0], {}

        registry = self.speaker_registry
        use_embeddings = registry is not None and registry.enabled

        merged: list[dict] = []
        # global speaker index -> (sum of L2-normalized embeddings, count)
        profiles: dict[int, tuple[np.ndarray, int]] = {}
        next_index = 0

        for window in windows:
            local_speakers = sorted({segment["speaker"] for segment in window})

            embeddings: dict[int, np.ndarray] = {}
            if use_embeddings:
                clips = [
                    self._speaker_clip(window, local, audio_data, sample_rate)
                    for local in local_speakers
                ]
                try:
                    for local, embedding in zip(local_speakers, registry.embed_batch(clips)):
                        if embedding is not None:
                            embeddings[local] = embedding
                except Exception as error:
                    LOGGER.warning(
                        "Cross-window speaker embedding failed; speakers will not be "
                        "merged across windows: %s",
                        error,
                    )
                    use_embeddings = False
                    embeddings = {}

            if not use_embeddings:
                if profiles:
                    # Mid-merge failure: earlier windows already occupy global
                    # indices, so this window's raw indices could silently
                    # merge two different people. Assign fresh indices instead.
                    fallback_mapping = {}
                    for local in local_speakers:
                        fallback_mapping[local] = next_index
                        next_index += 1
                    for segment in window:
                        segment["speaker"] = fallback_mapping[segment["speaker"]]
                # Otherwise no voiceprints were ever compared: preserve the
                # model's raw indices (see docstring).
                merged.extend(window)
                continue

            # Best-score-first unique assignment of window speakers to known ones.
            candidates: list[tuple[float, int, int]] = []
            for local, embedding in embeddings.items():
                for global_index, (vector_sum, count) in profiles.items():
                    profile = vector_sum / count
                    norm = float(np.linalg.norm(profile))
                    if norm == 0.0:
                        continue
                    score = float(np.dot(embedding, profile / norm))
                    if score >= self.speaker_chain_threshold:
                        candidates.append((score, local, global_index))
            candidates.sort(key=lambda item: item[0], reverse=True)

            mapping: dict[int, int] = {}
            used_globals: set[int] = set()
            for score, local, global_index in candidates:
                if local in mapping or global_index in used_globals:
                    continue
                mapping[local] = global_index
                used_globals.add(global_index)
            for local in local_speakers:
                if local not in mapping:
                    mapping[local] = next_index
                    next_index += 1

            for local, embedding in embeddings.items():
                global_index = mapping[local]
                if global_index in profiles:
                    vector_sum, count = profiles[global_index]
                    profiles[global_index] = (vector_sum + embedding, count + 1)
                else:
                    profiles[global_index] = (embedding.copy(), 1)

            for segment in window:
                segment["speaker"] = mapping[segment["speaker"]]
            merged.extend(window)

        speaker_embeddings: dict[int, np.ndarray] = {}
        for global_index, (vector_sum, count) in profiles.items():
            mean = vector_sum / count
            norm = float(np.linalg.norm(mean))
            if norm > 0.0:
                speaker_embeddings[global_index] = (mean / norm).astype(np.float32)
        return merged, speaker_embeddings

    @staticmethod
    def _speaker_clip(
        segments: list[dict],
        speaker_id: Optional[int],
        audio_data: np.ndarray,
        sample_rate: int,
    ) -> Optional[np.ndarray]:
        """Concatenate one speaker's segments into a single clip (None if empty)."""
        total_samples = len(audio_data)
        parts = []
        for segment in segments:
            if segment["speaker"] != speaker_id:
                continue
            start = max(0, int(segment["start"] * sample_rate))
            end = min(total_samples, int(segment["end"] * sample_rate))
            if end > start:
                parts.append(audio_data[start:end])
        return np.concatenate(parts) if parts else None

    def _identify_speakers(
        self,
        segments: list[dict],
        audio_data: np.ndarray,
        sample_rate: int,
        embeddings: Optional[dict[int, np.ndarray]] = None,
    ) -> dict[int, np.ndarray]:
        """Annotate segments with enrolled names, matching once per diarized speaker.

        Voiceprints already computed by _merge_diarized_windows are reused via
        ``embeddings``; only speakers without one get embedded here. All of a
        speaker's segments are concatenated into one clip before embedding:
        voice commands are often split into sub-0.4 s segments too short for a
        reliable voiceprint on their own, while the concatenation matches well.
        The name is then applied to all of that speaker's segments.

        Returns the per-speaker embeddings (input ones plus any computed here)
        so the pending-clip save can reuse them too.
        """
        embeddings = dict(embeddings) if embeddings else {}
        registry = self.speaker_registry
        if registry is None or not registry.enabled or not segments:
            return embeddings

        try:
            registry.reload_if_changed()
        except Exception as error:
            LOGGER.warning("Speaker enrollment reload failed: %s", error)
            return embeddings

        if not registry.has_profiles():
            return embeddings

        speaker_ids = sorted({segment["speaker"] for segment in segments})
        missing = [sid for sid in speaker_ids if sid not in embeddings]
        if missing:
            clips = [
                self._speaker_clip(segments, sid, audio_data, sample_rate)
                for sid in missing
            ]
            try:
                for sid, embedding in zip(missing, registry.embed_batch(clips)):
                    if embedding is not None:
                        embeddings[sid] = embedding
            except Exception as error:
                LOGGER.warning("Speaker identification failed: %s", error)
                return embeddings

        matches = {
            sid: registry.match_embedding(embeddings.get(sid)) for sid in speaker_ids
        }
        for segment in segments:
            match = matches.get(segment["speaker"])
            if match is not None and match.name:
                segment["name"] = match.name
                segment["score"] = match.score

        # Confident recognitions feed the person's adaptive voiceprint, so
        # profiles track real usage conditions (mics, rooms, voice drift).
        for sid, match in matches.items():
            embedding = embeddings.get(sid)
            if match.name and embedding is not None:
                registry.adapt(match.name, embedding, match.score)

        return embeddings

    @staticmethod
    def _dominant_speaker_index(segments: list[dict]) -> Optional[int]:
        """Diarized speaker index with the most speech time."""
        if not segments:
            return None
        durations: dict[int, float] = {}
        for segment in segments:
            length = max(0.0, segment["end"] - segment["start"])
            durations[segment["speaker"]] = durations.get(segment["speaker"], 0.0) + length
        return max(durations, key=durations.get)

    @classmethod
    def _dominant_speaker(cls, segments: list[dict]) -> tuple[Optional[str], Optional[float]]:
        """Enrolled name of the speaker with the most speech time, if recognized."""
        dominant = cls._dominant_speaker_index(segments)
        if dominant is None:
            return None, None
        for segment in segments:
            if segment["speaker"] == dominant and segment.get("name"):
                return segment["name"], segment.get("score")
        return None, None

    def _save_pending_utterance(
        self,
        segments: list[dict],
        audio_data: np.ndarray,
        sample_rate: int,
        text: str,
        embeddings: Optional[dict[int, np.ndarray]] = None,
    ) -> Optional[str]:
        """Buffer the unrecognized dominant speaker's audio for later enrollment.

        The clip (that speaker's concatenated segments) lands in the pending
        store with its transcript and ECAPA embedding (reused from speaker
        identification when available); the returned utterance id is exposed in
        the Transcript event so an LLM pipeline can ask "who is speaking?" and
        claim the clip for a person. Never raises.
        """
        store = self.pending_store
        registry = self.speaker_registry
        if store is None or registry is None or not registry.enabled or not segments:
            return None

        try:
            dominant = self._dominant_speaker_index(segments)
            clip = self._speaker_clip(segments, dominant, audio_data, sample_rate)
            if clip is None:
                return None

            embedding = (embeddings or {}).get(dominant)
            if embedding is None:
                try:
                    embedding = registry.embed(clip)
                except Exception as error:
                    LOGGER.warning("Pending-clip embedding failed (saving without): %s", error)

            # Closest enrolled profile (below threshold, or there would be no
            # pending clip) — recorded for threshold tuning in the UI/history.
            best_match = None
            best_score = None
            if embedding is not None:
                try:
                    near = registry.nearest(embedding)
                    if near.name is not None:
                        best_match, best_score = near.name, near.score
                except Exception as error:
                    LOGGER.debug("Nearest-profile lookup for pending clip failed: %s", error)

            return store.save(
                clip,
                sample_rate,
                text=text,
                embedding=embedding,
                best_match=best_match,
                best_score=best_score,
            )
        except Exception as error:
            LOGGER.warning("Could not save pending utterance: %s", error)
            return None

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
