"""whisper.cpp-backed transcription pipeline independent from HTTP/Wyoming transport."""

from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
import wave
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .audio import is_effectively_silent, read_audio_to_numpy
from .enrollment import read_role
from .history import RecognitionLog
from .pending import PendingStore
from .settings import DEFAULT_SPEAKER_TEXT_MODE, SettingsStore
from .speaker_id import SpeakerRegistry
from .vad import SileroVoiceActivityDetector, VadConfig


LOGGER = logging.getLogger("transcribe-wyoming.transcriber")

# How long a whisper.cpp reachability probe result is trusted before the
# server is probed again (see whispercpp_reachable).
PROBE_CACHE_SECONDS = 10.0

# Padding kept around the VAD speech span when cropping non-speech audio before
# generation. Leading/trailing noise makes the diarize model hallucinate
# (observed on Polish far-field clips: 1.2 s of speech in a 8.9 s clip produced
# looping gibberish; the cropped clip transcribed correctly). Kept small: the
# VAD span already includes VAD_SPEECH_PAD_MS, and every extra bit of trailing
# noise invites end-of-clip babble.
CROP_PAD_SECONDS = 0.1

# Prefix used when rendering anonymous diarized speakers into a single
# transcript. Localizable: this literal ends up in the transcript text that
# downstream consumers (e.g. an LLM) see, so deployments set it to match the
# spoken language (SPEAKER_LABEL=Mówca for Polish).
SPEAKER_LABEL = os.environ.get("SPEAKER_LABEL", "Speaker")

# Optional whisper initial prompt biasing decoding toward household vocabulary
# (assistant name, people, rooms, devices). Nouns/proper names only: example
# commands in the prompt get echoed back as hallucinations on noisy audio
# ("Która godzina?" transcribed from pure noise in testing).
WHISPER_INITIAL_PROMPT = os.environ.get("WHISPER_INITIAL_PROMPT", "").strip()

def segment_label(segment: dict) -> str:
    """Display label for a segment: enrolled name when known, else '<SPEAKER_LABEL> N'."""
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


class SpeechTranscriber:
    """Transcription pipeline around a whisper.cpp server.

    The server does the speech-to-text (over HTTP); this class owns the
    surrounding pipeline: Silero VAD cropping, ECAPA speaker identification,
    pending-voice enrollment and the recognition history. The whole utterance
    is treated as a single speaker (no diarization).
    """

    def __init__(
        self,
        *,
        default_language: str = "en",
        vad_config: Optional[VadConfig] = None,
        speaker_registry: Optional[SpeakerRegistry] = None,
        settings_store: Optional[SettingsStore] = None,
        pending_store: Optional[PendingStore] = None,
        whispercpp_url: Optional[str] = None,
    ) -> None:
        self.default_language = self.resolve_language(default_language)
        self.whispercpp_url = (
            whispercpp_url or os.environ.get("WHISPERCPP_URL", "http://whispercpp:4050")
        ).rstrip("/")
        self.backend = "whispercpp"
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
        # Serializes inference (and model swaps in load()) so async transports
        # can safely offload transcribe_pcm to worker threads.
        self._inference_lock = threading.Lock()
        # whisper.cpp reachability probe cache (see whispercpp_reachable).
        self._probe_ok = False
        self._probe_checked_at = float("-inf")
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
                "Language '%s' not in the supported-language list. Falling back to '%s'.",
                resolved,
                self.default_language,
            )
            return self.default_language

        return resolved

    def set_default_language(self, language: str) -> None:
        self.default_language = self.resolve_language(language)

    def set_vad_config(self, vad_config: VadConfig) -> None:
        self.vad_detector.update_config(vad_config)

    def load(self, model_name: Optional[str] = None) -> None:
        """Probe the whisper.cpp server (transcription runs server-side).

        A dead/unreachable server fails loudly at startup instead of as
        per-request empty transcripts.
        """
        LOGGER.info("Using whisper.cpp server at %s", self.whispercpp_url)
        if self.whispercpp_reachable(force=True):
            LOGGER.info("whisper.cpp server is reachable")
        else:
            LOGGER.warning(
                "whisper.cpp server not reachable yet; requests will retry"
            )

    def whispercpp_reachable(self, *, force: bool = False) -> bool:
        """Cached reachability probe of the whisper.cpp server.

        Cheap enough for /health and the Home Assistant sensor poll, cached
        (PROBE_CACHE_SECONDS) so request paths never stack probes.
        """
        now = time.monotonic()
        if not force and now - self._probe_checked_at < PROBE_CACHE_SECONDS:
            return self._probe_ok
        try:
            import requests

            requests.get(self.whispercpp_url + "/", timeout=3)
            self._probe_ok = True
        except Exception as error:
            if self._probe_ok:
                LOGGER.warning("whisper.cpp server unreachable: %s", error)
            self._probe_ok = False
        self._probe_checked_at = now
        return self._probe_ok

    def is_loaded(self) -> bool:
        """Ready to transcribe = the whisper.cpp server answers."""
        return self.whispercpp_reachable()

    def _whispercpp_transcribe(
        self, audio_data: np.ndarray, sample_rate: int, language: str,
        temperature: float = 0.0,
    ) -> str:
        """Transcribe one clip via the whisper.cpp server /inference endpoint."""
        import requests

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(
                np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16).tobytes()
            )
        buffer.seek(0)

        data = {
            "language": language,
            "response_format": "json",
            "temperature": str(temperature),
            # Beam search noticeably improves short degraded commands
            # ("Agata." vs "Pagata." on real clips); the server-side
            # --beam-size flag crashes at startup, per-request works.
            "beam_size": "5",
            # Suppress non-speech tokens ("(music)", "♪"): far-field noise
            # otherwise occasionally transcribes as sound-effect markup.
            "suppress_nst": "true",
        }
        if WHISPER_INITIAL_PROMPT:
            data["prompt"] = WHISPER_INITIAL_PROMPT
        response = requests.post(
            self.whispercpp_url + "/inference",
            files={"file": ("audio.wav", buffer, "audio/wav")},
            data=data,
            timeout=60,
        )
        response.raise_for_status()
        return str(response.json().get("text", "")).strip()

    def health_payload(self) -> dict:
        return {
            "status": "ok" if self.is_loaded() else "loading",
            "ready": self.is_loaded(),
            "model": "whisper.cpp",
            "whispercpp_url": self.whispercpp_url,
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
            raise RuntimeError("whisper.cpp server unreachable")

        resolved_language = self.resolve_language(language)
        duration_s = len(audio_data) / sample_rate

        # Async transports call this via worker threads; serialize all torch
        # work (VAD, generate, ECAPA) and block model swaps mid-transcription.
        lock_wait_start = time.perf_counter()
        with self._inference_lock:
            return self._transcribe_pcm_locked(
                audio_data,
                sample_rate=sample_rate,
                resolved_language=resolved_language,
                duration_s=duration_s,
                temperature=temperature,
                lock_wait_s=time.perf_counter() - lock_wait_start,
            )

    def _transcribe_pcm_locked(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int,
        resolved_language: str,
        duration_s: float,
        temperature: float,
        lock_wait_s: float = 0.0,
    ) -> TranscriptionResult:
        text_mode = self.settings_store.load().speaker_text_mode
        stages: dict[str, float] = {"lock_wait": lock_wait_s}
        stage_start = time.perf_counter()

        if is_effectively_silent(audio_data):
            LOGGER.info("No speech detected above silence threshold; returning empty transcription")
            return TranscriptionResult(
                text="",
                language=resolved_language,
                duration=round(duration_s, 2),
                processing_time=0.0,
                text_mode=text_mode,
            )

        stages["silence"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
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

        stages["vad"] = time.perf_counter() - stage_start
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

        stage_start = time.perf_counter()
        # Single-pass, no diarization: the whole speech span is one
        # segment/speaker; ECAPA identification below still runs on it.
        text_raw = self._whispercpp_transcribe(
            audio_data[crop_start:crop_end], sample_rate, resolved_language, temperature
        )
        segments = (
            [
                {
                    "speaker": 0,
                    "start": crop_offset,
                    "end": crop_end / sample_rate,
                    "text": text_raw,
                }
            ]
            if text_raw
            else []
        )
        merge_embeddings: dict[int, np.ndarray] = {}
        stages["generate"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        speaker_embeddings = self._identify_speakers(
            segments, audio_data, sample_rate, merge_embeddings
        )
        dominant_name, dominant_score = self._dominant_speaker(segments)
        stages["speaker_id"] = time.perf_counter() - stage_start
        text = render_speaker_text(segments, mode=text_mode)

        speaker_role = None
        utterance_id = None
        stage_start = time.perf_counter()
        if dominant_name and self.speaker_registry is not None:
            speaker_role = read_role(self.speaker_registry.enrollment_dir, dominant_name)
        elif text.strip():
            utterance_id = self._save_pending_utterance(
                segments, audio_data, sample_rate, text, speaker_embeddings
            )
        stages["pending"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
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
        stages["history"] = time.perf_counter() - stage_start

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
        LOGGER.info(
            "STT stages ms: lock_wait=%.0f silence=%.0f vad=%.0f generate=%.0f "
            "speaker_id=%.0f pending=%.0f history=%.0f",
            stages["lock_wait"] * 1000,
            stages["silence"] * 1000,
            stages["vad"] * 1000,
            stages["generate"] * 1000,
            stages["speaker_id"] * 1000,
            stages["pending"] * 1000,
            stages["history"] * 1000,
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
