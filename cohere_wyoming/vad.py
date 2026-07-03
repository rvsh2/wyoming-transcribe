"""Silero VAD integration with safe fallbacks for Wyoming/HTTP transcription."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch


LOGGER = logging.getLogger("cohere-wyoming.vad")


def _read_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _read_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        LOGGER.warning("Invalid integer value for %s=%r; using default=%s", name, value, default)
        return default


def _read_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        LOGGER.warning("Invalid float value for %s=%r; using default=%s", name, value, default)
        return default


@dataclass(frozen=True)
class VadConfig:
    enabled: bool = True
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    min_silence_duration_ms: int = 100
    speech_pad_ms: int = 30
    min_total_speech_ms: int = 60
    min_max_segment_ms: int = 40
    min_speech_rms: float = 0.012
    min_speech_to_noise_ratio: float = 3.0
    use_onnx: bool = False

    @classmethod
    def from_env(cls, **overrides) -> "VadConfig":
        config = cls(
            enabled=_read_bool_env("VAD_ENABLED", True),
            threshold=_read_float_env("VAD_THRESHOLD", 0.5),
            min_speech_duration_ms=_read_int_env("VAD_MIN_SPEECH_DURATION_MS", 250),
            min_silence_duration_ms=_read_int_env("VAD_MIN_SILENCE_DURATION_MS", 100),
            speech_pad_ms=_read_int_env("VAD_SPEECH_PAD_MS", 30),
            min_total_speech_ms=_read_int_env("VAD_MIN_TOTAL_SPEECH_MS", 60),
            min_max_segment_ms=_read_int_env("VAD_MIN_MAX_SEGMENT_MS", 40),
            min_speech_rms=_read_float_env("VAD_MIN_SPEECH_RMS", 0.012),
            min_speech_to_noise_ratio=_read_float_env("VAD_MIN_SPEECH_TO_NOISE_RATIO", 3.0),
            use_onnx=_read_bool_env("VAD_USE_ONNX", False),
        )
        return replace(config, **overrides)


@dataclass(frozen=True)
class VadDecision:
    has_speech: bool
    reason: str
    speech_segments: int
    total_speech_ms: int
    max_segment_ms: int
    speech_rms: float
    noise_rms: float
    speech_to_noise_ratio: float
    # Sample range spanning first to last detected speech (None when Silero
    # timestamps are unavailable, e.g. disabled/fallback modes).
    speech_start_sample: Optional[int] = None
    speech_end_sample: Optional[int] = None


class SileroVoiceActivityDetector:
    """Optional speech detector used to short-circuit silent/noisy audio."""

    def __init__(self, config: Optional[VadConfig] = None) -> None:
        self.config = config or VadConfig.from_env()
        self._model = None
        self._get_speech_timestamps = None
        self._load_silero_vad = None
        self._last_error: Optional[str] = None
        self._mode = "disabled" if not self.config.enabled else "fallback"

    def update_config(self, config: VadConfig) -> None:
        self.config = config
        self._last_error = None
        if not self.config.enabled:
            self._mode = "disabled"
            return
        self._mode = "fallback" if self._model is None else "silero"

    def status_payload(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "mode": self._mode,
            "threshold": self.config.threshold,
            "min_speech_duration_ms": self.config.min_speech_duration_ms,
            "min_silence_duration_ms": self.config.min_silence_duration_ms,
            "speech_pad_ms": self.config.speech_pad_ms,
            "min_total_speech_ms": self.config.min_total_speech_ms,
            "min_max_segment_ms": self.config.min_max_segment_ms,
            "min_speech_rms": self.config.min_speech_rms,
            "min_speech_to_noise_ratio": self.config.min_speech_to_noise_ratio,
            "use_onnx": self.config.use_onnx,
            "last_error": self._last_error,
        }

    def _ensure_loaded(self) -> bool:
        if not self.config.enabled:
            self._mode = "disabled"
            return False

        if self._model is not None and self._get_speech_timestamps is not None:
            self._mode = "silero"
            return True

        if self._get_speech_timestamps is None or self._load_silero_vad is None:
            try:
                silero_module = import_module("silero_vad")
                self._get_speech_timestamps = silero_module.get_speech_timestamps
                self._load_silero_vad = silero_module.load_silero_vad
            except Exception as err:
                self._last_error = str(err)
                self._mode = "fallback"
                LOGGER.warning("Silero VAD import failed, using fallback silence detection: %s", err)
                return False

        try:
            self._model = self._load_silero_vad(onnx=self.config.use_onnx)
            self._mode = "silero"
            LOGGER.info("Silero VAD loaded (onnx=%s)", self.config.use_onnx)
            return True
        except Exception as err:
            self._last_error = str(err)
            self._mode = "fallback"
            LOGGER.warning("Silero VAD load failed, using fallback silence detection: %s", err)
            return False

    @staticmethod
    def _build_decision(
        *,
        has_speech: bool,
        reason: str,
        speech_segments: int = 0,
        total_speech_ms: int = 0,
        max_segment_ms: int = 0,
        speech_rms: float = 0.0,
        noise_rms: float = 0.0,
        speech_to_noise_ratio: float = 0.0,
        speech_start_sample: Optional[int] = None,
        speech_end_sample: Optional[int] = None,
    ) -> VadDecision:
        return VadDecision(
            has_speech=has_speech,
            reason=reason,
            speech_segments=speech_segments,
            total_speech_ms=total_speech_ms,
            max_segment_ms=max_segment_ms,
            speech_rms=speech_rms,
            noise_rms=noise_rms,
            speech_to_noise_ratio=speech_to_noise_ratio,
            speech_start_sample=speech_start_sample,
            speech_end_sample=speech_end_sample,
        )

    def detect_speech(
        self,
        audio_data: np.ndarray,
        *,
        sample_rate: int = 16000,
    ) -> VadDecision:
        if not self.config.enabled:
            return self._build_decision(has_speech=True, reason="disabled")

        if sample_rate not in {8000, 16000}:
            LOGGER.info(
                "Silero VAD supports 8k/16k audio; skipping VAD for sample_rate=%s",
                sample_rate,
            )
            return self._build_decision(has_speech=True, reason="unsupported_sample_rate")

        if not self._ensure_loaded():
            return self._build_decision(has_speech=True, reason="fallback")

        audio_tensor = torch.from_numpy(np.asarray(audio_data, dtype=np.float32)).cpu()

        try:
            timestamps = self._get_speech_timestamps(
                audio_tensor,
                self._model,
                threshold=self.config.threshold,
                sampling_rate=sample_rate,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                min_silence_duration_ms=self.config.min_silence_duration_ms,
                speech_pad_ms=self.config.speech_pad_ms,
                return_seconds=False,
            )
        except Exception as err:
            self._last_error = str(err)
            self._mode = "fallback"
            LOGGER.warning("Silero VAD inference failed, allowing transcription fallback: %s", err)
            return self._build_decision(has_speech=True, reason="inference_failed")

        if not timestamps:
            return self._build_decision(has_speech=False, reason="no_segments")

        total_speech_samples = 0
        max_segment_samples = 0
        speech_samples: list[np.ndarray] = []
        noise_mask = np.ones(len(audio_data), dtype=bool)
        speech_start_sample: Optional[int] = None
        speech_end_sample: Optional[int] = None
        for segment in timestamps:
            start = int(segment.get("start", 0))
            end = int(segment.get("end", 0))
            duration = max(0, end - start)
            total_speech_samples += duration
            max_segment_samples = max(max_segment_samples, duration)
            if duration > 0:
                clipped_start = max(0, min(start, len(audio_data)))
                clipped_end = max(clipped_start, min(end, len(audio_data)))
                if clipped_end > clipped_start:
                    speech_samples.append(audio_data[clipped_start:clipped_end])
                    noise_mask[clipped_start:clipped_end] = False
                    if speech_start_sample is None:
                        speech_start_sample = clipped_start
                    else:
                        speech_start_sample = min(speech_start_sample, clipped_start)
                    if speech_end_sample is None:
                        speech_end_sample = clipped_end
                    else:
                        speech_end_sample = max(speech_end_sample, clipped_end)

        total_speech_ms = int(round(total_speech_samples * 1000 / sample_rate))
        max_segment_ms = int(round(max_segment_samples * 1000 / sample_rate))
        if speech_samples:
            speech_concat = np.concatenate(speech_samples).astype(np.float32, copy=False)
            speech_rms = float(np.sqrt(np.mean(np.square(speech_concat, dtype=np.float32))))
        else:
            speech_rms = 0.0
        noise_samples = audio_data[noise_mask]
        if noise_samples.size:
            noise_rms = float(np.sqrt(np.mean(np.square(noise_samples.astype(np.float32, copy=False), dtype=np.float32))))
        else:
            noise_rms = 0.0
        speech_to_noise_ratio = speech_rms / max(noise_rms, 1e-6)
        has_speech = (
            total_speech_ms >= self.config.min_total_speech_ms
            and max_segment_ms >= self.config.min_max_segment_ms
            and speech_rms >= self.config.min_speech_rms
            and speech_to_noise_ratio >= self.config.min_speech_to_noise_ratio
        )
        if has_speech:
            reason = "speech_detected"
        elif speech_rms < self.config.min_speech_rms:
            reason = "speech_too_quiet"
        elif speech_to_noise_ratio < self.config.min_speech_to_noise_ratio:
            reason = "speech_too_close_to_noise"
        else:
            reason = "segments_too_short"
        return self._build_decision(
            has_speech=has_speech,
            reason=reason,
            speech_segments=len(timestamps),
            total_speech_ms=total_speech_ms,
            max_segment_ms=max_segment_ms,
            speech_rms=round(speech_rms, 6),
            noise_rms=round(noise_rms, 6),
            speech_to_noise_ratio=round(speech_to_noise_ratio, 3),
            speech_start_sample=speech_start_sample,
            speech_end_sample=speech_end_sample,
        )
