"""Shared transcription runtime for HTTP and Wyoming transports."""

from .transcriber import SpeechTranscriber, TranscriptionResult

__all__ = ["SpeechTranscriber", "TranscriptionResult"]
