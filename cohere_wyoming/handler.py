"""Wyoming protocol handler for Cohere Transcribe."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from .audio import pcm16le_to_float32
from .transcriber import CohereTranscriber
from .wyoming_protocol import (
    AsyncEventHandler,
    AudioChunk,
    AudioStart,
    AudioStop,
    Event,
    Transcribe,
    Transcript,
)


LOGGER = logging.getLogger("cohere-wyoming.handler")


@dataclass
class AudioState:
    sample_rate: int = 16000
    width: int = 2
    channels: int = 1


class CohereWyomingEventHandler(AsyncEventHandler):
    """Collect audio chunks and answer with one final transcript."""

    def __init__(self, transcriber: CohereTranscriber, info_event: Event, *args, **kwargs):
        if args or kwargs:
            super().__init__(*args, **kwargs)
        self.transcriber = transcriber
        self.info_event = info_event
        self.requested_language: Optional[str] = None
        self.audio_state = AudioState()
        self.audio_chunks: list[bytes] = []

    async def handle_event(self, event: Event) -> bool:
        event_type = getattr(event, "type", None)

        if event_type == "describe":
            await self.write_event(self.info_event)
            return True

        if Transcribe.is_type(event_type):
            request = Transcribe.from_event(event)
            self.requested_language = getattr(request, "language", None)
            self.audio_chunks.clear()
            return True

        if AudioStart.is_type(event_type):
            audio_start = AudioStart.from_event(event)
            self.audio_state = AudioState(
                sample_rate=audio_start.rate,
                width=audio_start.width,
                channels=audio_start.channels,
            )
            self.audio_chunks.clear()
            return True

        if AudioChunk.is_type(event_type):
            chunk = AudioChunk.from_event(event)
            if getattr(chunk, "rate", None):
                self.audio_state.sample_rate = chunk.rate
            if getattr(chunk, "width", None):
                self.audio_state.width = chunk.width
            if getattr(chunk, "channels", None):
                self.audio_state.channels = chunk.channels
            self.audio_chunks.append(chunk.audio)
            return True

        if AudioStop.is_type(event_type):
            await self._finalize_transcription()
            return True

        LOGGER.debug("Ignoring unsupported Wyoming event type: %s", event_type)
        return True

    async def _finalize_transcription(self) -> None:
        if not self.audio_chunks:
            await self.write_event(Transcript(text="", language=self.requested_language).event())
            return

        pcm_audio = b"".join(self.audio_chunks)
        self.audio_chunks.clear()

        text = ""
        language = self.requested_language
        extra_data: dict = {}
        try:
            audio_data, sample_rate = pcm16le_to_float32(
                pcm_audio,
                sample_rate=self.audio_state.sample_rate,
                channels=self.audio_state.channels,
                width=self.audio_state.width,
            )
            # Inference takes seconds; run it off the event loop so other
            # connections (describe probes, a second satellite) stay served.
            result = await asyncio.to_thread(
                self.transcriber.transcribe_pcm,
                audio_data,
                sample_rate=sample_rate,
                language=self.requested_language,
            )
            text = result.text
            language = result.language
            # In "field"/"both" mode the dominant speaker's enrolled identity
            # rides along in the event data; HA ignores unknown keys, custom
            # pipeline components can consume them.
            if getattr(result, "text_mode", "prefix") in ("field", "both"):
                extra_data["speaker"] = result.speaker
                if result.speaker_score is not None:
                    extra_data["speaker_score"] = result.speaker_score
                if getattr(result, "speaker_role", None):
                    extra_data["speaker_role"] = result.speaker_role
            # The pending-clip id is sent in every mode: it cannot be derived
            # from the text and the "who are you?" enrollment flow needs it.
            if getattr(result, "utterance_id", None):
                extra_data["utterance_id"] = result.utterance_id
            self.transcriber.failure_streak = 0
        except Exception:
            # Always answer with a Transcript: without one Home Assistant's
            # voice pipeline hangs until its own timeout.
            LOGGER.exception("Transcription failed; returning an empty transcript")
            # Empty transcripts make failures invisible to the voice pipeline
            # ("I didn't understand"), so make persistent breakage loud here.
            streak = getattr(self.transcriber, "failure_streak", 0) + 1
            self.transcriber.failure_streak = streak
            if streak >= 3:
                LOGGER.error(
                    "%d consecutive transcription failures - the ASR server is "
                    "likely broken (check the model / a failed /load), while HA "
                    "only sees empty transcripts",
                    streak,
                )

        event = Transcript(text=text, language=language).event()
        event.data.update(extra_data)
        await self.write_event(event)
