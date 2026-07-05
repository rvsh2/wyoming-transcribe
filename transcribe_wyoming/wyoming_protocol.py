"""Optional Wyoming protocol imports with lightweight fallbacks for tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


try:
    from wyoming.asr import Transcribe, Transcript
    from wyoming.audio import AudioChunk, AudioStart, AudioStop
    from wyoming.event import Event
    from wyoming.info import AsrModel, AsrProgram, Attribution, Info
    from wyoming.server import AsyncEventHandler, AsyncServer

    WYOMING_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised indirectly in tests
    WYOMING_AVAILABLE = False

    @dataclass
    class Event:
        type: str
        data: dict[str, Any] = field(default_factory=dict)

    class AsyncEventHandler:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def write_event(self, event: Event) -> None:
            raise NotImplementedError

    class AsyncServer:
        @classmethod
        def from_uri(cls, uri: str, handler_factory):
            raise RuntimeError(
                "The 'wyoming' package is not installed. Run 'uv sync' before starting the Wyoming server."
            )

        async def run(self) -> None:
            raise RuntimeError(
                "The 'wyoming' package is not installed. Run 'uv sync' before starting the Wyoming server."
            )

    @dataclass
    class Attribution:
        name: Optional[str] = None
        url: Optional[str] = None

    @dataclass
    class AsrModel:
        name: str
        description: str = ""
        attribution: Optional[Attribution] = None
        installed: bool = True
        languages: list[str] = field(default_factory=list)
        version: Optional[str] = None

    @dataclass
    class AsrProgram:
        name: str
        description: str = ""
        attribution: Optional[Attribution] = None
        installed: bool = True
        models: list[AsrModel] = field(default_factory=list)
        version: Optional[str] = None

    @dataclass
    class Info:
        asr: list[AsrProgram] = field(default_factory=list)

        def event(self) -> Event:
            return Event("describe", {"asr": self.asr})

    @dataclass
    class Transcribe:
        language: Optional[str] = None

        @staticmethod
        def is_type(event_type: str) -> bool:
            return event_type == "transcribe"

        @classmethod
        def from_event(cls, event: Event) -> "Transcribe":
            return cls(language=event.data.get("language"))

    @dataclass
    class Transcript:
        text: str
        language: Optional[str] = None

        def event(self) -> Event:
            data = {"text": self.text}
            if self.language:
                data["language"] = self.language
            return Event("transcript", data)

    @dataclass
    class AudioStart:
        rate: int = 16000
        width: int = 2
        channels: int = 1

        @staticmethod
        def is_type(event_type: str) -> bool:
            return event_type == "audio-start"

        @classmethod
        def from_event(cls, event: Event) -> "AudioStart":
            return cls(
                rate=event.data.get("rate", 16000),
                width=event.data.get("width", 2),
                channels=event.data.get("channels", 1),
            )

    @dataclass
    class AudioChunk:
        audio: bytes = b""
        rate: int = 16000
        width: int = 2
        channels: int = 1

        @staticmethod
        def is_type(event_type: str) -> bool:
            return event_type == "audio-chunk"

        @classmethod
        def from_event(cls, event: Event) -> "AudioChunk":
            return cls(
                audio=event.data.get("audio", b""),
                rate=event.data.get("rate", 16000),
                width=event.data.get("width", 2),
                channels=event.data.get("channels", 1),
            )

    @dataclass
    class AudioStop:
        @staticmethod
        def is_type(event_type: str) -> bool:
            return event_type == "audio-stop"

        @classmethod
        def from_event(cls, event: Event) -> "AudioStop":
            return cls()
