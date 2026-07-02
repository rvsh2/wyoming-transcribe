"""Runtime settings shared between the HTTP/UI process and the Wyoming process.

Settings changed via the management API are persisted to a JSON file inside the
enrollment directory (already shared between both processes), and picked up by
the Wyoming process on the next transcription — the same mechanism used for
enrollment changes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("cohere-wyoming.settings")

# How speaker identity is delivered to the Wyoming/HTTP consumer:
#   prefix - transcript text lines are prefixed ("Krzysztof: ..."), no event field
#   field  - plain transcript text; identity only in the Transcript event's
#            "speaker" field (and verbose_json)
#   both   - prefixed text and the event field
SPEAKER_TEXT_MODES = ("prefix", "field", "both")
DEFAULT_SPEAKER_TEXT_MODE = "prefix"

SETTINGS_FILENAME = ".settings.json"


def _validated_mode(mode: Optional[str], fallback: str) -> str:
    if mode is None:
        return fallback
    normalized = mode.strip().lower()
    if normalized not in SPEAKER_TEXT_MODES:
        LOGGER.warning(
            "Unknown speaker text mode '%s'; using '%s' (valid: %s)",
            mode,
            fallback,
            ", ".join(SPEAKER_TEXT_MODES),
        )
        return fallback
    return normalized


@dataclass
class RuntimeSettings:
    speaker_text_mode: str = DEFAULT_SPEAKER_TEXT_MODE


class SettingsStore:
    """mtime-cached reader/writer of the shared settings file."""

    def __init__(self, path: str | os.PathLike, *, default_mode: str = DEFAULT_SPEAKER_TEXT_MODE):
        self.path = Path(path)
        self._default = RuntimeSettings(
            speaker_text_mode=_validated_mode(default_mode, DEFAULT_SPEAKER_TEXT_MODE)
        )
        self._cache: Optional[RuntimeSettings] = None
        self._mtime: Optional[float] = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "SettingsStore":
        enrollment_dir = os.environ.get("SPEAKER_ENROLLMENT_DIR", "speakers")
        default_mode = os.environ.get("SPEAKER_TEXT_MODE", DEFAULT_SPEAKER_TEXT_MODE)
        return cls(Path(enrollment_dir) / SETTINGS_FILENAME, default_mode=default_mode)

    def load(self) -> RuntimeSettings:
        with self._lock:
            try:
                mtime = self.path.stat().st_mtime
            except OSError:
                self._cache = None
                self._mtime = None
                return self._default

            if self._cache is not None and mtime == self._mtime:
                return self._cache

            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception as error:
                LOGGER.warning("Could not read settings file %s: %s", self.path, error)
                return self._default

            settings = RuntimeSettings(
                speaker_text_mode=_validated_mode(
                    data.get("speaker_text_mode"), self._default.speaker_text_mode
                )
            )
            self._cache = settings
            self._mtime = mtime
            return settings

    def save(self, *, speaker_text_mode: str) -> RuntimeSettings:
        normalized = speaker_text_mode.strip().lower()
        if normalized not in SPEAKER_TEXT_MODES:
            raise ValueError(
                f"Invalid speaker_text_mode '{speaker_text_mode}'; "
                f"valid values: {', '.join(SPEAKER_TEXT_MODES)}"
            )
        settings = RuntimeSettings(speaker_text_mode=normalized)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps({"speaker_text_mode": settings.speaker_text_mode}),
                encoding="utf-8",
            )
            tmp.replace(self.path)
            self._cache = settings
            self._mtime = self.path.stat().st_mtime
        return settings
