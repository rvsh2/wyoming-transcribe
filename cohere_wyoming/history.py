"""Recognition history: an append-only log of transcription decisions.

Every non-empty transcription appends one JSONL entry (timestamp, transcript,
recognized speaker + score + role, or the pending ``utterance_id`` when the
voice was unknown). The UI shows it as "Dziennik rozpoznań" so threshold tuning
and misidentifications stop being guesswork. Stored inside the enrollment dir,
so the Wyoming process writes it and the UI process reads it.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional


LOGGER = logging.getLogger("cohere-wyoming.history")

HISTORY_FILENAME = ".history.jsonl"
DEFAULT_MAX_ENTRIES = 200
# Compact (rewrite keeping the newest max_entries) once the file grows past
# this multiple, so appends stay cheap.
_COMPACT_FACTOR = 1.5
_MAX_TEXT_LENGTH = 300


class RecognitionLog:
    """Thread-safe JSONL ring log shared between both processes."""

    def __init__(
        self,
        enrollment_dir: str | os.PathLike,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self.path = Path(enrollment_dir) / HISTORY_FILENAME
        self.max_entries = max_entries
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, enrollment_dir: Optional[str | os.PathLike] = None) -> Optional["RecognitionLog"]:
        """Build from env; returns None when HISTORY_ENABLED is falsy."""
        enabled = os.environ.get("HISTORY_ENABLED", "true").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return None
        if enrollment_dir is None:
            enrollment_dir = os.environ.get("SPEAKER_ENROLLMENT_DIR", "speakers")
        return cls(
            enrollment_dir,
            max_entries=int(os.environ.get("HISTORY_MAX_ENTRIES", DEFAULT_MAX_ENTRIES)),
        )

    def append(
        self,
        *,
        text: str,
        language: str,
        duration: float,
        speaker: Optional[str] = None,
        score: Optional[float] = None,
        role: Optional[str] = None,
        utterance_id: Optional[str] = None,
    ) -> None:
        """Append one recognition entry; never raises."""
        entry = {
            "ts": round(time.time(), 3),
            "text": text[:_MAX_TEXT_LENGTH],
            "language": language,
            "duration": duration,
            "speaker": speaker,
            "score": score,
            "role": role,
            "utterance_id": utterance_id,
        }
        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._compact_locked()
        except Exception as error:
            LOGGER.warning("Could not append recognition history: %s", error)

    def _read_entries_locked(self) -> list[dict]:
        if not self.path.is_file():
            return []
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
        return entries

    def _compact_locked(self) -> None:
        try:
            line_count = sum(1 for _ in self.path.open("r", encoding="utf-8"))
        except OSError:
            return
        if line_count <= self.max_entries * _COMPACT_FACTOR:
            return
        entries = self._read_entries_locked()[-self.max_entries :]
        tmp = self.path.with_suffix(".jsonl.tmp")
        tmp.write_text(
            "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def recent(self, limit: int = 50) -> list[dict]:
        """Newest entries first."""
        with self._lock:
            entries = self._read_entries_locked()
        entries = entries[-max(0, limit):] if limit else entries
        return list(reversed(entries))
