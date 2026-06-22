"""Filesystem CRUD for speaker enrollment samples, shared by the web UI.

Samples are normalized to 16 kHz mono WAV on upload so they are consistent for
ECAPA embedding regardless of the original container/codec.
"""
from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path

import soundfile as sf

from .audio import TARGET_SAMPLE_RATE, read_audio_to_numpy


_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9 _\-À-ſ]+")


class EnrollmentError(ValueError):
    """Raised for invalid speaker names, sample ids, or unreadable audio."""


def safe_person(name: str) -> str:
    """Sanitize a person name into a safe directory name (no path traversal)."""
    cleaned = _UNSAFE_NAME.sub("", (name or "").strip()).strip()
    if not cleaned or cleaned in {".", ".."}:
        raise EnrollmentError("Invalid speaker name")
    return cleaned[:64]


def safe_sample_id(sample_id: str) -> str:
    """Validate that a sample id is a bare .wav filename."""
    if not sample_id or "/" in sample_id or "\\" in sample_id or ".." in sample_id:
        raise EnrollmentError("Invalid sample id")
    if not sample_id.endswith(".wav"):
        raise EnrollmentError("Invalid sample id")
    return sample_id


class EnrollmentStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _person_dir(self, name: str, *, must_exist: bool = False) -> Path:
        person = safe_person(name)
        path = self.root / person
        if must_exist and not path.is_dir():
            raise EnrollmentError(f"Speaker '{person}' does not exist")
        return path

    def list_speakers(self) -> list[dict]:
        if not self.root.is_dir():
            return []
        speakers = []
        for person_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if person_dir.name.startswith("."):
                continue
            speakers.append(
                {"name": person_dir.name, "samples": self._list_samples(person_dir)}
            )
        return speakers

    def _list_samples(self, person_dir: Path) -> list[dict]:
        samples = []
        for wav in sorted(person_dir.glob("*.wav")):
            try:
                info = sf.info(str(wav))
                seconds = round(info.frames / info.samplerate, 2) if info.samplerate else 0.0
            except Exception:
                seconds = 0.0
            samples.append(
                {
                    "id": wav.name,
                    "seconds": seconds,
                    "bytes": wav.stat().st_size,
                }
            )
        return samples

    def create_speaker(self, name: str) -> str:
        path = self._person_dir(name)
        path.mkdir(parents=True, exist_ok=True)
        return path.name

    def delete_speaker(self, name: str) -> None:
        path = self._person_dir(name, must_exist=True)
        shutil.rmtree(path)

    def add_sample(self, name: str, file_bytes: bytes, filename: str = "audio") -> dict:
        person_dir = self._person_dir(name)
        person_dir.mkdir(parents=True, exist_ok=True)
        if not file_bytes:
            raise EnrollmentError("Empty audio upload")
        try:
            audio, sample_rate = read_audio_to_numpy(file_bytes, filename)
        except ValueError as error:
            raise EnrollmentError(str(error)) from error

        sample_id = f"sample-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.wav"
        sf.write(str(person_dir / sample_id), audio, sample_rate)
        seconds = round(len(audio) / sample_rate, 2) if sample_rate else 0.0
        return {"id": sample_id, "seconds": seconds, "bytes": (person_dir / sample_id).stat().st_size}

    def sample_path(self, name: str, sample_id: str) -> Path:
        person_dir = self._person_dir(name, must_exist=True)
        path = person_dir / safe_sample_id(sample_id)
        if not path.is_file():
            raise EnrollmentError("Sample not found")
        return path

    def delete_sample(self, name: str, sample_id: str) -> None:
        self.sample_path(name, sample_id).unlink()
