"""Ring buffer of unrecognized-speaker utterances ("who are you?" enrollment).

When speaker identification runs but the dominant speaker of an utterance does
not match any enrolled profile, the Wyoming process saves that speaker's audio
(concatenated segments, 16 kHz mono WAV) plus metadata here. The clip's
``utterance_id`` is exposed in the Transcript event, so an LLM pipeline can ask
"who is speaking?" and claim the clip for a person via
``POST /speakers/{name}/samples/from-utterance/{id}``; the UI lists the same
clips (grouped by voice) for manual verification.

Layout (inside the enrollment dir, ignored by the profile scanner)::

    <enrollment_dir>/.pending/
        utt-<ms>-<hex>.wav    # the audio clip
        utt-<ms>-<hex>.json   # {created, seconds, text, best_match, best_score, embedding}
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


LOGGER = logging.getLogger("cohere-wyoming.pending")

PENDING_DIRNAME = ".pending"

# Clips shorter than this are useless as voice samples and are not kept.
# 0.6 s keeps short "who are you?" answers ("Jestem Anna") buffered — they anchor
# the voice-matched claim_latest flow — while the ECAPA embedding still works
# (its own minimum is 0.4 s).
DEFAULT_MIN_SECONDS = 0.6
# Ring-buffer size; oldest clips are pruned first.
DEFAULT_MAX_CLIPS = 40
# Two pending clips with embedding cosine similarity >= this are treated as the
# same (unknown) voice: grouped in the UI and auto-claimed together. Measured on
# real data: same speaker across different recordings/mics scores ~0.44+, while
# different ECAPA speakers stay well below ~0.25.
DEFAULT_CLUSTER_THRESHOLD = 0.40

_UTTERANCE_ID_RE = re.compile(r"^utt-\d+-[0-9a-f]{8}$")


class PendingError(ValueError):
    """Raised for unknown or invalid utterance ids."""


def safe_utterance_id(utterance_id: str) -> str:
    if not utterance_id or not _UTTERANCE_ID_RE.match(utterance_id):
        raise PendingError("Invalid utterance id")
    return utterance_id


class PendingStore:
    """Filesystem store for unrecognized-voice clips, shared by both processes."""

    def __init__(
        self,
        enrollment_dir: str | os.PathLike,
        *,
        min_seconds: float = DEFAULT_MIN_SECONDS,
        max_clips: int = DEFAULT_MAX_CLIPS,
        cluster_threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    ) -> None:
        self.root = Path(enrollment_dir) / PENDING_DIRNAME
        self.min_seconds = min_seconds
        self.max_clips = max_clips
        self.cluster_threshold = cluster_threshold
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, enrollment_dir: Optional[str | os.PathLike] = None) -> "PendingStore":
        if enrollment_dir is None:
            enrollment_dir = os.environ.get("SPEAKER_ENROLLMENT_DIR", "speakers")
        return cls(
            enrollment_dir,
            min_seconds=float(os.environ.get("PENDING_MIN_SECONDS", DEFAULT_MIN_SECONDS)),
            max_clips=int(os.environ.get("PENDING_MAX_CLIPS", DEFAULT_MAX_CLIPS)),
            cluster_threshold=float(
                os.environ.get("PENDING_CLUSTER_THRESHOLD", DEFAULT_CLUSTER_THRESHOLD)
            ),
        )

    # ----------------------------------------------------------------- write
    def save(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        text: str = "",
        best_match: Optional[str] = None,
        best_score: Optional[float] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """Persist an unrecognized-voice clip; returns its utterance id.

        Returns None (and saves nothing) for clips shorter than min_seconds.
        """
        seconds = len(audio) / sample_rate if sample_rate else 0.0
        if seconds < self.min_seconds:
            return None

        utterance_id = f"utt-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        meta = {
            "id": utterance_id,
            "created": time.time(),
            "seconds": round(seconds, 2),
            "text": text,
            "best_match": best_match,
            "best_score": best_score,
            "embedding": embedding.tolist() if embedding is not None else None,
        }
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            sf.write(str(self.root / f"{utterance_id}.wav"), audio, sample_rate)
            (self.root / f"{utterance_id}.json").write_text(
                json.dumps(meta), encoding="utf-8"
            )
            self._prune_locked()
        return utterance_id

    def _prune_locked(self) -> None:
        clips = sorted(self.root.glob("utt-*.json"))
        excess = len(clips) - self.max_clips
        for meta_path in clips[:max(0, excess)]:
            meta_path.with_suffix(".wav").unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------ read
    def _read_meta(self, meta_path: Path) -> Optional[dict]:
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as error:
            LOGGER.warning("Unreadable pending metadata %s: %s", meta_path, error)
            return None

    def list_clips(self) -> list[dict]:
        """All pending clips (with embeddings), oldest first."""
        if not self.root.is_dir():
            return []
        clips = []
        for meta_path in sorted(self.root.glob("utt-*.json")):
            meta = self._read_meta(meta_path)
            if meta is not None and meta_path.with_suffix(".wav").is_file():
                clips.append(meta)
        return clips

    def clusters(self) -> list[list[dict]]:
        """Group pending clips by voice (greedy embedding-similarity clustering).

        Clips without an embedding each form their own cluster. Cluster order:
        newest activity first; clips inside a cluster oldest first.
        """
        clips = self.list_clips()
        clusters: list[dict] = []  # {"centroid": vec|None, "count": int, "clips": [...]}
        for clip in clips:
            embedding = clip.get("embedding")
            vector = np.asarray(embedding, dtype=np.float32) if embedding else None
            best = None
            if vector is not None:
                best_score = self.cluster_threshold
                for cluster in clusters:
                    if cluster["centroid"] is None:
                        continue
                    centroid = cluster["centroid"] / cluster["count"]
                    norm = float(np.linalg.norm(centroid))
                    if norm == 0.0:
                        continue
                    score = float(np.dot(vector, centroid / norm))
                    if score >= best_score:
                        best, best_score = cluster, score
            if best is None:
                clusters.append(
                    {"centroid": vector.copy() if vector is not None else None,
                     "count": 1 if vector is not None else 0,
                     "clips": [clip]}
                )
            else:
                best["centroid"] = best["centroid"] + vector
                best["count"] += 1
                best["clips"].append(clip)

        ordered = sorted(
            clusters,
            key=lambda cluster: max(c.get("created", 0.0) for c in cluster["clips"]),
            reverse=True,
        )
        return [cluster["clips"] for cluster in ordered]

    def cluster_members(self, utterance_id: str) -> list[str]:
        """Ids of all clips sharing a cluster with the given clip (incl. itself)."""
        utterance_id = safe_utterance_id(utterance_id)
        for cluster in self.clusters():
            ids = [clip["id"] for clip in cluster]
            if utterance_id in ids:
                return ids
        raise PendingError("Utterance not found")

    def latest_voice_stats(self) -> Optional[dict]:
        """Stats of the voice cluster containing the newest pending clip.

        Lets an LLM agent ask "who are you?" only of regulars: a voice that has
        spoken several times has a large cluster, a one-off visitor has one clip.
        Returns None when the buffer is empty.
        """
        clips = self.list_clips()
        if not clips:
            return None
        newest = max(clips, key=lambda clip: clip.get("created", 0.0))
        for cluster in self.clusters():
            ids = [clip["id"] for clip in cluster]
            if newest["id"] in ids:
                return {
                    "utterance_id": newest["id"],
                    "utterances": len(cluster),
                    "seconds": round(sum(clip.get("seconds", 0.0) for clip in cluster), 2),
                    "newest_age_seconds": round(
                        max(0.0, time.time() - float(newest.get("created", 0.0))), 1
                    ),
                    "text": newest.get("text", ""),
                }
        return None  # pragma: no cover - newest clip always belongs to a cluster

    def audio_path(self, utterance_id: str) -> Path:
        path = self.root / f"{safe_utterance_id(utterance_id)}.wav"
        if not path.is_file():
            raise PendingError("Utterance not found")
        return path

    def get_meta(self, utterance_id: str) -> dict:
        meta_path = self.root / f"{safe_utterance_id(utterance_id)}.json"
        meta = self._read_meta(meta_path) if meta_path.is_file() else None
        if meta is None:
            raise PendingError("Utterance not found")
        return meta

    def delete(self, utterance_id: str) -> None:
        path = self.audio_path(utterance_id)
        with self._lock:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
