"""Speaker enrollment and identification via ECAPA-TDNN embeddings.

Anonymous diarized segments from the ASR model are matched to enrolled people by
comparing each segment's ECAPA voiceprint against per-person enrollment profiles.

Enrollment layout (one directory per person, any number of samples)::

    <enrollment_dir>/
        alice/  sample1.wav  sample2.wav
        bob/    sample1.wav

Matching is done per segment (not per diarized speaker label) because the ASR
model's own speaker labels are not always reliable.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


LOGGER = logging.getLogger("cohere-wyoming.speaker_id")

DEFAULT_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_THRESHOLD = 0.35
TARGET_SR = 16000

# --- profile adaptation from confident recognitions -------------------------
# A recognition with score >= this adds the utterance's embedding to the
# person's adaptive voiceprint, so profiles track microphones/voice drift.
DEFAULT_ADAPT_MIN_SCORE = 0.60
# Skip adaptation when another profile scores within this margin of the best
# match — an ambiguous embedding must not poison either profile.
ADAPT_MARGIN = 0.10
# Running-mean cap: adaptation stays responsive instead of freezing over time.
ADAPT_COUNT_CAP = 50
ADAPT_FILENAME = ".adapt.json"
# Segments shorter than this are too short for a reliable voiceprint.
MIN_MATCH_SAMPLES = int(0.4 * TARGET_SR)
# ECAPA gains nothing from very long clips; cap to bound batch padding/memory
# and keep enrollment and match-time embeddings computed identically.
MAX_MATCH_SAMPLES = int(15 * TARGET_SR)
CACHE_FILENAME = ".embeddings.json"


@dataclass
class SpeakerMatch:
    """Result of identifying one audio segment."""

    name: Optional[str]
    score: float


class SpeakerRegistry:
    """Lazy-loading ECAPA registry with per-person enrollment profiles."""

    def __init__(
        self,
        enrollment_dir: str | os.PathLike,
        *,
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[str] = None,
        enabled: bool = True,
        model_cache_dir: Optional[str] = None,
        adapt_enabled: bool = True,
        adapt_min_score: float = DEFAULT_ADAPT_MIN_SCORE,
    ) -> None:
        self.enrollment_dir = Path(enrollment_dir)
        self.model_name = model_name
        self.threshold = threshold
        self.prefer_device = device
        self.enabled = enabled
        self.model_cache_dir = model_cache_dir
        self.adapt_enabled = adapt_enabled
        self.adapt_min_score = adapt_min_score

        self._encoder = None
        self._device: Optional[str] = None
        self._profiles: dict[str, np.ndarray] = {}
        # person -> (running-mean adaptation vector, count); see adapt().
        self._adapt: dict[str, tuple[np.ndarray, int]] = {}
        self._file_cache: dict[str, dict] = {}
        self._signature: Optional[tuple] = None
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, *, device: Optional[str] = None, **overrides) -> "SpeakerRegistry":
        """Build a registry from SPEAKER_* environment variables (disabled by default)."""

        def _flag(name: str) -> bool:
            return os.environ.get(name, "false").strip().lower() in {"1", "true", "yes", "on"}

        params = {
            "enabled": _flag("SPEAKER_ID_ENABLED"),
            "model_name": os.environ.get("SPEAKER_MODEL", DEFAULT_MODEL),
            "threshold": float(os.environ.get("SPEAKER_MATCH_THRESHOLD", DEFAULT_THRESHOLD)),
            "device": device,
            "model_cache_dir": os.environ.get("SPEAKER_MODEL_CACHE") or None,
            "adapt_enabled": os.environ.get("SPEAKER_ADAPT_ENABLED", "true").strip().lower()
            not in {"0", "false", "no", "off"},
            "adapt_min_score": float(
                os.environ.get("SPEAKER_ADAPT_MIN_SCORE", DEFAULT_ADAPT_MIN_SCORE)
            ),
        }
        params.update(overrides)
        enrollment_dir = os.environ.get("SPEAKER_ENROLLMENT_DIR", "speakers")
        return cls(enrollment_dir, **params)

    # ------------------------------------------------------------------ model
    def _resolve_device(self) -> str:
        if self.prefer_device == "cpu":
            return "cpu"
        try:
            import torch

            # speechbrain requires an explicit device index ("cuda" alone logs
            # a parse warning and falls back to device 0).
            if self.prefer_device and self.prefer_device.startswith("cuda"):
                return self.prefer_device if ":" in self.prefer_device else "cuda:0"
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:  # pragma: no cover - torch always present in runtime
            pass
        return "cpu"

    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        self._device = self._resolve_device()
        savedir = self.model_cache_dir or str(self.enrollment_dir.parent / ".ecapa")
        LOGGER.info("Loading speaker embedding model %s on %s", self.model_name, self._device)
        self._encoder = EncoderClassifier.from_hparams(
            source=self.model_name,
            savedir=savedir,
            run_opts={"device": self._device},
        )

    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Return an L2-normalized ECAPA embedding, or None if audio is too short.

        Single-clip convenience wrapper so enrollment and match-time embeddings
        go through the exact same code path (embed_batch).
        """
        return self.embed_batch([audio])[0]

    def embed_batch(self, clips: list[Optional[np.ndarray]]) -> list[Optional[np.ndarray]]:
        """Embed several segments in one padded ECAPA forward pass.

        Returns one L2-normalized embedding per input position; None for clips
        that are missing or too short. Clips are capped at MAX_MATCH_SAMPLES to
        bound the padded batch width.
        """
        results: list[Optional[np.ndarray]] = [None] * len(clips)
        valid = [
            (index, np.asarray(clip, dtype=np.float32)[:MAX_MATCH_SAMPLES])
            for index, clip in enumerate(clips)
            if clip is not None and len(clip) >= MIN_MATCH_SAMPLES
        ]
        if not valid:
            return results

        import torch

        self._ensure_encoder()
        max_len = max(len(clip) for _, clip in valid)
        batch = torch.zeros(len(valid), max_len, dtype=torch.float32)
        wav_lens = torch.zeros(len(valid), dtype=torch.float32)
        for row, (_, clip) in enumerate(valid):
            batch[row, : len(clip)] = torch.from_numpy(clip)
            wav_lens[row] = len(clip) / max_len
        if self._device and self._device.startswith("cuda"):
            batch = batch.to(self._device)
            wav_lens = wav_lens.to(self._device)
        with torch.no_grad():
            embeddings = self._encoder.encode_batch(batch, wav_lens=wav_lens)
        embeddings = embeddings.squeeze(1).detach().cpu().numpy()

        for row, (index, _) in enumerate(valid):
            vec = np.asarray(embeddings[row], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(vec))
            if norm > 0.0:
                results[index] = vec / norm
        return results

    # ------------------------------------------------------------- enrollment
    def _scan(self) -> list[tuple[str, Path]]:
        """Return (person, wav_path) pairs sorted deterministically."""
        if not self.enrollment_dir.is_dir():
            return []
        pairs: list[tuple[str, Path]] = []
        for person_dir in sorted(p for p in self.enrollment_dir.iterdir() if p.is_dir()):
            if person_dir.name.startswith("."):
                continue
            for wav in sorted(person_dir.glob("*.wav")):
                pairs.append((person_dir.name, wav))
        return pairs

    @staticmethod
    def _signature_of(pairs: list[tuple[str, Path]]) -> tuple:
        signature = []
        for person, wav in pairs:
            stat = wav.stat()
            signature.append((person, wav.name, int(stat.st_mtime), stat.st_size))
        return tuple(signature)

    def _load_disk_cache(self) -> None:
        cache_path = self.enrollment_dir / CACHE_FILENAME
        if not cache_path.is_file():
            return
        try:
            self._file_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Could not read embedding cache: %s", error)
            self._file_cache = {}

    def _save_disk_cache(self) -> None:
        cache_path = self.enrollment_dir / CACHE_FILENAME
        try:
            tmp = cache_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self._file_cache), encoding="utf-8")
            tmp.replace(cache_path)
        except Exception as error:  # pragma: no cover - defensive
            LOGGER.warning("Could not write embedding cache: %s", error)

    def _embed_sample(self, wav: Path) -> Optional[np.ndarray]:
        key = str(wav.relative_to(self.enrollment_dir))
        stat = wav.stat()
        cached = self._file_cache.get(key)
        if cached and cached.get("mtime") == int(stat.st_mtime) and cached.get("size") == stat.st_size:
            return np.asarray(cached["emb"], dtype=np.float32)

        import librosa

        audio, _ = librosa.load(str(wav), sr=TARGET_SR, mono=True)
        embedding = self.embed(np.asarray(audio, dtype=np.float32))
        if embedding is None:
            return None
        self._file_cache[key] = {
            "mtime": int(stat.st_mtime),
            "size": stat.st_size,
            "emb": embedding.tolist(),
        }
        return embedding

    def reload_if_changed(self) -> bool:
        """Rebuild profiles if the enrollment directory changed. Returns True if reloaded.

        The per-file stat scan runs on every call; a directory-mtime short-circuit
        was considered but rejected because second-granular mtimes can miss a
        change made within the same second as the previous scan.
        """
        with self._lock:
            pairs = self._scan()
            signature = self._signature_of(pairs)
            if signature == self._signature:
                return False

            if not self._file_cache:
                self._load_disk_cache()

            samples: dict[str, list[np.ndarray]] = {}
            for person, wav in pairs:
                embedding = self._embed_sample(wav)
                if embedding is not None:
                    samples.setdefault(person, []).append(embedding)

            profiles: dict[str, np.ndarray] = {}
            for person, embeddings in samples.items():
                mean = np.mean(np.stack(embeddings), axis=0)
                norm = float(np.linalg.norm(mean))
                if norm > 0.0:
                    profiles[person] = mean / norm

            # Drop cache entries for files that no longer exist.
            valid_keys = {str(wav.relative_to(self.enrollment_dir)) for _, wav in pairs}
            self._file_cache = {k: v for k, v in self._file_cache.items() if k in valid_keys}

            self._profiles = profiles
            self._adapt = self._load_adaptation(profiles)
            self._signature = signature
            self._save_disk_cache()
            LOGGER.info("Loaded %d speaker profile(s): %s", len(profiles), ", ".join(sorted(profiles)))
            return True

    def _adapt_path(self, name: str) -> Path:
        return self.enrollment_dir / name / ADAPT_FILENAME

    def _load_adaptation(self, profiles: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, int]]:
        adapt: dict[str, tuple[np.ndarray, int]] = {}
        for name in profiles:
            path = self._adapt_path(name)
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                vector = np.asarray(data["vector"], dtype=np.float32)
                adapt[name] = (vector, int(data.get("count", 1)))
            except Exception as error:
                LOGGER.warning("Unreadable adaptation file %s: %s", path, error)
        return adapt

    # ------------------------------------------------------------- inference
    def has_profiles(self) -> bool:
        return bool(self._profiles)

    def _effective_profile(self, name: str) -> np.ndarray:
        """Enrolled-sample mean blended 1:1 with the adaptive usage voiceprint.

        The enrolled mean never changes, so adaptation cannot fully replace a
        profile — it can only pull it halfway toward real usage conditions.
        """
        profile = self._profiles[name]
        adapt = self._adapt.get(name)
        if adapt is None:
            return profile
        combined = profile + adapt[0]
        norm = float(np.linalg.norm(combined))
        return combined / norm if norm > 0.0 else profile

    def _match_embedding(self, embedding: Optional[np.ndarray]) -> SpeakerMatch:
        if embedding is None or not self._profiles:
            return SpeakerMatch(None, 0.0)

        best_name: Optional[str] = None
        best_score = -1.0
        for name in self._profiles:
            score = float(np.dot(embedding, self._effective_profile(name)))
            if score > best_score:
                best_name, best_score = name, score

        if best_score >= self.threshold:
            return SpeakerMatch(best_name, round(best_score, 3))
        return SpeakerMatch(None, round(best_score, 3))

    def match_embedding(self, embedding: Optional[np.ndarray]) -> SpeakerMatch:
        """Match a precomputed embedding against enrolled profiles."""
        if not self.enabled or not self._profiles:
            return SpeakerMatch(None, 0.0)
        return self._match_embedding(embedding)

    def adapt(self, name: str, embedding: Optional[np.ndarray], score: float) -> bool:
        """Fold a confidently recognized utterance into the person's voiceprint.

        Guards: adaptation must be enabled, the score must clear
        adapt_min_score, and no other profile may score within ADAPT_MARGIN of
        the match (an ambiguous embedding must not poison either profile).
        Returns True when the profile was updated. Never raises.
        """
        if (
            not self.adapt_enabled
            or embedding is None
            or score < self.adapt_min_score
            or name not in self._profiles
        ):
            return False

        try:
            with self._lock:
                for other in self._profiles:
                    if other == name:
                        continue
                    other_score = float(np.dot(embedding, self._effective_profile(other)))
                    if other_score >= score - ADAPT_MARGIN:
                        LOGGER.debug(
                            "Skipping adaptation for %s: ambiguous vs %s (%.3f vs %.3f)",
                            name,
                            other,
                            score,
                            other_score,
                        )
                        return False

                vector, count = self._adapt.get(name, (np.zeros_like(embedding), 0))
                effective_count = min(count, ADAPT_COUNT_CAP)
                updated = (vector * effective_count + embedding) / (effective_count + 1)
                norm = float(np.linalg.norm(updated))
                if norm == 0.0:
                    return False
                updated = updated / norm
                self._adapt[name] = (updated, count + 1)
                self._adapt_path(name).write_text(
                    json.dumps({"vector": updated.tolist(), "count": count + 1}),
                    encoding="utf-8",
                )
            LOGGER.info(
                "Adapted voiceprint of %s (utterance #%d, score %.3f)", name, count + 1, score
            )
            return True
        except Exception as error:
            LOGGER.warning("Voiceprint adaptation for %s failed: %s", name, error)
            return False

    def identify(self, audio: np.ndarray) -> SpeakerMatch:
        """Identify one audio segment against enrolled profiles."""
        if not self.enabled or not self._profiles:
            return SpeakerMatch(None, 0.0)
        return self._match_embedding(self.embed(audio))

    def identify_batch(self, clips: list[Optional[np.ndarray]]) -> list[SpeakerMatch]:
        """Identify several segments using one batched embedding pass."""
        if not self.enabled or not self._profiles:
            return [SpeakerMatch(None, 0.0) for _ in clips]
        return [self._match_embedding(emb) for emb in self.embed_batch(clips)]

    def status_payload(self) -> dict:
        return {
            "enabled": self.enabled,
            "model": self.model_name,
            "threshold": self.threshold,
            "enrollment_dir": str(self.enrollment_dir),
            "speakers": sorted(self._profiles),
            "adaptation": {
                "enabled": self.adapt_enabled,
                "min_score": self.adapt_min_score,
                "counts": {name: count for name, (_v, count) in self._adapt.items()},
            },
        }
