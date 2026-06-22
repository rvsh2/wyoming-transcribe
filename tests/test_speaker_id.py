"""Unit tests for SpeakerRegistry identification and enrollment logic.

The ECAPA encoder is never loaded here; embeddings are injected/mocked so the
tests stay fast and offline.
"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from cohere_wyoming.speaker_id import SpeakerMatch, SpeakerRegistry


def unit(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class IdentifyTests(unittest.TestCase):
    def _registry(self, **kwargs) -> SpeakerRegistry:
        return SpeakerRegistry("/nonexistent", **kwargs)

    def test_disabled_registry_returns_no_match(self):
        reg = self._registry(enabled=False)
        reg._profiles = {"alice": unit([1, 0, 0])}
        self.assertEqual(reg.identify(np.ones(8000, dtype=np.float32)), SpeakerMatch(None, 0.0))

    def test_no_profiles_returns_no_match(self):
        reg = self._registry()
        self.assertEqual(reg.identify(np.ones(8000, dtype=np.float32)), SpeakerMatch(None, 0.0))

    def test_matches_nearest_profile_above_threshold(self):
        reg = self._registry(threshold=0.35)
        reg._profiles = {"alice": unit([1, 0, 0]), "bob": unit([0, 1, 0])}
        reg.embed = lambda audio: unit([0.9, 0.1, 0.0])
        match = reg.identify(np.ones(8000, dtype=np.float32))
        self.assertEqual(match.name, "alice")
        self.assertGreater(match.score, 0.35)

    def test_below_threshold_returns_no_name_but_score(self):
        reg = self._registry(threshold=0.5)
        reg._profiles = {"alice": unit([1, 0, 0]), "bob": unit([0, 1, 0])}
        reg.embed = lambda audio: unit([0.0, 0.0, 1.0])
        match = reg.identify(np.ones(8000, dtype=np.float32))
        self.assertIsNone(match.name)

    def test_short_segment_without_embedding_returns_no_match(self):
        reg = self._registry()
        reg._profiles = {"alice": unit([1, 0, 0])}
        reg.embed = lambda audio: None
        self.assertIsNone(reg.identify(np.ones(10, dtype=np.float32)).name)


class ReloadTests(unittest.TestCase):
    def test_builds_profiles_and_detects_changes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "alice").mkdir()
            (root / "bob").mkdir()
            (root / "alice" / "a1.wav").write_bytes(b"x")
            (root / "bob" / "b1.wav").write_bytes(b"y")

            reg = SpeakerRegistry(root)
            vectors = {"alice": unit([1, 0, 0]), "bob": unit([0, 1, 0])}

            with patch.object(reg, "_embed_sample", side_effect=lambda wav: vectors[wav.parent.name]):
                changed = reg.reload_if_changed()
                self.assertTrue(changed)
                self.assertEqual(sorted(reg._profiles), ["alice", "bob"])
                self.assertTrue(reg.has_profiles())

                # No change -> no reload.
                self.assertFalse(reg.reload_if_changed())

                # Add a person -> reload.
                (root / "carol").mkdir()
                (root / "carol" / "c1.wav").write_bytes(b"z")
                vectors["carol"] = unit([0, 0, 1])
                self.assertTrue(reg.reload_if_changed())
                self.assertIn("carol", reg._profiles)

    def test_missing_directory_yields_no_profiles(self):
        reg = SpeakerRegistry("/does/not/exist")
        reg.reload_if_changed()
        self.assertFalse(reg.has_profiles())
        # Stable once loaded: a second call with no change does not reload.
        self.assertFalse(reg.reload_if_changed())


if __name__ == "__main__":
    unittest.main()
