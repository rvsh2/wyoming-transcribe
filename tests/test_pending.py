import tempfile
import unittest
from pathlib import Path

import numpy as np

from cohere_wyoming.pending import PendingError, PendingStore


def tone(seconds: float = 2.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return 0.1 * np.sin(2 * np.pi * 220.0 * t)


class PendingStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PendingStore(self._tmp.name, max_clips=3)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_save_and_list_roundtrip(self):
        utterance_id = self.store.save(
            tone(), 16000, text="zgaś światło", embedding=np.array([1.0, 0.0], dtype=np.float32)
        )
        self.assertIsNotNone(utterance_id)

        clips = self.store.list_clips()
        self.assertEqual(len(clips), 1)
        self.assertEqual(clips[0]["id"], utterance_id)
        self.assertEqual(clips[0]["text"], "zgaś światło")
        self.assertAlmostEqual(clips[0]["seconds"], 2.0, places=1)
        self.assertTrue(self.store.audio_path(utterance_id).is_file())

    def test_too_short_clip_is_not_saved(self):
        self.assertIsNone(self.store.save(tone(0.5), 16000, text="krótkie"))
        self.assertEqual(self.store.list_clips(), [])

    def test_prunes_oldest_beyond_max_clips(self):
        ids = [self.store.save(tone(), 16000, text=f"clip {i}") for i in range(5)]
        clips = self.store.list_clips()
        self.assertEqual(len(clips), 3)
        kept = {clip["id"] for clip in clips}
        self.assertEqual(kept, set(ids[2:]))
        # Pruned clips lose both files.
        self.assertFalse((self.store.root / f"{ids[0]}.wav").exists())
        self.assertFalse((self.store.root / f"{ids[0]}.json").exists())

    def test_clusters_group_same_voice(self):
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        a1 = self.store.save(tone(), 16000, text="a1", embedding=alice)
        b1 = self.store.save(tone(), 16000, text="b1", embedding=bob)
        a2 = self.store.save(tone(), 16000, text="a2", embedding=alice)

        clusters = self.store.clusters()
        self.assertEqual(len(clusters), 2)
        grouped = {frozenset(clip["id"] for clip in cluster) for cluster in clusters}
        self.assertIn(frozenset({a1, a2}), grouped)
        self.assertIn(frozenset({b1}), grouped)

        self.assertEqual(set(self.store.cluster_members(a1)), {a1, a2})

    def test_latest_voice_stats(self):
        self.assertIsNone(self.store.latest_voice_stats())

        anna = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        self.store.save(tone(2.0), 16000, text="a1", embedding=anna)
        self.store.save(tone(3.0), 16000, text="bob", embedding=bob)
        newest = self.store.save(tone(4.0), 16000, text="a2", embedding=anna)

        stats = self.store.latest_voice_stats()
        # Newest clip is Anna's; her cluster has 2 clips totalling ~6 s.
        self.assertEqual(stats["utterance_id"], newest)
        self.assertEqual(stats["utterances"], 2)
        self.assertAlmostEqual(stats["seconds"], 6.0, places=1)
        self.assertLess(stats["newest_age_seconds"], 5.0)
        self.assertEqual(stats["text"], "a2")

    def test_delete_and_invalid_ids(self):
        utterance_id = self.store.save(tone(), 16000, text="x")
        self.store.delete(utterance_id)
        self.assertEqual(self.store.list_clips(), [])
        with self.assertRaises(PendingError):
            self.store.audio_path(utterance_id)
        with self.assertRaises(PendingError):
            self.store.audio_path("../../etc/passwd")
        with self.assertRaises(PendingError):
            self.store.cluster_members("utt-1-zzzzzzzz")


if __name__ == "__main__":
    unittest.main()
