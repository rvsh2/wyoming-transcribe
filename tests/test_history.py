import tempfile
import unittest
from pathlib import Path

from transcribe_wyoming.history import RecognitionLog


class RecognitionLogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.log = RecognitionLog(self._tmp.name, max_entries=10)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_append_and_recent_roundtrip(self):
        self.log.append(
            text="zgaś światło",
            language="pl",
            duration=2.5,
            speaker="Krzysztof",
            score=0.85,
            role="admin",
        )
        self.log.append(text="kto to?", language="pl", duration=1.5, utterance_id="utt-1-aabbccdd")

        entries = self.log.recent()
        self.assertEqual(len(entries), 2)
        # Newest first.
        self.assertEqual(entries[0]["text"], "kto to?")
        self.assertEqual(entries[0]["utterance_id"], "utt-1-aabbccdd")
        self.assertIsNone(entries[0]["speaker"])
        self.assertEqual(entries[1]["speaker"], "Krzysztof")
        self.assertEqual(entries[1]["role"], "admin")

    def test_limit_returns_newest(self):
        for index in range(5):
            self.log.append(text=f"wpis {index}", language="pl", duration=1.0)
        entries = self.log.recent(limit=2)
        self.assertEqual([e["text"] for e in entries], ["wpis 4", "wpis 3"])

    def test_compaction_keeps_newest_max_entries(self):
        for index in range(30):
            self.log.append(text=f"wpis {index}", language="pl", duration=1.0)
        entries = self.log.recent(limit=0)
        self.assertLessEqual(len(entries), 15)  # max_entries * compact factor
        self.assertEqual(entries[0]["text"], "wpis 29")

    def test_append_never_raises_on_broken_dir(self):
        broken = RecognitionLog(Path(self._tmp.name) / "no" / "such")
        broken.path = Path("/proc/definitely-not-writable/x.jsonl")
        broken.append(text="x", language="pl", duration=1.0)  # must not raise

    def test_from_env_disabled(self):
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"HISTORY_ENABLED": "false"}):
            self.assertIsNone(RecognitionLog.from_env(self._tmp.name))


if __name__ == "__main__":
    unittest.main()
