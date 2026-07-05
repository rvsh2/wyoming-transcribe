import tempfile
import unittest
from pathlib import Path

from transcribe_wyoming.settings import (
    DEFAULT_SPEAKER_TEXT_MODE,
    SPEAKER_TEXT_MODES,
    SettingsStore,
)


class SettingsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / ".settings.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_missing_file_returns_default(self):
        store = SettingsStore(self.path)
        self.assertEqual(store.load().speaker_text_mode, DEFAULT_SPEAKER_TEXT_MODE)

    def test_save_and_load_roundtrip(self):
        store = SettingsStore(self.path)
        store.save(speaker_text_mode="field")
        self.assertEqual(store.load().speaker_text_mode, "field")

        # A second store (the other process) sees the same file.
        other = SettingsStore(self.path)
        self.assertEqual(other.load().speaker_text_mode, "field")

    def test_save_rejects_invalid_mode(self):
        store = SettingsStore(self.path)
        with self.assertRaises(ValueError):
            store.save(speaker_text_mode="loud")

    def test_invalid_mode_in_file_falls_back_to_default(self):
        self.path.write_text('{"speaker_text_mode": "bogus"}', encoding="utf-8")
        store = SettingsStore(self.path, default_mode="both")
        self.assertEqual(store.load().speaker_text_mode, "both")

    def test_invalid_default_mode_falls_back(self):
        store = SettingsStore(self.path, default_mode="nonsense")
        self.assertEqual(store.load().speaker_text_mode, DEFAULT_SPEAKER_TEXT_MODE)

    def test_modes_constant_matches_expectations(self):
        self.assertEqual(SPEAKER_TEXT_MODES, ("prefix", "field", "both"))


if __name__ == "__main__":
    unittest.main()
