"""Tests for the EnrollmentStore filesystem CRUD layer."""
import io
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from transcribe_wyoming.enrollment import (
    EnrollmentError,
    EnrollmentStore,
    read_role,
    safe_person,
    safe_sample_id,
)


def wav_bytes(duration_s: float = 0.5, sample_rate: int = 16000) -> bytes:
    frames = int(duration_s * sample_rate)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    pcm = np.clip(0.25 * np.sin(2 * np.pi * 440.0 * t) * 32767.0, -32768, 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class SanitizationTests(unittest.TestCase):
    def test_safe_person_strips_unsafe_characters(self):
        self.assertEqual(safe_person("  Krzysztof R.  "), "Krzysztof R")
        self.assertEqual(safe_person("Anna_Łąka-1"), "Anna_Łąka-1")

    def test_safe_person_rejects_traversal(self):
        for bad in ["", "..", "/", "."]:
            with self.assertRaises(EnrollmentError):
                safe_person(bad)
        # Traversal characters are stripped, neutralizing the path rather than erroring.
        self.assertEqual(safe_person("../../etc"), "etc")
        self.assertEqual(safe_person("a/../b"), "ab")

    def test_safe_sample_id_rejects_traversal_and_non_wav(self):
        for bad in ["../x.wav", "a/b.wav", "evil.sh", "", "x.txt"]:
            with self.assertRaises(EnrollmentError):
                safe_sample_id(bad)
        self.assertEqual(safe_sample_id("sample-1.wav"), "sample-1.wav")


class EnrollmentStoreTests(unittest.TestCase):
    def test_full_crud_lifecycle(self):
        with TemporaryDirectory() as tmp:
            store = EnrollmentStore(tmp)
            self.assertEqual(store.list_speakers(), [])

            store.create_speaker("Krzysztof")
            self.assertTrue((Path(tmp) / "Krzysztof").is_dir())

            sample = store.add_sample("Krzysztof", wav_bytes(0.6), "rec.wav")
            self.assertTrue(sample["id"].endswith(".wav"))
            self.assertGreater(sample["seconds"], 0.4)

            speakers = store.list_speakers()
            self.assertEqual(len(speakers), 1)
            self.assertEqual(speakers[0]["name"], "Krzysztof")
            self.assertEqual(len(speakers[0]["samples"]), 1)

            path = store.sample_path("Krzysztof", sample["id"])
            self.assertTrue(path.is_file())

            store.delete_sample("Krzysztof", sample["id"])
            self.assertEqual(store.list_speakers()[0]["samples"], [])

            store.delete_speaker("Krzysztof")
            self.assertEqual(store.list_speakers(), [])

    def test_uploaded_sample_is_normalized_to_16k_mono(self):
        with TemporaryDirectory() as tmp:
            store = EnrollmentStore(tmp)
            store.create_speaker("Anna")
            sample = store.add_sample("Anna", wav_bytes(0.5, sample_rate=44100), "hi.wav")
            import soundfile as sf

            info = sf.info(str(store.sample_path("Anna", sample["id"])))
            self.assertEqual(info.samplerate, 16000)
            self.assertEqual(info.channels, 1)

    def test_empty_upload_rejected(self):
        with TemporaryDirectory() as tmp:
            store = EnrollmentStore(tmp)
            with self.assertRaises(EnrollmentError):
                store.add_sample("Anna", b"", "empty.wav")

    def test_delete_missing_speaker_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(EnrollmentError):
                EnrollmentStore(tmp).delete_speaker("ghost")

    def test_roles_default_set_and_validate(self):
        with TemporaryDirectory() as tmp:
            store = EnrollmentStore(tmp)
            store.create_speaker("Krzysztof")

            # Default role for anyone (also unknown names) is "user".
            self.assertEqual(read_role(tmp, "Krzysztof"), "user")
            self.assertEqual(read_role(tmp, "ghost"), "user")
            self.assertEqual(store.list_speakers()[0]["role"], "user")

            store.set_role("Krzysztof", "admin")
            self.assertEqual(read_role(tmp, "Krzysztof"), "admin")
            self.assertEqual(store.list_speakers()[0]["role"], "admin")

            with self.assertRaises(EnrollmentError):
                store.set_role("Krzysztof", "root")
            with self.assertRaises(EnrollmentError):
                store.set_role("ghost", "admin")

    def test_role_meta_file_does_not_break_sample_listing(self):
        with TemporaryDirectory() as tmp:
            store = EnrollmentStore(tmp)
            store.create_speaker("Anna")
            store.set_role("Anna", "guest")
            store.add_sample("Anna", wav_bytes(0.5), "hi.wav")
            speakers = store.list_speakers()
            self.assertEqual(len(speakers[0]["samples"]), 1)
            self.assertEqual(speakers[0]["role"], "guest")


if __name__ == "__main__":
    unittest.main()
