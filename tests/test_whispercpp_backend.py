import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from transcribe_wyoming.transcriber import SpeechTranscriber


def _speechy_vad(num_samples: int):
    return SimpleNamespace(
        detect_speech=lambda *_a, **_k: SimpleNamespace(
            has_speech=True,
            reason="ok",
            speech_segments=1,
            total_speech_ms=1000,
            max_segment_ms=1000,
            speech_rms=0.05,
            noise_rms=0.001,
            speech_to_noise_ratio=50.0,
            speech_start_sample=0,
            speech_end_sample=num_samples,
        )
    )


class WhisperCppBackendTests(unittest.TestCase):
    def _transcriber(self) -> SpeechTranscriber:
        transcriber = SpeechTranscriber(
            whispercpp_url="http://fake:4050"
        )
        return transcriber

    def test_is_loaded_without_local_model(self):
        transcriber = self._transcriber()
        self.assertTrue(transcriber.is_loaded())

    def test_transcribes_via_http_and_builds_single_speaker_segment(self):
        transcriber = self._transcriber()
        num_samples = 16000
        transcriber.vad_detector = _speechy_vad(num_samples)

        posted = {}

        def fake_post(url, files=None, data=None, timeout=None):
            posted["url"] = url
            posted["language"] = data["language"]
            posted["wav_bytes"] = files["file"][1].read()
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"text": " Ala ma kota. "},
            )

        t = np.arange(num_samples, dtype=np.float32) / 16000.0
        audio = 0.05 * np.sin(2 * np.pi * 220.0 * t)
        with patch("requests.post", side_effect=fake_post):
            result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        self.assertEqual(posted["url"], "http://fake:4050/inference")
        self.assertEqual(posted["language"], "pl")
        # 44-byte WAV header + 16-bit samples
        self.assertEqual(len(posted["wav_bytes"]), 44 + num_samples * 2)
        self.assertIn("Ala ma kota.", result.text)
        self.assertEqual(len(result.segments), 1)
        segment = result.segments[0]
        self.assertEqual(segment["speaker"], 0)
        self.assertEqual(segment["start"], 0.0)
        self.assertAlmostEqual(segment["end"], 1.0, places=2)

    def test_empty_server_text_yields_empty_transcript(self):
        transcriber = self._transcriber()
        num_samples = 16000
        transcriber.vad_detector = _speechy_vad(num_samples)

        def fake_post(url, files=None, data=None, timeout=None):
            return SimpleNamespace(
                raise_for_status=lambda: None, json=lambda: {"text": "  "}
            )

        audio = 0.05 * np.ones(num_samples, dtype=np.float32)
        with patch("requests.post", side_effect=fake_post):
            result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        self.assertEqual(result.text, "")
        self.assertEqual(result.segments, [])


if __name__ == "__main__":
    unittest.main()
