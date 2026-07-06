import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from transcribe_wyoming import audio as audio_module
from transcribe_wyoming.transcriber import SpeechTranscriber


class FakeInputs(dict):
    def to(self, *_args, **_kwargs):
        return self


class AudioAndTranscriberTests(unittest.TestCase):
    def test_audio_preprocessing_falls_back_to_ffmpeg_for_webm(self):
        ffmpeg_audio = np.array([0.1, -0.2, 0.3], dtype=np.float32)

        with patch.object(audio_module.sf, "read", side_effect=RuntimeError("sf failed")), patch.object(
            audio_module.librosa, "load", side_effect=RuntimeError("librosa failed")
        ), patch.object(
            audio_module.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=0,
                stdout=ffmpeg_audio.tobytes(),
                stderr=b"",
            ),
        ) as ffmpeg_run:
            audio_data, sample_rate = audio_module.read_audio_to_numpy(
                b"fake-webm", "recording.webm"
            )

        self.assertEqual(sample_rate, 16000)
        self.assertTrue(np.array_equal(audio_data, ffmpeg_audio))
        ffmpeg_run.assert_called_once()

    def test_silent_audio_returns_empty_transcription_without_generate(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        fake_model = SimpleNamespace(device="cpu", dtype=torch.float32)
        fake_model.generate = unittest.mock.Mock()
        fake_processor = unittest.mock.Mock()

        transcriber.model = fake_model
        transcriber.processor = fake_processor

        result = transcriber.transcribe_pcm(
            np.zeros(16000, dtype=np.float32),
            sample_rate=16000,
            language="pl",
        )

        self.assertEqual(result.text, "")
        self.assertEqual(result.language, "pl")
        self.assertEqual(result.duration, 1.0)
        self.assertEqual(result.processing_time, 0.0)
        fake_model.generate.assert_not_called()

    def test_low_level_noise_is_treated_as_silence(self):
        rng = np.random.default_rng(1234)
        low_noise = rng.normal(0.0, 0.003, 16000).astype(np.float32)

        self.assertTrue(audio_module.is_effectively_silent(low_noise))

    def test_speech_like_segment_is_not_treated_as_silence(self):
        audio = np.zeros(16000, dtype=np.float32)
        t = np.arange(3200, dtype=np.float32) / 16000.0
        audio[4800:8000] = 0.05 * np.sin(2 * np.pi * 220.0 * t)

        self.assertFalse(audio_module.is_effectively_silent(audio))

    def test_low_level_noise_returns_empty_transcription_without_generate(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        fake_model = SimpleNamespace(device="cpu", dtype=torch.float32)
        fake_model.generate = unittest.mock.Mock()
        fake_processor = unittest.mock.Mock()

        transcriber.model = fake_model
        transcriber.processor = fake_processor

        rng = np.random.default_rng(4321)
        noisy_silence = rng.normal(0.0, 0.003, 16000).astype(np.float32)
        result = transcriber.transcribe_pcm(
            noisy_silence,
            sample_rate=16000,
            language="pl",
        )

        self.assertEqual(result.text, "")
        fake_model.generate.assert_not_called()

    def test_vad_rejected_audio_returns_empty_transcription_without_generate(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        fake_model = SimpleNamespace(device="cpu", dtype=torch.float32)
        fake_model.generate = unittest.mock.Mock()
        fake_processor = unittest.mock.Mock()

        transcriber.model = fake_model
        transcriber.processor = fake_processor
        transcriber.vad_detector = SimpleNamespace(
            detect_speech=lambda *_args, **_kwargs: SimpleNamespace(
                has_speech=False,
                reason="no_segments",
                speech_segments=0,
                total_speech_ms=0,
                max_segment_ms=0,
                speech_rms=0.0,
                noise_rms=0.0,
                speech_to_noise_ratio=0.0,
            )
        )

        t = np.arange(3200, dtype=np.float32) / 16000.0
        audio = 0.05 * np.sin(2 * np.pi * 220.0 * t)
        result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        self.assertEqual(result.text, "")
        fake_model.generate.assert_not_called()

    def test_vad_fallback_allows_transcription(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        transcriber._whispercpp_transcribe = lambda *_a, **_k: "speech detected"
        transcriber.vad_detector = SimpleNamespace(
            detect_speech=lambda *_args, **_kwargs: SimpleNamespace(
                has_speech=True,
                reason="fallback",
                speech_segments=0,
                total_speech_ms=0,
                max_segment_ms=0,
                speech_rms=0.0,
                noise_rms=0.0,
                speech_to_noise_ratio=0.0,
            )
        )

        t = np.arange(3200, dtype=np.float32) / 16000.0
        audio = 0.05 * np.sin(2 * np.pi * 220.0 * t)
        result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        self.assertEqual(result.text, "Speaker 0: speech detected")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0]["speaker"], 0)
        self.assertEqual(result.segments[0]["text"], "speech detected")

    def test_speech_bounds_pads_and_clips_vad_span(self):
        bounds = SpeechTranscriber._speech_bounds

        span = SimpleNamespace(speech_start_sample=96000, speech_end_sample=112000)
        # 0.1 s pad = 1600 samples at 16 kHz.
        self.assertEqual(bounds(span, 160000, 16000), (94400, 113600))

        at_edges = SimpleNamespace(speech_start_sample=1000, speech_end_sample=159000)
        self.assertEqual(bounds(at_edges, 160000, 16000), (0, 160000))

        no_span = SimpleNamespace(speech_start_sample=None, speech_end_sample=None)
        self.assertEqual(bounds(no_span, 160000, 16000), (0, 160000))

        inverted = SimpleNamespace(speech_start_sample=5000, speech_end_sample=4000)
        self.assertEqual(bounds(inverted, 160000, 16000), (0, 160000))

    def test_transcription_crops_to_vad_speech_span_and_keeps_global_timestamps(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        seen_audio_lengths = []

        def fake_whisper(audio, sample_rate, language, temperature=0.0):
            seen_audio_lengths.append(len(audio))
            return "która godzina"

        transcriber._whispercpp_transcribe = fake_whisper
        # 10 s clip with speech only between 6 s and 7 s.
        transcriber.vad_detector = SimpleNamespace(
            detect_speech=lambda *_args, **_kwargs: SimpleNamespace(
                has_speech=True,
                reason="speech_detected",
                speech_segments=1,
                total_speech_ms=1000,
                max_segment_ms=1000,
                speech_rms=0.05,
                noise_rms=0.005,
                speech_to_noise_ratio=10.0,
                speech_start_sample=6 * 16000,
                speech_end_sample=7 * 16000,
            )
        )

        rng = np.random.default_rng(1234)
        audio = (0.05 * rng.normal(size=10 * 16000)).astype(np.float32)
        result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        # The STT call saw only the padded speech span (1 s + 2 * 0.1 s pad).
        self.assertEqual(seen_audio_lengths, [int(1.2 * 16000)])
        # The single segment spans the padded speech crop on the full-clip
        # timeline (6-7 s speech +/- 0.1 s CROP_PAD_SECONDS).
        self.assertEqual(result.segments[0]["start"], 5.9)
        self.assertEqual(result.segments[0]["end"], 7.1)
        self.assertEqual(result.duration, 10.0)
        self.assertEqual(result.text, "Speaker 0: która godzina")

    def test_vad_rejects_too_quiet_detected_speech(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        fake_model = SimpleNamespace(device="cpu", dtype=torch.float32)
        fake_model.generate = unittest.mock.Mock()
        fake_processor = unittest.mock.Mock()

        transcriber.model = fake_model
        transcriber.processor = fake_processor
        transcriber.vad_detector = SimpleNamespace(
            detect_speech=lambda *_args, **_kwargs: SimpleNamespace(
                has_speech=False,
                reason="speech_too_quiet",
                speech_segments=1,
                total_speech_ms=180,
                max_segment_ms=180,
                speech_rms=0.008,
                noise_rms=0.003,
                speech_to_noise_ratio=2.667,
            )
        )

        t = np.arange(3200, dtype=np.float32) / 16000.0
        audio = 0.008 * np.sin(2 * np.pi * 220.0 * t)
        result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        self.assertEqual(result.text, "")
        fake_model.generate.assert_not_called()

    def test_vad_rejects_speech_too_close_to_noise(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        fake_model = SimpleNamespace(device="cpu", dtype=torch.float32)
        fake_model.generate = unittest.mock.Mock()
        fake_processor = unittest.mock.Mock()

        transcriber.model = fake_model
        transcriber.processor = fake_processor
        transcriber.vad_detector = SimpleNamespace(
            detect_speech=lambda *_args, **_kwargs: SimpleNamespace(
                has_speech=False,
                reason="speech_too_close_to_noise",
                speech_segments=1,
                total_speech_ms=220,
                max_segment_ms=220,
                speech_rms=0.018,
                noise_rms=0.007,
                speech_to_noise_ratio=2.57,
            )
        )

        t = np.arange(3200, dtype=np.float32) / 16000.0
        audio = 0.018 * np.sin(2 * np.pi * 220.0 * t)
        result = transcriber.transcribe_pcm(audio, sample_rate=16000, language="pl")

        self.assertEqual(result.text, "")
        fake_model.generate.assert_not_called()


class SpeakerTextAndIdentificationTests(unittest.TestCase):
    def test_render_speaker_text_modes(self):
        from transcribe_wyoming.transcriber import render_speaker_text

        segments = [
            {"speaker": 0, "name": "Krzysztof", "start": 0.0, "end": 1.0, "text": "zgaś"},
            {"speaker": 0, "name": "Krzysztof", "start": 1.0, "end": 2.0, "text": "światło"},
            {"speaker": 1, "start": 2.0, "end": 3.0, "text": "dobranoc"},
        ]

        self.assertEqual(
            render_speaker_text(segments, mode="prefix"),
            "Krzysztof: zgaś światło\nSpeaker 1: dobranoc",
        )
        self.assertEqual(
            render_speaker_text(segments, mode="both"),
            "Krzysztof: zgaś światło\nSpeaker 1: dobranoc",
        )
        self.assertEqual(
            render_speaker_text(segments, mode="field"),
            "zgaś światło\ndobranoc",
        )

    def test_identify_speakers_concatenates_per_speaker(self):
        from transcribe_wyoming.speaker_id import SpeakerMatch

        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        received_clips = []

        class FakeRegistry:
            enabled = True

            def reload_if_changed(self):
                return False

            def has_profiles(self):
                return True

            def embed_batch(self, clips):
                received_clips.extend(clips)
                return [np.array([1.0, 0.0], dtype=np.float32), None]

            def match_embedding(self, embedding):
                if embedding is None:
                    return SpeakerMatch(None, 0.1)
                return SpeakerMatch("Krzysztof", 0.8)

            def adapt(self, name, embedding, score):
                return False

        transcriber.speaker_registry = FakeRegistry()
        # Two sub-0.4s segments for speaker 0 (individually too short for a
        # voiceprint) and one for speaker 1.
        segments = [
            {"speaker": 0, "start": 0.0, "end": 0.3, "text": "zgaś"},
            {"speaker": 1, "start": 0.4, "end": 0.6, "text": "co?"},
            {"speaker": 0, "start": 0.7, "end": 1.0, "text": "światło"},
        ]
        audio = np.arange(16000, dtype=np.float32)

        transcriber._identify_speakers(segments, audio, 16000)

        # One clip per speaker: speaker 0's clip concatenates both its segments.
        self.assertEqual(len(received_clips), 2)
        self.assertEqual(len(received_clips[0]), int(0.3 * 16000) + int(0.3 * 16000))
        self.assertEqual(len(received_clips[1]), int(0.2 * 16000))
        # The name lands on all of speaker 0's segments, none on speaker 1's.
        self.assertEqual(segments[0].get("name"), "Krzysztof")
        self.assertEqual(segments[2].get("name"), "Krzysztof")
        self.assertIsNone(segments[1].get("name"))

    def test_dominant_speaker_prefers_most_speech_time(self):
        segments = [
            {"speaker": 0, "name": "Krzysztof", "score": 0.8, "start": 0.0, "end": 5.0, "text": "a"},
            {"speaker": 1, "name": "Anna", "score": 0.7, "start": 5.0, "end": 6.0, "text": "b"},
        ]
        name, score = SpeechTranscriber._dominant_speaker(segments)
        self.assertEqual(name, "Krzysztof")
        self.assertEqual(score, 0.8)

    def test_dominant_speaker_unrecognized_returns_none(self):
        segments = [{"speaker": 0, "start": 0.0, "end": 5.0, "text": "a"}]
        name, score = SpeechTranscriber._dominant_speaker(segments)
        self.assertIsNone(name)
        self.assertIsNone(score)

    def test_save_pending_utterance_buffers_dominant_voice(self):
        import tempfile

        from transcribe_wyoming.pending import PendingStore

        with tempfile.TemporaryDirectory() as tmp:
            from transcribe_wyoming.speaker_id import SpeakerMatch

            transcriber = SpeechTranscriber()
            transcriber.whispercpp_reachable = lambda **_: True
            transcriber.pending_store = PendingStore(tmp)
            embedding = np.array([0.6, 0.8], dtype=np.float32)
            transcriber.speaker_registry = SimpleNamespace(
                enabled=True,
                embed=lambda _clip: embedding,
                nearest=lambda _embedding: SpeakerMatch("Anna", 0.31),
            )
            # Dominant speaker 0 (4s) vs speaker 1 (1s).
            segments = [
                {"speaker": 0, "start": 0.0, "end": 4.0, "text": "zgaś światło w salonie"},
                {"speaker": 1, "start": 4.0, "end": 5.0, "text": "ok"},
            ]
            audio = np.random.default_rng(7).normal(0, 0.05, 6 * 16000).astype(np.float32)

            utterance_id = transcriber._save_pending_utterance(
                segments, audio, 16000, "zgaś światło w salonie\nok"
            )

            self.assertIsNotNone(utterance_id)
            clips = transcriber.pending_store.list_clips()
            self.assertEqual(len(clips), 1)
            self.assertAlmostEqual(clips[0]["seconds"], 4.0, places=1)
            self.assertEqual(clips[0]["embedding"], embedding.tolist())
            # Closest (sub-threshold) profile is recorded for threshold tuning.
            self.assertEqual(clips[0]["best_match"], "Anna")
            self.assertEqual(clips[0]["best_score"], 0.31)

    def test_save_pending_utterance_never_raises(self):
        transcriber = SpeechTranscriber()
        transcriber.whispercpp_reachable = lambda **_: True
        transcriber.pending_store = SimpleNamespace(
            save=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
        )
        transcriber.speaker_registry = SimpleNamespace(enabled=True, embed=lambda _c: None)
        segments = [{"speaker": 0, "start": 0.0, "end": 2.0, "text": "x"}]
        audio = np.zeros(3 * 16000, dtype=np.float32)

        self.assertIsNone(
            transcriber._save_pending_utterance(segments, audio, 16000, "x")
        )


