import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from cohere_wyoming import audio as audio_module
from cohere_wyoming.transcriber import CohereTranscriber


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

    def test_load_model_falls_back_to_cpu_when_cuda_load_fails(self):
        transcriber = CohereTranscriber()

        class FakeModel:
            def __init__(self):
                self.moved_to = []
                self.eval_called = False

            def to(self, target_device):
                self.moved_to.append(str(target_device))
                if str(target_device) == "cuda:0":
                    raise torch.OutOfMemoryError("CUDA out of memory")
                return self

            def eval(self):
                self.eval_called = True

        class FakeTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, _token):
                return 5

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        fake_model = FakeModel()
        fake_processor = FakeProcessor()

        with patch.object(
            transcriber,
            "load_model_artifacts",
            return_value=(fake_processor, fake_model),
        ), patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.device_count", return_value=2
        ), patch(
            "torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 3090"
        ), patch("torch.cuda.empty_cache") as empty_cache:
            transcriber.load("fallback-model")

        self.assertEqual(fake_model.moved_to, ["cuda:0", "cpu"])
        self.assertTrue(fake_model.eval_called)
        self.assertIs(transcriber.model, fake_model)
        self.assertIs(transcriber.processor, fake_processor)
        self.assertEqual(str(transcriber.device), "cpu")
        empty_cache.assert_called_once()

    def test_load_model_artifacts_prefers_local_cache_before_network(self):
        transcriber = CohereTranscriber()
        call_log = []
        fake_model = object()
        fake_processor = object()

        def fake_processor_from_pretrained(model_name, **kwargs):
            call_log.append(("processor", kwargs.copy()))
            if kwargs.get("local_files_only"):
                raise OSError("missing local cache")
            return fake_processor

        def fake_model_from_pretrained(model_name, **kwargs):
            call_log.append(("model", kwargs.copy()))
            return fake_model

        with patch(
            "transformers.AutoProcessor.from_pretrained",
            side_effect=fake_processor_from_pretrained,
        ), patch(
            "transformers.AutoModelForSpeechSeq2Seq.from_pretrained",
            side_effect=fake_model_from_pretrained,
        ):
            processor, model = transcriber.load_model_artifacts("offline-first-model")

        dtype = transcriber._model_dtype()
        self.assertEqual(
            call_log,
            [
                ("processor", {"trust_remote_code": False, "local_files_only": True}),
                ("processor", {"trust_remote_code": False}),
                ("model", {"trust_remote_code": False, "dtype": dtype}),
            ],
        )
        self.assertIs(processor, fake_processor)
        self.assertIs(model, fake_model)

    def test_silent_audio_returns_empty_transcription_without_generate(self):
        transcriber = CohereTranscriber()
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
        transcriber = CohereTranscriber()
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
        transcriber = CohereTranscriber()
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
        transcriber = CohereTranscriber()

        class FakeModel:
            device = "cpu"
            dtype = torch.float32

            def generate(self, **_kwargs):
                return torch.tensor([[1, 2, 3]])

        class FakeTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, _token):
                return 5

            def decode(self, *_args, **_kwargs):
                return (
                    "<|diarize|><|spltoken0|><|t:0.0|> speech detected <|t:1.0|><|endoftext|>"
                )

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def __call__(self, *_args, **_kwargs):
                return FakeInputs(
                    input_features=torch.zeros(1, 4, 8),
                    attention_mask=torch.ones(1, 4),
                )

        transcriber.model = FakeModel()
        transcriber.processor = FakeProcessor()
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

        self.assertEqual(result.text, "Mówca 0: speech detected")
        self.assertEqual(len(result.segments), 1)
        self.assertEqual(result.segments[0]["speaker"], 0)
        self.assertEqual(result.segments[0]["text"], "speech detected")

    def test_vad_rejects_too_quiet_detected_speech(self):
        transcriber = CohereTranscriber()
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
        transcriber = CohereTranscriber()
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
        from cohere_wyoming.transcriber import render_speaker_text

        segments = [
            {"speaker": 0, "name": "Krzysztof", "start": 0.0, "end": 1.0, "text": "zgaś"},
            {"speaker": 0, "name": "Krzysztof", "start": 1.0, "end": 2.0, "text": "światło"},
            {"speaker": 1, "start": 2.0, "end": 3.0, "text": "dobranoc"},
        ]

        self.assertEqual(
            render_speaker_text(segments, mode="prefix"),
            "Krzysztof: zgaś światło\nMówca 1: dobranoc",
        )
        self.assertEqual(
            render_speaker_text(segments, mode="both"),
            "Krzysztof: zgaś światło\nMówca 1: dobranoc",
        )
        self.assertEqual(
            render_speaker_text(segments, mode="field"),
            "zgaś światło\ndobranoc",
        )

    def test_identify_speakers_concatenates_per_speaker(self):
        from cohere_wyoming.speaker_id import SpeakerMatch

        transcriber = CohereTranscriber()
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
        name, score = CohereTranscriber._dominant_speaker(segments)
        self.assertEqual(name, "Krzysztof")
        self.assertEqual(score, 0.8)

    def test_dominant_speaker_unrecognized_returns_none(self):
        segments = [{"speaker": 0, "start": 0.0, "end": 5.0, "text": "a"}]
        name, score = CohereTranscriber._dominant_speaker(segments)
        self.assertIsNone(name)
        self.assertIsNone(score)

    def test_save_pending_utterance_buffers_dominant_voice(self):
        import tempfile

        from cohere_wyoming.pending import PendingStore

        with tempfile.TemporaryDirectory() as tmp:
            transcriber = CohereTranscriber()
            transcriber.pending_store = PendingStore(tmp)
            embedding = np.array([0.6, 0.8], dtype=np.float32)
            transcriber.speaker_registry = SimpleNamespace(
                enabled=True,
                embed=lambda _clip: embedding,
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

    def test_save_pending_utterance_never_raises(self):
        transcriber = CohereTranscriber()
        transcriber.pending_store = SimpleNamespace(
            save=lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
        )
        transcriber.speaker_registry = SimpleNamespace(enabled=True, embed=lambda _c: None)
        segments = [{"speaker": 0, "start": 0.0, "end": 2.0, "text": "x"}]
        audio = np.zeros(3 * 16000, dtype=np.float32)

        self.assertIsNone(
            transcriber._save_pending_utterance(segments, audio, 16000, "x")
        )


class MergeDiarizedWindowsTests(unittest.TestCase):
    """Speaker indices restart per window; merging must remap them globally."""

    @staticmethod
    def _window(speaker: int, start: float, end: float, text: str) -> list[dict]:
        return [{"speaker": speaker, "start": start, "end": end, "text": text}]

    @staticmethod
    def _registry(embeddings_per_call: list[list[np.ndarray]]):
        calls = iter(embeddings_per_call)
        return SimpleNamespace(enabled=True, embed_batch=lambda clips: next(calls))

    def test_different_voices_in_two_windows_get_distinct_speakers(self):
        transcriber = CohereTranscriber()
        transcriber.speaker_registry = self._registry(
            [
                [np.array([1.0, 0.0], dtype=np.float32)],
                [np.array([0.0, 1.0], dtype=np.float32)],
            ]
        )
        windows = [
            self._window(0, 0.0, 10.0, "alice speaking"),
            self._window(0, 35.0, 45.0, "bob speaking"),
        ]

        merged = transcriber._merge_diarized_windows(
            windows, np.zeros(70 * 16000, dtype=np.float32), 16000
        )

        self.assertEqual([segment["speaker"] for segment in merged], [0, 1])

    def test_same_voice_in_two_windows_keeps_one_speaker(self):
        transcriber = CohereTranscriber()
        voice = np.array([0.6, 0.8], dtype=np.float32)
        transcriber.speaker_registry = self._registry([[voice], [voice.copy()]])
        windows = [
            self._window(0, 0.0, 10.0, "alice part one"),
            self._window(0, 35.0, 45.0, "alice part two"),
        ]

        merged = transcriber._merge_diarized_windows(
            windows, np.zeros(70 * 16000, dtype=np.float32), 16000
        )

        self.assertEqual([segment["speaker"] for segment in merged], [0, 0])

    def test_without_registry_raw_indices_are_preserved(self):
        transcriber = CohereTranscriber()
        transcriber.speaker_registry = None
        windows = [
            self._window(0, 0.0, 10.0, "first window"),
            self._window(0, 35.0, 45.0, "second window"),
        ]

        merged = transcriber._merge_diarized_windows(
            windows, np.zeros(70 * 16000, dtype=np.float32), 16000
        )

        # Without voiceprints the model's own indices are kept (pre-remap
        # behavior): one person over multiple windows stays one speaker.
        self.assertEqual([segment["speaker"] for segment in merged], [0, 0])

    def test_embedding_failure_preserves_raw_indices(self):
        transcriber = CohereTranscriber()

        def broken_embed(_clips):
            raise RuntimeError("no speechbrain")

        transcriber.speaker_registry = SimpleNamespace(enabled=True, embed_batch=broken_embed)
        windows = [
            self._window(0, 0.0, 10.0, "first window"),
            self._window(0, 35.0, 45.0, "second window"),
        ]

        merged = transcriber._merge_diarized_windows(
            windows, np.zeros(70 * 16000, dtype=np.float32), 16000
        )

        self.assertEqual([segment["speaker"] for segment in merged], [0, 0])

    def test_two_speakers_per_window_are_matched_pairwise(self):
        transcriber = CohereTranscriber()
        alice = np.array([1.0, 0.0], dtype=np.float32)
        bob = np.array([0.0, 1.0], dtype=np.float32)
        # Window 2 hears them in reverse local order (bob=0, alice=1).
        transcriber.speaker_registry = self._registry([[alice, bob], [bob, alice]])
        windows = [
            self._window(0, 0.0, 5.0, "alice") + self._window(1, 5.0, 10.0, "bob"),
            self._window(0, 35.0, 40.0, "bob again") + self._window(1, 40.0, 45.0, "alice again"),
        ]

        merged = transcriber._merge_diarized_windows(
            windows, np.zeros(70 * 16000, dtype=np.float32), 16000
        )

        self.assertEqual([segment["speaker"] for segment in merged], [0, 1, 1, 0])


class TranscribeWindowTruncationTests(unittest.TestCase):
    def test_truncated_window_is_split_and_retried(self):
        transcriber = CohereTranscriber()
        calls = []

        def fake_generate(chunk, _sample_rate, _language, _temperature):
            calls.append(len(chunk))
            if len(chunk) > 16000 * 20:
                return "<|diarize|>ignored", True
            return (
                "<|diarize|><|spltoken0|><|t:0.0|> part <|t:1.0|><|endoftext|>",
                False,
            )

        with patch.object(transcriber, "_generate_diarized", side_effect=fake_generate):
            windows = transcriber._transcribe_window(
                np.zeros(30 * 16000, dtype=np.float32), 60.0, 16000, "pl", 0.0
            )

        # One truncated 30s pass retried as two 15s passes.
        self.assertEqual(calls, [30 * 16000, 15 * 16000, 15 * 16000])
        self.assertEqual(len(windows), 2)
        # Offsets stay on the global timeline.
        self.assertEqual(windows[0][0]["start"], 60.0)
        self.assertEqual(windows[1][0]["start"], 75.0)

    def test_truncated_short_window_is_kept_with_warning(self):
        transcriber = CohereTranscriber()

        def fake_generate(_chunk, _sample_rate, _language, _temperature):
            return (
                "<|diarize|><|spltoken0|><|t:0.0|> dense speech <|t:1.0|>",
                True,
            )

        with patch.object(transcriber, "_generate_diarized", side_effect=fake_generate):
            with self.assertLogs("cohere-wyoming.transcriber", level="WARNING") as logs:
                windows = transcriber._transcribe_window(
                    np.zeros(10 * 16000, dtype=np.float32), 0.0, 16000, "pl", 0.0
                )

        self.assertEqual(len(windows), 1)
        self.assertTrue(any("token cap" in message for message in logs.output))


class PromptTokenTests(unittest.TestCase):
    def _transcriber(self, known: dict) -> CohereTranscriber:
        transcriber = CohereTranscriber(default_language="pl")

        class Tok:
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return known.get(token, 0)  # 0 == unk == "missing"

        class Proc:
            tokenizer = Tok()

        transcriber.processor = Proc()
        transcriber.model = SimpleNamespace(device="cpu")
        return transcriber

    @staticmethod
    def _structural(**extra) -> dict:
        from cohere_wyoming.transcriber import DIARIZE_PROMPT_TEMPLATE

        known = {tpl: 5 for tpl in DIARIZE_PROMPT_TEMPLATE if "{lang}" not in tpl}
        known.update(extra)
        return known

    def test_validate_structural_tokens_raises_when_missing(self):
        known = self._structural()
        del known["<|diarize|>"]
        with self.assertRaises(RuntimeError):
            self._transcriber(known)._validate_structural_prompt_tokens()

    def test_validate_structural_tokens_passes_when_present(self):
        self._transcriber(self._structural())._validate_structural_prompt_tokens()

    def test_language_token_falls_back_to_default_then_en(self):
        transcriber = self._transcriber(self._structural(**{"<|en|>": 62}))
        transcriber.default_language = "de"  # also absent -> should reach <|en|>
        self.assertEqual(transcriber._language_token_id("pl"), 62)

    def test_build_prompt_ids_uses_requested_language_when_present(self):
        transcriber = self._transcriber(self._structural(**{"<|pl|>": 99}))
        ids = transcriber._build_prompt_ids("pl").tolist()[0]
        self.assertEqual(ids.count(99), 2)  # <|{lang}|> appears twice in the prompt


if __name__ == "__main__":
    unittest.main()
