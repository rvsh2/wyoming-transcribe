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

        fake_model = FakeModel()
        fake_processor = object()

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


if __name__ == "__main__":
    unittest.main()
