import asyncio
import io
import json
import sys
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np
import torch

import server


def make_wav_bytes(
    duration_s: float = 0.1,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    frames = int(duration_s * sample_rate)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    signal = 0.25 * np.sin(2 * np.pi * 440.0 * t)
    pcm = np.clip(signal * 32767.0, -32768, 32767).astype(np.int16)
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    audio = io.BytesIO()
    with wave.open(audio, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return audio.getvalue()


class FakeUploadFile:
    def __init__(self, content: bytes, filename: str = "test.wav") -> None:
        self._content = content
        self.filename = filename

    async def read(self) -> bytes:
        return self._content


def make_upload_file(content: bytes, filename: str = "test.wav") -> FakeUploadFile:
    return FakeUploadFile(content=content, filename=filename)


class CohereTranscribeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_bytes = make_wav_bytes()
        self.service_model = patch.object(server.service, "model", object())
        self.service_processor = patch.object(server.service, "processor", object())
        self.service_vad = patch.object(
            server.service.vad_detector,
            "detect_speech",
            return_value=SimpleNamespace(
                has_speech=True,
                reason="test_override",
                speech_segments=1,
                total_speech_ms=100,
                max_segment_ms=100,
                speech_rms=0.03,
                noise_rms=0.002,
                speech_to_noise_ratio=15.0,
            ),
        )
        self.service_model.start()
        self.service_processor.start()
        self.service_vad.start()

    def tearDown(self) -> None:
        self.service_vad.stop()
        self.service_processor.stop()
        self.service_model.stop()

    def run_async(self, coro):
        return asyncio.run(coro)

    def decode_json_response(self, response) -> dict:
        return json.loads(response.body.decode("utf-8"))

    def call_inference(self, **overrides):
        params = {
            "file": make_upload_file(self.audio_bytes),
            "temperature": 0.0,
            "temperature_inc": 0.2,
            "response_format": "json",
            "language": None,
            "encode": True,
            "no_timestamps": False,
            "prompt": None,
            "translate": False,
        }
        params.update(overrides)
        return self.run_async(server.inference(**params))

    def call_openai_transcriptions(self, **overrides):
        params = {
            "file": make_upload_file(self.audio_bytes),
            "model_name": None,
            "language": None,
            "response_format": "json",
            "temperature": 0.0,
            "prompt": None,
        }
        params.update(overrides)
        return self.run_async(server.openai_transcriptions(**params))

    def test_index_page_includes_compatibility_notes(self):
        response = self.run_async(server.index())
        self.assertIn("Compatibility Notes", response)
        self.assertIn("/inference", response)
        self.assertIn("/health", response)
        self.assertIn("Quick Transcription", response)
        self.assertIn('id="transcribe-form"', response)
        self.assertIn("Language is a hint for the model", response)
        self.assertIn("Record Voice", response)
        self.assertIn("source-summary", response)

    def test_health_endpoint_reports_ready_state(self):
        with patch.object(server.service, "model_name", "CohereLabs/test-model"), patch.object(
            server.service, "device", "cuda:0"
        ), patch.object(server.service, "backend", "native"):
            response = self.run_async(server.health())

        payload = self.decode_json_response(response)
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["ready"])
        self.assertEqual(payload["model"], "CohereLabs/test-model")
        self.assertEqual(payload["device"], "cuda:0")
        self.assertEqual(payload["backend"], "native")

    def test_inference_returns_json_response(self):
        with patch.object(server.service, "transcribe_pcm", return_value=SimpleNamespace(asdict=lambda: {
            "text": "hello world",
            "language": "en",
            "duration": 0.1,
            "processing_time": 0.01,
        })):
            response = self.call_inference()

        self.assertEqual(self.decode_json_response(response), {"text": "hello world"})

    def test_inference_supports_text_verbose_json_srt_and_vtt(self):
        mocked_result = SimpleNamespace(
            asdict=lambda: {
                "text": "hello world",
                "language": "en",
                "duration": 1.25,
                "processing_time": 0.02,
            }
        )

        with patch.object(server.service, "transcribe_pcm", return_value=mocked_result):
            text_response = self.call_inference(response_format="text")
            verbose_response = self.call_inference(response_format="verbose_json")
            srt_response = self.call_inference(response_format="srt")
            vtt_response = self.call_inference(response_format="vtt")

        self.assertEqual(text_response.body.decode("utf-8"), "hello world\n")
        self.assertEqual(
            self.decode_json_response(verbose_response)["segments"][0]["end"], 1.25
        )
        self.assertIn("00:00:01,250", srt_response.body.decode("utf-8"))
        self.assertTrue(vtt_response.body.decode("utf-8").startswith("WEBVTT"))

    def test_index_page_marks_asr_available_when_model_loaded(self):
        response = self.run_async(server.index())
        self.assertIn("const ASR_AVAILABLE = true;", response)
        self.assertNotIn("__ASR_AVAILABLE__", response)

    def test_index_page_marks_asr_unavailable_without_model(self):
        with patch.object(server.service, "model", None):
            response = self.run_async(server.index())
        self.assertIn("const ASR_AVAILABLE = false;", response)

    def test_srt_and_vtt_render_one_cue_per_diarized_segment(self):
        result = {
            "text": "alice: hej\nMówca 1: cześć",
            "language": "pl",
            "duration": 10.0,
            "processing_time": 0.1,
            "segments": [
                {"speaker": 0, "name": "alice", "start": 0.0, "end": 2.5, "text": "hej"},
                {"speaker": 1, "start": 3.0, "end": 4.0, "text": "cześć"},
            ],
        }

        srt = server.format_subtitles(result, srt=True)
        self.assertEqual(
            srt,
            "1\n00:00:00,000 --> 00:00:02,500\nalice: hej\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nMówca 1: cześć\n\n",
        )

        vtt = server.format_subtitles(result, srt=False)
        self.assertTrue(vtt.startswith("WEBVTT\n\n"))
        self.assertIn("00:00:00.000 --> 00:00:02.500\nalice: hej", vtt)
        self.assertIn("00:00:03.000 --> 00:00:04.000\nMówca 1: cześć", vtt)

    def test_subtitles_fall_back_to_single_cue_without_segments(self):
        result = {"text": "hello world", "language": "en", "duration": 1.25}
        srt = server.format_subtitles(result, srt=True)
        self.assertEqual(srt, "1\n00:00:00,000 --> 00:00:01,250\nhello world\n\n")

    def test_settings_endpoints_roundtrip(self):
        import tempfile
        from pathlib import Path

        from cohere_wyoming.settings import SettingsStore

        with tempfile.TemporaryDirectory() as tmp:
            store = SettingsStore(Path(tmp) / ".settings.json")
            with patch.object(server.service, "settings_store", store):
                payload = self.decode_json_response(self.run_async(server.get_settings()))
                self.assertEqual(payload["speaker_text_mode"], "prefix")
                self.assertEqual(payload["speaker_text_modes"], ["prefix", "field", "both"])

                response = self.run_async(server.update_settings(speaker_text_mode="field"))
                self.assertEqual(self.decode_json_response(response)["speaker_text_mode"], "field")

                payload = self.decode_json_response(self.run_async(server.get_settings()))
                self.assertEqual(payload["speaker_text_mode"], "field")

                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(server.update_settings(speaker_text_mode="bogus"))
                self.assertEqual(ctx.exception.status_code, 400)

    def test_api_token_middleware(self):
        from fastapi.testclient import TestClient

        with patch.object(server, "API_TOKEN", "sekret"):
            client = TestClient(server.app)
            # Open paths stay reachable for the healthcheck and info page.
            self.assertEqual(client.get("/health").status_code, 200)
            # Protected path without/with the token.
            self.assertEqual(client.get("/settings").status_code, 401)
            self.assertEqual(
                client.get("/settings", headers={"X-API-Token": "zly"}).status_code, 401
            )
            self.assertEqual(
                client.get("/settings", headers={"X-API-Token": "sekret"}).status_code, 200
            )
            self.assertEqual(
                client.get(
                    "/settings", headers={"Authorization": "Bearer sekret"}
                ).status_code,
                200,
            )

    def test_verbose_json_includes_dominant_speaker(self):
        mocked_result = SimpleNamespace(
            asdict=lambda: {
                "text": "Krzysztof: zgaś światło",
                "language": "pl",
                "duration": 2.0,
                "processing_time": 0.1,
                "speaker": "Krzysztof",
                "speaker_score": 0.82,
                "segments": [
                    {"speaker": 0, "name": "Krzysztof", "start": 0.0, "end": 2.0, "text": "zgaś światło"}
                ],
            }
        )
        with patch.object(server.service, "transcribe_pcm", return_value=mocked_result):
            response = self.call_inference(response_format="verbose_json")

        payload = self.decode_json_response(response)
        self.assertEqual(payload["speaker"], "Krzysztof")
        self.assertEqual(payload["speaker_score"], 0.82)

    def test_pending_claim_and_role_endpoints(self):
        import tempfile

        import numpy as np

        from cohere_wyoming.enrollment import EnrollmentStore
        from cohere_wyoming.pending import PendingStore

        with tempfile.TemporaryDirectory() as tmp:
            pending = PendingStore(tmp)
            enrollment = EnrollmentStore(tmp)
            voice = np.array([1.0, 0.0], dtype=np.float32)
            other = np.array([0.0, 1.0], dtype=np.float32)
            t = np.arange(32000, dtype=np.float32) / 16000
            clip = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            id_a = pending.save(clip, 16000, text="pierwsze", embedding=voice)
            id_b = pending.save(clip, 16000, text="drugie", embedding=voice)
            id_c = pending.save(clip, 16000, text="ktoś inny", embedding=other)

            with patch.object(server, "pending_store", lambda: pending), patch.object(
                server, "enrollment_store", lambda: enrollment
            ):
                payload = self.decode_json_response(self.run_async(server.list_pending()))
                self.assertEqual(payload["count"], 3)
                self.assertEqual(len(payload["clusters"]), 2)
                self.assertNotIn("embedding", payload["clusters"][0]["clips"][0])

                # Claiming one clip pulls in its whole voice cluster.
                response = self.run_async(
                    server.claim_pending("Krzysztof", id_a, include_cluster=True)
                )
                claim = self.decode_json_response(response)
                self.assertEqual(set(claim["claimed"]), {id_a, id_b})
                self.assertEqual(len(claim["samples"]), 2)

                speakers = enrollment.list_speakers()
                self.assertEqual(speakers[0]["name"], "Krzysztof")
                self.assertEqual(len(speakers[0]["samples"]), 2)
                self.assertEqual(len(pending.list_clips()), 1)

                # Role endpoint.
                response = self.run_async(server.set_speaker_role("Krzysztof", role="admin"))
                self.assertEqual(self.decode_json_response(response)["role"], "admin")
                self.assertEqual(enrollment.list_speakers()[0]["role"], "admin")

                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(server.set_speaker_role("Krzysztof", role="root"))
                self.assertEqual(ctx.exception.status_code, 400)
                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(server.claim_pending("X", "utt-1-00000000", include_cluster=False))
                self.assertEqual(ctx.exception.status_code, 404)

                # The other voice stays pending for manual verification.
                payload = self.decode_json_response(self.run_async(server.list_pending()))
                self.assertEqual(payload["count"], 1)
                self.assertEqual(payload["clusters"][0]["clips"][0]["id"], id_c)

    def test_claim_latest_claims_newest_voice_cluster(self):
        import tempfile

        import numpy as np

        from cohere_wyoming.enrollment import EnrollmentStore
        from cohere_wyoming.pending import PendingStore

        with tempfile.TemporaryDirectory() as tmp:
            pending = PendingStore(tmp)
            enrollment = EnrollmentStore(tmp)
            anna = np.array([1.0, 0.0], dtype=np.float32)
            bob = np.array([0.0, 1.0], dtype=np.float32)
            t = np.arange(32000, dtype=np.float32) / 16000
            clip = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            id_bob = pending.save(clip, 16000, text="bob mówi", embedding=bob)
            id_anna1 = pending.save(clip, 16000, text="zgaś światło", embedding=anna)
            id_anna2 = pending.save(clip, 16000, text="jestem Anna", embedding=anna)

            with patch.object(server, "pending_store", lambda: pending), patch.object(
                server, "enrollment_store", lambda: enrollment
            ):
                # The newest clip (Anna's answer) anchors the claim; Bob's
                # earlier interjection stays untouched.
                response = self.run_async(
                    server.claim_latest("Anna", include_cluster=True, max_age_seconds=300.0, anchor_utterance_id=None)
                )
                claim = self.decode_json_response(response)
                self.assertEqual(set(claim["claimed"]), {id_anna1, id_anna2})

                remaining = [clip["id"] for clip in pending.list_clips()]
                self.assertEqual(remaining, [id_bob])

                # Stale newest clip -> 409 (the answer was not buffered).
                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(
                        server.claim_latest("Ktoś", include_cluster=True, max_age_seconds=0.0, anchor_utterance_id=None)
                    )
                self.assertEqual(ctx.exception.status_code, 409)

                pending.delete(id_bob)
                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(
                        server.claim_latest("Ktoś", include_cluster=True, max_age_seconds=300.0, anchor_utterance_id=None)
                    )
                self.assertEqual(ctx.exception.status_code, 404)

    def test_claim_latest_with_anchor_survives_interjection(self):
        import tempfile

        import numpy as np

        from cohere_wyoming.enrollment import EnrollmentStore
        from cohere_wyoming.pending import PendingStore

        with tempfile.TemporaryDirectory() as tmp:
            pending = PendingStore(tmp)
            enrollment = EnrollmentStore(tmp)
            anna = np.array([1.0, 0.0], dtype=np.float32)
            tv = np.array([0.0, 1.0], dtype=np.float32)
            t = np.arange(32000, dtype=np.float32) / 16000
            clip = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            id_anna = pending.save(clip, 16000, text="jestem Anna", embedding=anna)
            id_tv = pending.save(clip, 16000, text="reklama w tv", embedding=tv)

            with patch.object(server, "pending_store", lambda: pending), patch.object(
                server, "enrollment_store", lambda: enrollment
            ):
                # The TV clip is newer, but the anchor pins Anna's cluster.
                response = self.run_async(
                    server.claim_latest(
                        "Anna",
                        include_cluster=True,
                        max_age_seconds=300.0,
                        anchor_utterance_id=id_anna,
                    )
                )
                claim = self.decode_json_response(response)
                self.assertEqual(claim["claimed"], [id_anna])
                self.assertEqual([c["id"] for c in pending.list_clips()], [id_tv])

                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(
                        server.claim_latest(
                            "Anna",
                            include_cluster=True,
                            max_age_seconds=300.0,
                            anchor_utterance_id="utt-1-00000000",
                        )
                    )
                self.assertEqual(ctx.exception.status_code, 404)

    def test_claim_is_all_or_nothing_when_clip_vanishes(self):
        import tempfile

        import numpy as np

        from cohere_wyoming.enrollment import EnrollmentStore
        from cohere_wyoming.pending import PendingStore

        with tempfile.TemporaryDirectory() as tmp:
            pending = PendingStore(tmp)
            enrollment = EnrollmentStore(tmp)
            voice = np.array([1.0, 0.0], dtype=np.float32)
            t = np.arange(32000, dtype=np.float32) / 16000
            clip = (0.1 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            id_a = pending.save(clip, 16000, text="a", embedding=voice)
            id_b = pending.save(clip, 16000, text="b", embedding=voice)
            # Simulate the ring buffer pruning clip B between cluster
            # resolution and audio read (the mid-claim race).
            pending.audio_path(id_b).unlink()

            with patch.object(server, "pending_store", lambda: pending), patch.object(
                server, "enrollment_store", lambda: enrollment
            ), patch.object(pending, "cluster_members", return_value=[id_a, id_b]):
                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(server.claim_pending("Anna", id_a, include_cluster=True))
                self.assertEqual(ctx.exception.status_code, 404)

            # Nothing was enrolled and the intact clip is still pending.
            self.assertEqual(enrollment.list_speakers(), [])
            self.assertIn(id_a, [c["id"] for c in pending.list_clips()])

    def test_api_token_middleware_handles_non_ascii(self):
        from fastapi.testclient import TestClient

        with patch.object(server, "API_TOKEN", "sekret"):
            client = TestClient(server.app)
            # Non-ASCII client token (raw bytes on the wire) must yield 401,
            # not a 500 TypeError from secrets.compare_digest.
            response = client.get(
                "/settings", headers={b"X-API-Token": "zażółć".encode("utf-8")}
            )
            self.assertEqual(response.status_code, 401)

        # A non-ASCII server-side token can never match (headers are latin-1
        # on the wire) but must degrade to 401s, not 500s.
        with patch.object(server, "API_TOKEN", "zażółć-gęślą"):
            client = TestClient(server.app)
            self.assertEqual(client.get("/settings").status_code, 401)

    def test_health_reports_asr_ready(self):
        # Model "loaded" in this process (setUp patches it) -> asr_ready True
        # without probing any port.
        payload = self.decode_json_response(self.run_async(server.health()))
        self.assertTrue(payload["asr_ready"])

        with patch.object(server.service, "model", None), patch.object(
            server, "WYOMING_PROBE_PORT", 1
        ):
            payload = self.decode_json_response(self.run_async(server.health()))
        self.assertFalse(payload["asr_ready"])
        self.assertFalse(payload["ready"])

    def test_export_import_roundtrip(self):
        import tarfile
        import tempfile
        from pathlib import Path

        from cohere_wyoming.enrollment import EnrollmentStore

        with tempfile.TemporaryDirectory() as source_dir, tempfile.TemporaryDirectory() as target_dir:
            store = EnrollmentStore(source_dir)
            store.create_speaker("Krzysztof")
            store.set_role("Krzysztof", "admin")
            (Path(source_dir) / "Krzysztof" / "sample-1.wav").write_bytes(b"RIFFfake")
            (Path(source_dir) / ".pending").mkdir()
            (Path(source_dir) / ".pending" / "utt-1-aabbccdd.wav").write_bytes(b"x")
            (Path(source_dir) / ".history.jsonl").write_text("{}\n")

            with patch.object(server.service.speaker_registry, "enrollment_dir", Path(source_dir)):
                response = self.run_async(server.export_enrollment())
            archive = bytes(response.body)
            names = tarfile.open(fileobj=io.BytesIO(archive)).getnames()
            self.assertIn("Krzysztof/sample-1.wav", names)
            self.assertIn("Krzysztof/.meta.json", names)
            # Transient pending clips and the operational log stay out of backups
            # (restoring an old history over a newer one would lose entries).
            self.assertFalse(any(name.startswith(".pending") for name in names))
            self.assertNotIn(".history.jsonl", names)

            with patch.object(server.service.speaker_registry, "enrollment_dir", Path(target_dir)):
                result = self.run_async(server.import_enrollment(make_upload_file(archive, "backup.tar.gz")))
            self.assertEqual(self.decode_json_response(result)["status"], "ok")
            restored = EnrollmentStore(target_dir).list_speakers()
            self.assertEqual(restored[0]["name"], "Krzysztof")
            self.assertEqual(restored[0]["role"], "admin")
            self.assertEqual(len(restored[0]["samples"]), 1)

    def test_import_rejects_unsafe_archive(self):
        import tarfile
        import tempfile
        from pathlib import Path

        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            payload = io.BytesIO(b"evil")
            info = tarfile.TarInfo("../../evil.txt")
            info.size = 4
            archive.addfile(info, payload)

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(server.service.speaker_registry, "enrollment_dir", Path(tmp)):
                with self.assertRaises(HTTPException) as ctx:
                    self.run_async(server.import_enrollment(make_upload_file(buffer.getvalue(), "backup.tar.gz")))
        self.assertEqual(ctx.exception.status_code, 400)

    def test_history_endpoint_returns_entries(self):
        import tempfile
        from pathlib import Path

        from cohere_wyoming.history import RecognitionLog

        with tempfile.TemporaryDirectory() as tmp:
            RecognitionLog(tmp).append(text="zgaś światło", language="pl", duration=2.0, speaker="Krzysztof")
            with patch.object(server.service.speaker_registry, "enrollment_dir", Path(tmp)):
                payload = self.decode_json_response(self.run_async(server.recognition_history(limit=10)))
        self.assertEqual(len(payload["entries"]), 1)
        self.assertEqual(payload["entries"][0]["speaker"], "Krzysztof")

    def test_openai_endpoint_returns_verbose_json(self):
        mocked_result = SimpleNamespace(
            asdict=lambda: {
                "text": "openai shape",
                "language": "pl",
                "duration": 0.2,
                "processing_time": 0.03,
            }
        )
        with patch.object(server.service, "transcribe_pcm", return_value=mocked_result):
            response = self.call_openai_transcriptions(
                model_name="ignored-model",
                response_format="verbose_json",
            )

        payload = self.decode_json_response(response)
        self.assertEqual(payload["text"], "openai shape")
        self.assertEqual(payload["segments"][0]["start"], 0.0)

    def test_load_endpoint_returns_ok_on_success(self):
        with patch.object(server.service, "load") as load_model:
            response = self.run_async(server.load(model_path="CohereLabs/mock-model"))

        self.assertEqual(
            self.decode_json_response(response),
            {"status": "ok", "model": "CohereLabs/mock-model"},
        )
        load_model.assert_called_once()

    def test_empty_upload_returns_400(self):
        with self.assertRaises(HTTPException) as ctx:
            self.run_async(
                server.inference(
                    file=make_upload_file(b"", "empty.wav"),
                    temperature=0.0,
                    temperature_inc=0.2,
                    response_format="json",
                    language=None,
                    encode=True,
                    no_timestamps=False,
                    prompt=None,
                    translate=False,
                )
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Empty audio file")

    def test_invalid_audio_returns_400(self):
        with patch.object(server, "read_audio_to_numpy", side_effect=ValueError("bad audio")):
            with self.assertRaises(HTTPException) as ctx:
                self.call_inference()

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "bad audio")

    def test_unsupported_language_falls_back_to_default_language(self):
        with patch.object(server.service, "default_language", "en"), patch.object(
            server.service, "transcribe_pcm", side_effect=lambda audio, sample_rate, language, temperature: SimpleNamespace(
                asdict=lambda: {
                    "text": f"lang={language}",
                    "language": language,
                    "duration": 0.1,
                    "processing_time": 0.01,
                }
            )
        ):
            response = self.call_inference(language="xx")

        self.assertEqual(self.decode_json_response(response)["text"], "lang=en")

    def test_auto_language_uses_server_default_language(self):
        with patch.object(server.service, "default_language", "pl"), patch.object(
            server.service, "transcribe_pcm", side_effect=lambda audio, sample_rate, language, temperature: SimpleNamespace(
                asdict=lambda: {
                    "text": f"lang={language}",
                    "language": language,
                    "duration": 0.1,
                    "processing_time": 0.01,
                }
            )
        ):
            response = self.call_inference(language="auto")

        self.assertEqual(self.decode_json_response(response)["text"], "lang=pl")

    def test_missing_model_returns_503(self):
        with patch.object(server.service, "model", None), patch.object(server.service, "processor", None):
            with self.assertRaises(HTTPException) as ctx:
                self.call_inference()

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail, "Model not loaded")

    def test_transcription_failure_returns_500(self):
        with patch.object(server.service, "transcribe_pcm", side_effect=RuntimeError("boom")):
            with self.assertRaises(HTTPException) as ctx:
                self.call_inference()

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Transcription failed: boom", ctx.exception.detail)

    def test_load_failure_returns_500(self):
        with patch.object(server.service, "load", side_effect=RuntimeError("cannot load")):
            with self.assertRaises(HTTPException) as ctx:
                self.run_async(server.load(model_path="broken-model"))

        self.assertEqual(ctx.exception.status_code, 500)
        self.assertIn("Failed to load model: cannot load", ctx.exception.detail)

    def test_compatibility_only_parameters_do_not_break_inference(self):
        mocked_result = SimpleNamespace(
            asdict=lambda: {
                "text": "compat ok",
                "language": "en",
                "duration": 0.1,
                "processing_time": 0.01,
            }
        )
        with patch.object(server.service, "transcribe_pcm", return_value=mocked_result):
            response = self.call_inference(
                temperature_inc=0.5,
                prompt="hello",
                encode=False,
                no_timestamps=True,
                translate=True,
            )

        self.assertEqual(self.decode_json_response(response)["text"], "compat ok")

    def test_audio_preprocessing_resamples_mono_and_stereo_wav(self):
        stereo_44k = make_wav_bytes(sample_rate=44100, channels=2)
        mono_22k = make_wav_bytes(sample_rate=22050, channels=1)

        stereo_audio, stereo_sr = server.read_audio_to_numpy(stereo_44k, "stereo.wav")
        mono_audio, mono_sr = server.read_audio_to_numpy(mono_22k, "mono.wav")

        self.assertEqual(stereo_sr, 16000)
        self.assertEqual(mono_sr, 16000)
        self.assertEqual(stereo_audio.ndim, 1)
        self.assertEqual(mono_audio.ndim, 1)
        self.assertGreater(len(stereo_audio), 0)
        self.assertGreater(len(mono_audio), 0)

    def test_temperature_controls_generate_sampling(self):
        class FakeBatch(dict):
            def to(self, device, dtype=None):
                return self

        fake_inputs = FakeBatch(
            {
                "audio_chunk_index": [0],
                "input_features": torch.zeros(1, 4, 8),
                "attention_mask": torch.ones(1, 4),
            }
        )

        class FakeModel:
            device = "cpu"
            dtype = torch.float32

            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return torch.tensor([[1, 2, 3]])

        class FakeTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, _token):
                return 5

            def decode(self, *args, **kwargs):
                return "<|diarize|><|spltoken0|><|t:0.0|> decoded <|t:1.0|><|endoftext|>"

        class FakeProcessor:
            tokenizer = FakeTokenizer()

            def __call__(self, *args, **kwargs):
                return fake_inputs

        fake_model = FakeModel()
        fake_processor = FakeProcessor()

        with patch.object(server.service, "model", fake_model), patch.object(
            server.service, "processor", fake_processor
        ):
            audio = np.full(1600, 0.2, dtype=np.float32)
            server.service.transcribe_pcm(audio, sample_rate=16000, language="pl", temperature=0.0)
            server.service.transcribe_pcm(audio, sample_rate=16000, language="pl", temperature=0.7)

        greedy_call, sampling_call = fake_model.calls
        self.assertFalse(greedy_call["do_sample"])
        self.assertNotIn("temperature", greedy_call)
        self.assertTrue(sampling_call["do_sample"])
        self.assertEqual(sampling_call["temperature"], 0.7)

    def test_build_segments_exposes_name_and_clamps_end(self):
        result = {
            "text": "x",
            "duration": 5.0,
            "segments": [
                {"speaker": 1, "name": "Anna", "score": 0.7, "start": 3.7, "end": 0.4, "text": "hi"}
            ],
        }
        segment = server.build_segments(result)[0]
        self.assertEqual(segment["name"], "Anna")
        self.assertEqual(segment["score"], 0.7)
        self.assertGreaterEqual(segment["end"], segment["start"])

    def test_silent_audio_returns_empty_transcription_without_model_call(self):
        silent_audio = np.zeros(16000, dtype=np.float32)

        with patch.object(server, "transcribe_audio") as transcribe_audio:
            result = server.run_transcription_request(
                audio_data=silent_audio,
                sr=16000,
                language="pl",
                temperature=0.0,
            )

        self.assertEqual(result["text"], "")
        self.assertEqual(result["language"], "pl")
        self.assertEqual(result["duration"], 1.0)
        self.assertEqual(result["processing_time"], 0.0)
        transcribe_audio.assert_not_called()

    def test_cli_no_longer_exposes_trust_remote_code_flags(self):
        argv = ["server.py"]
        with patch.object(sys, "argv", argv):
            args = server.parse_args()

        self.assertFalse(hasattr(args, "trust_remote_code"))
        self.assertFalse(hasattr(args, "no_trust_remote_code"))

    def test_load_model_only_swaps_globals_after_successful_load(self):
        old_model = object()
        old_processor = object()
        old_device = object()

        with patch.object(server.service, "model", old_model), patch.object(
            server.service, "processor", old_processor
        ), patch.object(server.service, "device", old_device), patch(
            "transformers.AutoProcessor.from_pretrained", return_value=object()
        ), patch(
            "transformers.AutoModelForSpeechSeq2Seq.from_pretrained",
            side_effect=RuntimeError("load failed"),
        ):
            with self.assertRaises(RuntimeError):
                server.service.load("broken-model")

            self.assertIs(server.service.model, old_model)
            self.assertIs(server.service.processor, old_processor)
            self.assertIs(server.service.device, old_device)


if __name__ == "__main__":
    unittest.main()
