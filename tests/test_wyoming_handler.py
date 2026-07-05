import asyncio
import unittest
from types import SimpleNamespace

import numpy as np

from transcribe_wyoming.handler import TranscribeEventHandler
from transcribe_wyoming.wyoming_protocol import Event


class CollectingHandler(TranscribeEventHandler):
    def __init__(self, transcriber, info_event):
        super().__init__(transcriber, info_event)
        self.events = []

    async def write_event(self, event):
        self.events.append(event)


class WyomingHandlerTests(unittest.TestCase):
    def run_async(self, coro):
        return asyncio.run(coro)

    def test_describe_returns_info_event(self):
        handler = CollectingHandler(transcriber=SimpleNamespace(), info_event=Event("describe", {"name": "info"}))
        self.run_async(handler.handle_event(Event("describe", {})))
        self.assertEqual(handler.events[0].type, "describe")

    def test_audio_stop_returns_transcript(self):
        class FakeTranscriber:
            def __init__(self):
                self.calls = []

            def transcribe_pcm(self, audio_data, *, sample_rate, language):
                self.calls.append((audio_data, sample_rate, language))
                return SimpleNamespace(text="hello wyoming", language="pl")

        transcriber = FakeTranscriber()
        handler = CollectingHandler(transcriber=transcriber, info_event=Event("describe", {}))

        pcm = (np.array([0, 1000, -1000, 500], dtype="<i2")).tobytes()
        self.run_async(handler.handle_event(Event("transcribe", {"language": "pl"})))
        self.run_async(handler.handle_event(Event("audio-start", {"rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-chunk", {"audio": pcm, "rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))

        self.assertEqual(handler.events[-1].type, "transcript")
        self.assertEqual(handler.events[-1].data["text"], "hello wyoming")
        _, sample_rate, language = transcriber.calls[0]
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(language, "pl")

    def test_empty_audio_returns_empty_transcript(self):
        handler = CollectingHandler(transcriber=SimpleNamespace(), info_event=Event("describe", {}))
        self.run_async(handler.handle_event(Event("transcribe", {"language": "en"})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))
        self.assertEqual(handler.events[-1].data["text"], "")

    def _run_pipeline(self, transcriber):
        handler = CollectingHandler(transcriber=transcriber, info_event=Event("describe", {}))
        pcm = (np.array([0, 1000, -1000, 500], dtype="<i2")).tobytes()
        self.run_async(handler.handle_event(Event("transcribe", {"language": "pl"})))
        self.run_async(handler.handle_event(Event("audio-start", {"rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-chunk", {"audio": pcm, "rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))
        return handler.events[-1]

    def test_transcript_event_carries_speaker_field_in_field_mode(self):
        class FakeTranscriber:
            def transcribe_pcm(self, *_args, **_kwargs):
                return SimpleNamespace(
                    text="zgaś światło",
                    language="pl",
                    text_mode="field",
                    speaker="Krzysztof",
                    speaker_score=0.82,
                )

        event = self._run_pipeline(FakeTranscriber())
        self.assertEqual(event.data["text"], "zgaś światło")
        self.assertEqual(event.data["speaker"], "Krzysztof")
        self.assertEqual(event.data["speaker_score"], 0.82)

    def test_transcript_event_carries_utterance_id_in_every_mode(self):
        class FakeTranscriber:
            def transcribe_pcm(self, *_args, **_kwargs):
                return SimpleNamespace(
                    text="Speaker 0: kto to?",
                    language="pl",
                    text_mode="prefix",
                    speaker=None,
                    speaker_score=None,
                    speaker_role=None,
                    utterance_id="utt-123-abcdef01",
                )

        event = self._run_pipeline(FakeTranscriber())
        # utterance_id is needed for the "who are you?" flow even in prefix mode.
        self.assertEqual(event.data["utterance_id"], "utt-123-abcdef01")
        self.assertNotIn("speaker", event.data)

    def test_transcript_event_carries_role_in_field_mode(self):
        class FakeTranscriber:
            def transcribe_pcm(self, *_args, **_kwargs):
                return SimpleNamespace(
                    text="zgaś światło",
                    language="pl",
                    text_mode="field",
                    speaker="Krzysztof",
                    speaker_score=0.82,
                    speaker_role="admin",
                    utterance_id=None,
                )

        event = self._run_pipeline(FakeTranscriber())
        self.assertEqual(event.data["speaker"], "Krzysztof")
        self.assertEqual(event.data["speaker_role"], "admin")
        self.assertNotIn("utterance_id", event.data)

    def test_transcript_event_omits_speaker_field_in_prefix_mode(self):
        class FakeTranscriber:
            def transcribe_pcm(self, *_args, **_kwargs):
                return SimpleNamespace(
                    text="Krzysztof: zgaś światło",
                    language="pl",
                    text_mode="prefix",
                    speaker="Krzysztof",
                    speaker_score=0.82,
                )

        event = self._run_pipeline(FakeTranscriber())
        self.assertEqual(event.data["text"], "Krzysztof: zgaś światło")
        self.assertNotIn("speaker", event.data)

    def test_transcription_error_still_returns_transcript(self):
        class FailingTranscriber:
            def transcribe_pcm(self, *_args, **_kwargs):
                raise RuntimeError("model exploded")

        handler = CollectingHandler(transcriber=FailingTranscriber(), info_event=Event("describe", {}))

        pcm = (np.array([0, 1000, -1000, 500], dtype="<i2")).tobytes()
        self.run_async(handler.handle_event(Event("transcribe", {"language": "pl"})))
        self.run_async(handler.handle_event(Event("audio-start", {"rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-chunk", {"audio": pcm, "rate": 16000, "width": 2, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))

        # HA's pipeline hangs without a Transcript, so errors must still answer.
        self.assertEqual(handler.events[-1].type, "transcript")
        self.assertEqual(handler.events[-1].data["text"], "")

    def test_bad_audio_width_still_returns_transcript(self):
        class UnusedTranscriber:
            def transcribe_pcm(self, *_args, **_kwargs):
                raise AssertionError("should not be reached")

        handler = CollectingHandler(transcriber=UnusedTranscriber(), info_event=Event("describe", {}))

        pcm = bytes(12)
        self.run_async(handler.handle_event(Event("transcribe", {"language": "pl"})))
        self.run_async(handler.handle_event(Event("audio-start", {"rate": 16000, "width": 4, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-chunk", {"audio": pcm, "rate": 16000, "width": 4, "channels": 1})))
        self.run_async(handler.handle_event(Event("audio-stop", {})))

        self.assertEqual(handler.events[-1].type, "transcript")
        self.assertEqual(handler.events[-1].data["text"], "")
