"""Unit tests for diarize output parsing, chunking and rendering.

The RAW strings below are real outputs captured from syvai/cohere-transcribe-diarize
during the implementation spike (see DIARIZE_PLAN.md).
"""
import unittest

import numpy as np

from cohere_wyoming.transcriber import (
    chunk_audio,
    parse_diarized_output,
    render_speaker_text,
)


# Real spike output for Recording.wav (12.6s, multi-speaker).
SPIKE_RAW = (
    "<|startofcontext|><|startoftranscript|><|emo:undefined|><|en|><|en|><|pnc|>"
    "<|noitn|><|timestamp|><|diarize|>"
    "<|spltoken0|><|t:0.3|> Robimy now the testeras do WAVA, how it works.<|t:3.5|>"
    "<|spltoken1|><|t:3.7|> Yeah, yeah. I'm gonna go ahead and see what you guys think about that.<|t:0.4|>"
    "<|spltoken2|><|t:0.7|> Robimy now the testeras do WAVA, like działa nash cooker whiskers,<|t:5.8|>"
    "<|spltoken3|><|t:5.4|> Mm-hmm.<|t:0.5|>"
    "<|spltoken0|><|t:6.0|> If it does not work, it's a bit of a problem.<|t:0.6|>"
    "<|endoftext|>"
)

# Model-card example format.
CARD_RAW = "<|spltoken0|><|t:0.0|> Welcome back.<|t:2.4|><|spltoken1|><|t:2.4|>"


class ParseDiarizedOutputTests(unittest.TestCase):
    def test_parses_all_segments_from_real_output(self):
        segments = parse_diarized_output(SPIKE_RAW)
        self.assertEqual([s["speaker"] for s in segments], [0, 1, 2, 3, 0])
        self.assertEqual(segments[0]["text"], "Robimy now the testeras do WAVA, how it works.")
        self.assertEqual(segments[0]["start"], 0.3)
        self.assertEqual(segments[0]["end"], 3.5)
        self.assertEqual(segments[3]["text"], "Mm-hmm.")
        self.assertEqual(segments[4]["text"], "If it does not work, it's a bit of a problem.")

    def test_strips_prompt_prefix_and_special_tokens(self):
        segments = parse_diarized_output(SPIKE_RAW)
        for segment in segments:
            self.assertNotIn("<|", segment["text"])
            self.assertNotIn("startoftranscript", segment["text"])

    def test_offset_shifts_timestamps(self):
        segments = parse_diarized_output(SPIKE_RAW, offset=30.0)
        self.assertEqual(segments[0]["start"], 30.3)
        self.assertEqual(segments[0]["end"], 33.5)

    def test_card_format_drops_trailing_empty_segment(self):
        segments = parse_diarized_output(CARD_RAW)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["speaker"], 0)
        self.assertEqual(segments[0]["text"], "Welcome back.")
        self.assertEqual(segments[0]["end"], 2.4)

    def test_leading_text_before_first_speaker_header_is_kept(self):
        # Real Polish multi-window failure mode: the model emitted the first
        # sentences without a <|spltokenN|><|t:...|> header (plus a stray
        # structural token mid-text) before switching to headed segments.
        # The parser used to drop that leading text entirely.
        raw = (
            "<|startofcontext|><|startoftranscript|><|emo:undefined|><|pl|><|pl|><|pnc|>"
            "<|noitn|><|timestamp|><|diarize|>"
            " Robimy nowy test.<|startoftranscript|> Dzień dobry wszystkim.<|t:19.8|>"
            "<|spltoken0|><|t:18.9|> Jak działa whisper?<|t:26.7|>"
            "<|spltoken1|><|t:26.1|> To dobrze.<|t:28.0|><|endoftext|>"
        )
        segments = parse_diarized_output(raw, offset=5.0)
        self.assertEqual(
            [s["text"] for s in segments],
            ["Robimy nowy test. Dzień dobry wszystkim.", "Jak działa whisper?", "To dobrze."],
        )
        # Leading text is attributed to the first headed speaker, spanning
        # from the window start to that speaker's first timestamp.
        self.assertEqual(segments[0]["speaker"], 0)
        self.assertEqual(segments[0]["start"], 5.0)
        self.assertEqual(segments[0]["end"], 23.9)

    def test_plain_text_without_diarize_tokens_falls_back(self):
        segments = parse_diarized_output("just some plain text")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["speaker"], 0)
        self.assertEqual(segments[0]["text"], "just some plain text")

    def test_empty_output_yields_no_segments(self):
        self.assertEqual(parse_diarized_output("<|diarize|><|endoftext|>"), [])

    def test_missing_final_timestamp_falls_back_to_window_duration(self):
        # Real single-segment case: the model often omits the closing <|t:END|>.
        # Without the duration fallback the segment collapses to end == start,
        # which yields a zero-length clip for speaker ID / pending voices.
        raw = "<|diarize|><|spltoken0|><|t:0.0|> Zgaś światło w salonie.<|endoftext|>"
        segments = parse_diarized_output(raw, duration=12.6)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 12.6)

        # Offset applies to the fallback end as well.
        segments = parse_diarized_output(raw, offset=30.0, duration=12.6)
        self.assertEqual(segments[0]["end"], 42.6)

        # Without duration the old (degenerate) behavior is preserved.
        segments = parse_diarized_output(raw)
        self.assertEqual(segments[0]["end"], 0.0)

    def test_plain_text_fallback_uses_duration(self):
        segments = parse_diarized_output("just some plain text", offset=5.0, duration=3.0)
        self.assertEqual(segments[0]["start"], 5.0)
        self.assertEqual(segments[0]["end"], 8.0)


class RenderSpeakerTextTests(unittest.TestCase):
    def test_renders_one_line_per_turn(self):
        text = render_speaker_text(parse_diarized_output(SPIKE_RAW))
        lines = text.split("\n")
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].startswith("Mówca 0: Robimy"))
        self.assertTrue(lines[1].startswith("Mówca 1: Yeah"))
        self.assertTrue(lines[4].startswith("Mówca 0: If it does not work"))

    def test_merges_consecutive_same_speaker(self):
        segments = [
            {"speaker": 0, "start": 0.0, "end": 1.0, "text": "Hello there."},
            {"speaker": 0, "start": 1.0, "end": 2.0, "text": "How are you?"},
            {"speaker": 1, "start": 2.0, "end": 3.0, "text": "Fine."},
        ]
        self.assertEqual(
            render_speaker_text(segments),
            "Mówca 0: Hello there. How are you?\nMówca 1: Fine.",
        )

    def test_empty_segments_render_empty_string(self):
        self.assertEqual(render_speaker_text([]), "")

    def test_uses_enrolled_name_when_present(self):
        segments = [
            {"speaker": 0, "start": 0.0, "end": 1.0, "text": "Hi.", "name": "Krzysztof"},
            {"speaker": 1, "start": 1.0, "end": 2.0, "text": "Yo."},
        ]
        self.assertEqual(
            render_speaker_text(segments),
            "Krzysztof: Hi.\nMówca 1: Yo.",
        )


class ChunkAudioTests(unittest.TestCase):
    def test_short_audio_is_single_chunk(self):
        audio = np.zeros(16000 * 5, dtype=np.float32)
        chunks = chunk_audio(audio, 16000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0][1], 0.0)

    def test_long_audio_is_split_into_30s_windows(self):
        # Uniform audio has no quiet point to prefer, so full windows are kept.
        audio = np.zeros(16000 * 70, dtype=np.float32)
        chunks = chunk_audio(audio, 16000)
        self.assertEqual(len(chunks), 3)
        self.assertEqual([offset for _, offset in chunks], [0.0, 30.0, 60.0])
        self.assertEqual(len(chunks[0][0]), 16000 * 30)
        self.assertEqual(len(chunks[2][0]), 16000 * 10)

    def test_split_lands_on_quiet_gap_instead_of_hard_boundary(self):
        sr = 16000
        t = np.arange(70 * sr, dtype=np.float32) / sr
        audio = (0.25 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        # One second of silence at 26.5-27.5 s; the first window must be cut
        # there instead of mid-sine at exactly 30 s.
        audio[int(26.5 * sr) : int(27.5 * sr)] = 0.0

        chunks = chunk_audio(audio, sr)

        offsets = [offset for _, offset in chunks]
        self.assertEqual(offsets[0], 0.0)
        self.assertTrue(26.5 <= offsets[1] <= 27.5, offsets)
        for chunk, _offset in chunks:
            self.assertLessEqual(len(chunk), 30 * sr)
        # No samples are lost or duplicated by the search.
        self.assertEqual(sum(len(chunk) for chunk, _ in chunks), len(audio))


if __name__ == "__main__":
    unittest.main()
