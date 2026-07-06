import unittest

from transcribe_wyoming.transcriber import render_speaker_text, segment_label


def seg(speaker, text, name=None, start=0.0, end=1.0):
    s = {"speaker": speaker, "start": start, "end": end, "text": text}
    if name:
        s["name"] = name
    return s


class RenderSpeakerTextTests(unittest.TestCase):
    def test_renders_one_line_per_turn(self):
        text = render_speaker_text([seg(0, "Hello there."), seg(1, "Hi.")])
        self.assertEqual(text, "Speaker 0: Hello there.\nSpeaker 1: Hi.")

    def test_merges_consecutive_same_speaker(self):
        text = render_speaker_text([seg(0, "One."), seg(0, "Two.")])
        self.assertEqual(text, "Speaker 0: One. Two.")

    def test_empty_segments_render_empty_string(self):
        self.assertEqual(render_speaker_text([]), "")

    def test_uses_enrolled_name_when_present(self):
        text = render_speaker_text([seg(0, "Turn off the lights.", name="Krzysztof")])
        self.assertEqual(text, "Krzysztof: Turn off the lights.")

    def test_field_mode_omits_labels(self):
        text = render_speaker_text([seg(0, "Hello.", name="Krzysztof")], mode="field")
        self.assertEqual(text, "Hello.")

    def test_segment_label_falls_back_to_indexed_label(self):
        self.assertEqual(segment_label(seg(3, "x")), "Speaker 3")
        self.assertEqual(segment_label(seg(0, "x", name="Ala")), "Ala")


if __name__ == "__main__":
    unittest.main()
