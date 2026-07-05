import unittest

from transcribe_wyoming.__main__ import parse_args


class CliTests(unittest.TestCase):
    def test_parse_args_supports_uri_and_device(self):
        args = parse_args(
            [
                "--uri",
                "tcp://0.0.0.0:10300",
                "--device",
                "cpu",
                "--language",
                "pl",
                "--disable-vad",
                "--vad-threshold",
                "0.6",
            ]
        )
        self.assertEqual(args.uri, "tcp://0.0.0.0:10300")
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.language, "pl")
        self.assertTrue(args.disable_vad)
        self.assertEqual(args.vad_threshold, 0.6)


if __name__ == "__main__":
    unittest.main()
