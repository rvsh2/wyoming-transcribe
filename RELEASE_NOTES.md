# Release Notes

## 2026-06-22

### Speaker Diarization Engine

- Switched the ASR model to `syvai/cohere-transcribe-diarize` (loads with `trust_remote_code=False`, `bfloat16` on GPU).
- Transcripts now carry per-speaker prefixes (`Mówca 0:` / enrolled name), with timestamped segments exposed in `verbose_json`.
- Diarize prompt is built from decoder tokens (`<|diarize|>`, `<|timestamp|>`); audio longer than 30 s is split into windows automatically.

### Speaker Identification (ECAPA-TDNN)

- Added `cohere_wyoming/speaker_id.py`: per-person enrollment profiles, on-disk embedding cache, and per-segment identification.
- Matching is per segment (not per model label), which corrects the model's occasional mislabeling; verified end-to-end on a clean multi-speaker English clip (matches 0.66–1.0).
- Configurable via `SPEAKER_*` env vars; off by default.

### Speaker Enrollment Web UI

- New panel in the HTTP server (port `8580`) to define people and record/upload/play back/delete voice samples — full CRUD over enrollment.
- Endpoints: `GET/POST /speakers`, `DELETE /speakers/{name}`, `POST/GET/DELETE /speakers/{name}/samples/...`.
- Container runs the Wyoming server and the UI together (`docker-entrypoint.sh`; UI uses `--no-load-model` so the ASR model loads only once).

### Fixes

- `read_audio_to_numpy` now decodes M4A/MP4 (ffmpeg via temp file instead of stdin pipe) and no longer crashes building its error message.

### Review follow-ups

- Clamp `end >= start` in verbose_json segments (the model can emit end < start).
- Warm the speaker registry at Wyoming startup so the first request doesn't block the event loop on ECAPA load.
- `/speakers` reports enrolled people from disk (the UI process never loads profiles).
- `_build_prompt_ids` fails fast on a missing structural token and falls back to English for a missing language token; prompt ids are cached per language.
- Batched ECAPA embedding for per-segment identification; `delete_speaker` uses `rmtree`; sample filenames include a uuid to avoid same-millisecond collisions.

### Verification

- `60` unit tests pass (new suites: diarization parsing, speaker identification, enrollment).
- End-to-end checks on the real model + ECAPA on RTX 3090.

## 2026-03-29

### Native Production Backend

- Switched production inference to the native `transformers==5.4.0` path using `AutoProcessor` and `AutoModelForSpeechSeq2Seq`.
- Removed production dependence on `trust_remote_code=True`.
- Kept GPU execution on the CUDA 12.4 / PyTorch 2.6.0 stack validated on RTX 3090.

### Audio Normalization

- Standardized all incoming audio to mono `16 kHz` before inference.
- Confirmed support for mixed source sample rates such as `16 kHz`, `22.05 kHz`, `44.1 kHz`, and `48 kHz`.
- Preserved support for common upload formats including WAV, MP3, FLAC, and OGG.

### Verification

- Revalidated the API with endpoint tests and Docker runtime checks.
- Confirmed successful native-backend transcription for:
  - `Recording2.wav` -> `Dzien dobry wszystkim.`
  - `Recording.wav` -> meaningful Polish transcription with minor ASR errors only

### Notes

- `whisper.cpp` route compatibility remains unchanged.
- Compatibility-only request parameters are still accepted where already supported by the API wrapper.
