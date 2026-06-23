# wyoming-transcribe

Speech recognition server for Home Assistant built around the Wyoming protocol and the `syvai/cohere-transcribe-diarize` model.

In practice, this is a self-hosted speech-to-text service in the same general category as Whisper-based setups, but using Cohere Transcribe with **speaker diarization** (who spoke when) and optional **speaker identification** (mapping voices to named people you enrolled).

The main interface is the Wyoming server. `server.py` is still available as a small HTTP debug server with basic `whisper.cpp`-style compatibility, and now also hosts the speaker-enrollment web UI.

## What Is In This Repository

- `cohere_wyoming/` - shared runtime, transcription backend, diarization parsing, speaker identification, and Wyoming handler
- `python -m cohere_wyoming` - main server for Home Assistant
- `python server.py` - HTTP debug server + speaker-enrollment web UI
- HTTP frontend for file upload, microphone recording, and managing enrolled speakers
- Docker and Compose setup (runs the Wyoming server and the UI together)
- unit tests for backend, diarization, speaker identification, enrollment, HTTP, and Wyoming handler flows

## Diarization Output

Transcripts are returned with per-speaker prefixes, one line per turn:

```
Mówca 0: When do you usually play?
Mówca 1: We have a game this weekend.
```

When speaker identification is enabled and a voice matches an enrolled person, the
label becomes that person's name (e.g. `Krzysztof:`). Notes:

- The model is optimized for English; diarization/timestamps are weaker in other languages.
- Each generation pass covers up to 30 s; longer audio is split into windows automatically.
- The model's own speaker labels can be imperfect; enrollment-based identification matches
  each segment to a voiceprint independently and can correct them.

## Requirements

- Python 3.11+
- `uv` as the primary dependency manager
- GPU is preferred, but CPU fallback is supported
- `HF_TOKEN` if you are using the gated Hugging Face model

## Quick Start

### 1. Install with uv

```bash
cp .env.example .env
UV_CACHE_DIR=/tmp/uv-cache uv venv
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

Set `HF_TOKEN` in `.env` before the first model download.

### 2. Start the Wyoming server

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m cohere_wyoming \
  --uri tcp://0.0.0.0:10300 \
  --language pl
```

The default port for Home Assistant integration is `10300`.

### 3. HTTP debug server + speaker-enrollment UI

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python server.py --host 0.0.0.0 --port 8580 --language pl
# UI-only (no ASR model loaded, just enrollment management):
UV_CACHE_DIR=/tmp/uv-cache uv run python server.py --host 0.0.0.0 --port 8580 --no-load-model
```

The HTTP frontend supports file upload, browser microphone recording, and a **Speaker
recognition** panel to define people, record/upload/play back/delete voice samples — full
control over enrollment. Formats are decoded through the `ffmpeg` fallback (WAV, MP3, FLAC,
OGG, WebM, M4A).

## Docker

The simplest option:

```bash
docker compose up --build -d
```

`compose.yml` runs the Wyoming service on port `10300`, the enrollment UI on port `8580`,
mounts `./speakers` for enrolled voices, and includes a ready-to-use VAD preset for Home
Assistant. The container starts both services from `docker-entrypoint.sh` (the UI process
runs with `--no-load-model`, so the ASR model is loaded only once, by the Wyoming process).

### Startup, warmup and model cache

On startup the Wyoming process loads the diarize model and then runs a short **ASR warmup**
(a dummy `_generate_diarized` pass) so the *first* real transcription is fast instead of
paying CUDA-kernel compilation on the first request.

The model is downloaded from Hugging Face **once** into the cache volume
(`/root/.cache/huggingface`) and loaded from that local copy on every subsequent start
(load ≈ a few seconds, no re-download). Note: this model is distributed via **HF Xet**, so
make sure the **first download fully completes** — if the container is killed mid-download,
the weights stay `*.incomplete` (no `model.safetensors` snapshot symlink) and get re-fetched
on each boot. To pre-fetch reliably without the container, run once:

```bash
docker run --rm -v /opt/wyoming-transcribe/data:/root/.cache/huggingface \
  -e HF_TOKEN="$HF_TOKEN" python:3.12-slim \
  sh -c "pip install -q huggingface_hub && hf download syvai/cohere-transcribe-diarize"
```

Do **not** set `HF_HUB_DISABLE_XET=1` (breaks weight resolution for this model).

Manual container start is also possible:

```bash
docker build -t wyoming-transcribe .
docker run --gpus all -p 10300:10300 -p 8580:8580 \
  -v /opt/wyoming-transcribe/speakers:/app/speakers \
  -e HF_TOKEN=hf_your_token_here \
  -e SPEAKER_ID_ENABLED=true \
  wyoming-transcribe \
  --uri tcp://0.0.0.0:10300 \
  --language pl
```

The Docker image uses `uv.lock`, so builds stay aligned with the locked dependency set.

## Home Assistant

Typical setup:

1. Start the Wyoming server.
2. Add it in Home Assistant through the `Wyoming Protocol` integration.
3. In Home Assistant, enter the host and port `10300`.

Currently supported events:

- `describe`
- `transcribe`
- `audio-start`
- `audio-chunk`
- `audio-stop`

Transcription runs after the full utterance is received, on `audio-stop`.

## Silence and Noise Handling

The backend has several layers to reduce hallucinations on silence:

- prefer local Hugging Face cache before network download
- fall back from CUDA to CPU if GPU model loading fails
- detect effective silence with a fast energy-based filter (`RMS/peak`)
- use `silero-vad` as a more precise speech detector
- apply additional `speech RMS` and `speech-to-noise ratio` checks to reject very quiet sounds close to background noise

If `silero-vad` cannot be loaded, the server falls back to the simpler silence detector and keeps running.

Important environment variables:

- `VAD_ENABLED=true`
- `VAD_THRESHOLD=0.5`
- `VAD_MIN_SPEECH_DURATION_MS=250`
- `VAD_MIN_SILENCE_DURATION_MS=100`
- `VAD_SPEECH_PAD_MS=30`
- `VAD_MIN_TOTAL_SPEECH_MS=60`
- `VAD_MIN_MAX_SEGMENT_MS=40`
- `VAD_MIN_SPEECH_RMS=0.012`
- `VAD_MIN_SPEECH_TO_NOISE_RATIO=3.0`
- `VAD_USE_ONNX=false`

Useful CLI options:

- `--disable-vad`
- `--vad-threshold 0.6`

Recommended starting preset for Home Assistant:

```env
VAD_ENABLED=true
VAD_THRESHOLD=0.54
VAD_MIN_SPEECH_DURATION_MS=180
VAD_MIN_SILENCE_DURATION_MS=120
VAD_SPEECH_PAD_MS=50
VAD_MIN_TOTAL_SPEECH_MS=70
VAD_MIN_MAX_SEGMENT_MS=45
VAD_MIN_SPEECH_RMS=0.014
VAD_MIN_SPEECH_TO_NOISE_RATIO=2.6
```

Practical tuning guidance:

- if you still get hallucinations on silence or noise, increase `VAD_THRESHOLD`
- if very quiet vowels or speech-like noise still pass through, increase `VAD_MIN_SPEECH_RMS`
- if sounds only slightly louder than background noise still pass through, increase `VAD_MIN_SPEECH_TO_NOISE_RATIO`
- if short commands get cut off, lower `VAD_MIN_TOTAL_SPEECH_MS` and `VAD_MIN_MAX_SEGMENT_MS`
- if the start or end of speech gets clipped, increase `VAD_SPEECH_PAD_MS`

## Speaker Recognition (enrollment)

Anonymous diarized speakers (`Mówca 0`, `Mówca 1`, ...) can be mapped to named people by
enrolling voice samples. Each transcript segment's ECAPA-TDNN voiceprint is compared to the
enrolled profiles; the closest match above a threshold becomes the speaker's name.

Enrollment layout (managed via the UI on port `8580`, or by hand):

```
speakers/
  Krzysztof/  sample1.wav  sample2.wav
  Anna/       sample1.wav
```

Environment variables:

- `SPEAKER_ID_ENABLED=true` — turn identification on (off by default)
- `SPEAKER_ENROLLMENT_DIR=/app/speakers` — enrollment directory
- `SPEAKER_MATCH_THRESHOLD=0.35` — cosine threshold; raise to reduce false matches
- `SPEAKER_MODEL=speechbrain/spkrec-ecapa-voxceleb` — embedding model
- `SPEAKER_MODEL_CACHE=/root/.cache/huggingface/ecapa` — where to cache the ECAPA model

Practical guidance:

- Enroll 10–30 s of clean speech per person, ideally from the same microphone as real use.
- Very short utterances (1–2 words) give weak voiceprints and may stay anonymous.
- Speaker labels reset per clip; identification is global, so enrollment also stabilizes them.
- The ECAPA cost is negligible next to the ASR model, so it does not affect realtime latency.

## Supported Languages

`en`, `fr`, `de`, `it`, `es`, `pt`, `el`, `nl`, `pl`, `zh`, `ja`, `ko`, `vi`, `ar`

Diarization and timestamps are validated mainly for English; other languages transcribe but
may diarize less reliably.

## Current Limitations

- no partial transcripts
- no language auto-detection
- no `zeroconf`
- no native streaming results
- diarization speaker labels are per-clip and can be imperfect (mitigated by enrollment)
- HTTP remains a helper/debug layer (now also hosting the enrollment UI)

## Tests

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

HTTP smoke test script:

```bash
tests/test_api.sh
```

## Reference

- `wyoming-faster-whisper`: https://github.com/rhasspy/wyoming-faster-whisper

## License

This repository is licensed under the Apache License 2.0.
