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

## How to use it (end-to-end walkthrough)

The typical life of this service, from zero to an LLM that knows who is speaking:

**Step 1 — start the server.** `docker compose up -d` (details in [Docker](#docker)).
Two services run in one container: the Wyoming ASR server on port `10300` (this is
what Home Assistant talks to) and the management UI/API on port `8580`.

**Step 2 — plug into Home Assistant.** Add the *Wyoming Protocol* integration
pointing at `<host>:10300` and select it as speech-to-text in your voice pipeline.
Transcription works immediately; every transcript line is prefixed with the speaker
(`Mówca 0: ...`).

**Step 3 — open the management UI** (`http://127.0.0.1:8580` on the Docker host, via
SSH tunnel, or as a Home Assistant sidebar panel — see
[HACS integration](#hacs-integration-ui-panel--sensors-in-ha)). From here you manage
everything below.

**Step 4 — teach it who people are.** Two ways, freely combined:

- *Up front*: add a person in the "Mówcy" card and record/upload 10–30 s of their
  speech.
- *From real usage*: just talk to the assistant. Every utterance whose voice is not
  recognized lands in the **"Nierozpoznane głosy"** card, grouped by voice. Listen,
  check who it is, and assign the whole group to a person (existing or new) with two
  clicks. The person's profile is built from all clips in the group at once.

From the next utterance the voice is recognized and the transcript says
`Krzysztof: zgaś światło` instead of `Mówca 0: zgaś światło`.

**Step 5 — give people roles.** Next to each person pick `admin`, `user` (default)
or `guest`. The role travels with the transcription (see below), so the LLM can be
told e.g. "only admin may unlock doors".

**Step 6 — decide how the pipeline receives identity** (the "Ustawienia" card,
switchable at runtime):

| Mode | Transcript text | Extra fields in the Wyoming `Transcript` event |
|---|---|---|
| `prefix` (default) | `Krzysztof: zgaś światło` | only `utterance_id` (when voice unknown) |
| `field` | `zgaś światło` | `speaker`, `speaker_score`, `speaker_role`, `utterance_id` |
| `both` | `Krzysztof: zgaś światło` | `speaker`, `speaker_score`, `speaker_role`, `utterance_id` |

Start with `prefix` — it needs zero pipeline changes; an LLM conversation agent
simply sees who speaks in the text. Switch to `field`/`both` once your pipeline
consumes the event fields.

**Step 7 (optional) — let the LLM enroll people itself.** When the event carries
`speaker: null` and an `utterance_id`, the agent can ask *"kim jesteś?"* and, after
the answer, call one tool:

```bash
curl -X POST -H "X-API-Token: $API_TOKEN" -F include_cluster=true \
  "http://HOST:8580/speakers/Krzysztof/samples/from-utterance/<utterance_id>"
```

That single call creates the person (if new), claims **all** buffered clips of that
voice as enrollment samples, and from the next sentence the person is recognized.
Full details: [Unknown voices](#unknown-voices--who-are-you-enrollment).

### API cheat sheet

All endpoints live on port `8580`; with `API_TOKEN` set, send
`X-API-Token: <token>` (or `Authorization: Bearer <token>`) — only `/` and `/health`
are open.

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness/liveness, VAD + speaker-ID status |
| `POST /inference`, `POST /v1/audio/transcriptions` | HTTP transcription (whisper.cpp / OpenAI shape; needs a process with the model loaded) |
| `GET`/`POST /settings` | read / switch `speaker_text_mode` at runtime |
| `GET /speakers` | people, their samples and roles |
| `POST /speakers` (form `name`) | add a person |
| `POST /speakers/{name}/samples` (file upload) | add a voice sample |
| `POST /speakers/{name}/role` (form `role`) | set `admin`/`user`/`guest` |
| `GET /pending` | unrecognized clips, grouped by voice |
| `GET /pending/{id}/audio` / `DELETE /pending/{id}` | listen to / discard a clip |
| `POST /speakers/{name}/samples/from-utterance/{id}` | claim a clip (+its voice group) as samples — the LLM enrollment tool |

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
runs with `--no-load-model`, so the ASR model is loaded only once, by the Wyoming process);
if either service dies, the container exits so Docker's restart policy brings both back up.

The enrollment UI / management API has **no authentication** and includes destructive
endpoints (deleting speakers, hot-swapping the model), so compose publishes port `8580`
on `127.0.0.1` only. To use the UI from another machine, tunnel it
(`ssh -L 8580:127.0.0.1:8580 <docker-host>`) or put it behind an authenticating reverse
proxy. HTTP transcription (`POST /inference`) is unavailable in the default Docker setup —
the UI process runs without the ASR model; speech-to-text is served by the Wyoming process
on port `10300`.

### Startup, warmup and model cache

On startup the Wyoming process loads the diarize model and then runs a short **warmup**
(Silero VAD load + a dummy `_generate_diarized` pass) so the *first* real transcription
is fast instead of paying lazy model loading and CUDA-kernel compilation on the first
request.

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
docker run --gpus all -p 10300:10300 -p 127.0.0.1:8580:8580 \
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

### HACS integration (UI panel + sensors in HA)

This repo ships a Home Assistant custom integration (`custom_components/wyoming_transcribe`)
that embeds the enrollment/management UI as a **sidebar panel** in HA and adds status
sensors (model status, enrolled speakers, pending unrecognized voices — handy for an
automation like "notify me when a new unknown voice is waiting").

Install:

1. On the server: set `API_TOKEN=<secret>` and `UI_BIND=0.0.0.0` in `.env`, then
   `docker compose up -d` (never expose port 8580 without a token).
2. HACS → Integrations → ⋮ → **Custom repositories** → add this repo URL, category
   *Integration* → install **Wyoming Transcribe** → restart HA.
3. Settings → Devices & services → **Add integration** → *Wyoming Transcribe* → enter
   the Docker host's LAN address, port `8580` and the API token.
4. A "Wyoming Transcribe" entry appears in the sidebar (admin-only). Inside the
   embedded UI, paste the same token into the *API token* box once (stored in the
   browser).

Note: the sidebar panel is an iframe, so the browser talks to port 8580 directly —
the host you configure must be reachable from clients, not only from HA. Without
HACS you can get the same panel with a manual `panel_iframe` entry in
`configuration.yaml`.

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
- `SPEAKER_CHAIN_THRESHOLD=0.40` — cosine threshold for linking the *same anonymous
  speaker* across 30 s diarization windows in recordings longer than 30 s (raise to
  split more, lower to merge more)

Practical guidance:

- Enroll 10–30 s of clean speech per person, ideally from the same microphone as real use.
- Identification is done once per diarized speaker on the concatenation of all their
  segments, so even a short command builds a usable voiceprint.
- Anonymous `Mówca N` labels are made consistent across 30 s windows by ECAPA voiceprint
  matching; enrollment naming is still the most reliable way to stabilize identity.

### Speaker identity for the HA pipeline

How identity is delivered is configurable (`SPEAKER_TEXT_MODE`, or at runtime in the
UI settings card — the Wyoming process picks the change up on the next transcription):

- `prefix` (default) — transcript text lines are prefixed: `Krzysztof: zgaś światło`.
  Simplest for an LLM conversation agent; no pipeline changes needed.
- `field` — the text stays clean (`zgaś światło`) and the Wyoming `Transcript` event
  carries extra data: `speaker` (enrolled name of the dominant speaker, or `null`) and
  `speaker_score`. Home Assistant ignores unknown event fields, so this is safe today
  and usable by a custom pipeline component later.
- `both` — prefixed text and the event fields.

`verbose_json` HTTP responses include `speaker`/`speaker_score` in all modes.

### Unknown voices & "who are you?" enrollment

When speaker identification is enabled and the **dominant speaker of an utterance does
not match any enrolled person**, the Wyoming process buffers that speaker's audio
(their concatenated segments, min. `PENDING_MIN_SECONDS`, default 1 s) in
`<enrollment_dir>/.pending/` together with the transcript and an ECAPA voiceprint.
The `Transcript` event then carries an `utterance_id` (in **every** text mode).

Three ways to resolve a pending voice:

1. **UI (manual verification)** — the "Nierozpoznane głosy" card lists pending clips
   *grouped by voice* (one group = one person, matched by voiceprint similarity,
   `PENDING_CLUSTER_THRESHOLD`, default 0.40). Play each clip, check who it is, and
   assign the group to an existing or new person. Assigning a group moves all its
   clips to that person as enrollment samples at once, so the profile is immediately
   built from several samples instead of one weak clip.
2. **LLM pipeline ("who are you?")** — when the event has `speaker: null` and an
   `utterance_id`, the conversation agent can ask the person for their name and then
   call (as a tool):

   ```bash
   curl -X POST -H "X-API-Token: $API_TOKEN" \
     -F include_cluster=true \
     "http://HOST:8580/speakers/Krzysztof/samples/from-utterance/utt-1783015551000-ab12cd34"
   ```

   `include_cluster=true` (default) claims **all pending clips of the same voice**,
   so one confirmed answer enrolls the person with several samples. The person is
   created automatically if missing. From the next utterance the voice is recognized
   (`speaker: "Krzysztof"`).
3. **Ignore/delete** — pending clips live in a ring buffer (`PENDING_MAX_CLIPS`,
   default 40); oldest are pruned automatically, or delete them in the UI.

Pending API: `GET /pending` (clusters), `GET /pending/{id}/audio`,
`DELETE /pending/{id}`, `POST /speakers/{name}/samples/from-utterance/{id}`.

Privacy note: this stores voice clips of unrecognized people (guests included) on
disk until pruned, claimed, or deleted.

### Speaker roles (authorization for the pipeline)

Every enrolled person has a role: `admin`, `user` (default) or `guest`, set in the UI
next to the person (or via `POST /speakers/{name}/role` with `role=admin`). The role
of the recognized dominant speaker is delivered as:

- `speaker_role` in the Wyoming `Transcript` event (modes `field`/`both`),
- `speaker_role` in `verbose_json` HTTP responses,
- `role` per person in `GET /speakers`.

The STT server only *labels* the speaker — enforcement belongs in the HA pipeline
(e.g. the LLM prompt: "only `admin` may unlock doors; `guest` may only ask
questions"). Voice can be imitated, so do not treat the role as strong
authentication for critical actions.

### Management API auth

Set `API_TOKEN` to require a token on every endpoint except `/` and `/health`
(clients send `X-API-Token: <token>` or `Authorization: Bearer <token>`; the UI has a
token box that stores it in the browser). With `API_TOKEN` unset, auth is disabled and
the loopback-only port publishing in `compose.yml` is the safety boundary.
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
