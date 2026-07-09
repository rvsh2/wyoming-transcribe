# Wyoming Transcribe

Self-hosted speech-to-text for Home Assistant (Wyoming protocol) with
**speaker identification** (mapping voices to named people you enrolled).

The speech-to-text itself runs in a [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
server (e.g. `large-v3-turbo`) reached over HTTP; this service owns the rest of
the pipeline: Silero VAD cropping, ECAPA voiceprint speaker identification,
unknown-voice enrollment and recognition history. Each utterance is treated as
a single speaker. Transcripts come back with a speaker prefix; when the voice
matches an enrolled person, the label becomes their name:

```
Krzysztof: turn off the lights
Speaker 0: could you turn on the TV
```

(Each utterance is one speaker; unrecognized voices are always labeled with
index 0.)

The label for unenrolled speakers is configurable via `SPEAKER_LABEL`
(default `Speaker`; set e.g. `SPEAKER_LABEL=Mówca` for Polish deployments —
the label ends up in the transcript your conversation agent sees).

Point `WHISPERCPP_URL` (default `http://whispercpp:4050`) at a running
[whisper.cpp server](https://github.com/ggml-org/whisper.cpp/tree/master/examples/server),
e.g. `whisper-server --host 0.0.0.0 --port 4050 --model /models/ggml-large-v3-turbo.bin`
(beam search is requested per inference call, not via server flags).

## Quick start (Docker)

```bash
cp .env.example .env   # set API_TOKEN, optionally UI_BIND / WHISPERCPP_URL
docker compose up --build -d
```

One container runs two services:

- **Wyoming ASR server on port `10300`** — what Home Assistant talks to.
- **Management UI/API on port `8580`** — speakers, roles, unknown voices,
  settings, backup. Published on `127.0.0.1` only by default; to reach it from
  another machine (e.g. HA on a different host) set `UI_BIND=0.0.0.0` **and**
  `API_TOKEN=<secret>` in `.env`. With `API_TOKEN` set, all endpoints except
  `/` and `/health` require `X-API-Token: <token>` (or `Authorization: Bearer`).

The (public) Silero VAD and ECAPA speaker models download from Hugging Face
once into the cache volume on first start.

## Home Assistant

1. Add the **Wyoming Protocol** integration pointing at `<host>:10300` and
   select it as speech-to-text in your voice pipeline. Transcription works
   immediately.
2. (Recommended) Install the custom integration below for the management panel
   and automation hooks.

### Custom integration (HACS): panel + sensors + services

The repo ships `custom_components/wyoming_transcribe`: the full
management UI as a native HA sidebar panel (admin-only), status sensors (model
status, enrolled speakers, pending voices) and automation/LLM hooks. The panel
talks to the server through an authenticated proxy — the API token lives only
in the integration's config, and port `8580` needs to be reachable from the HA
host only. Microphone recording in the panel requires HA over HTTPS (same
browser rule as Assist); on plain http you can still upload files or assign
real utterances.

Install:

1. On the server: set `API_TOKEN` (and `UI_BIND=0.0.0.0` if HA runs on another
   host), `docker compose up -d`.
2. HACS → Integrations → ⋮ → **Custom repositories** → add this repo URL,
   category *Integration* → install **Wyoming Transcribe** → restart HA.
3. Settings → Devices & services → **Add integration** →
   *Wyoming Transcribe* → enter the server host, port `8580` and the API
   token.

Hooks for automations and LLM tools:

- **Event `wyoming_transcribe_new_pending`** — a new unrecognized voice
  landed in the pending buffer (data: `utterance_id`, `text`, `seconds`,
  `created`, `voice_utterances`). Pending clips are polled every 15 s.
- **Service `wyoming_transcribe.claim_latest`** — voice-anchored
  enrollment of the newest unknown utterance (the "who are you?" tool).
- **Service `wyoming_transcribe.claim_utterance`** — enroll an explicit
  clip by `utterance_id`.
- **Service `wyoming_transcribe.check_latest_voice`** — how much the
  current unknown voice has already talked; returns a `should_ask` verdict.
- **Service `wyoming_transcribe.set_role`** — set `admin`/`user`/`guest`.

Ready-made scripts and system-prompt snippets for an LLM conversation agent
(the "who are you?" enrollment flow, role-based authorization patterns):
**[docs/llm-agent.md](docs/llm-agent.md)**.

## Speaker identification

Enable with `SPEAKER_ID_ENABLED=true`. The utterance's ECAPA-TDNN voiceprint
is compared to enrolled profiles; the closest match above
`SPEAKER_MATCH_THRESHOLD` becomes the name.

- **Enrollment**: add a person in the panel and record/upload 10–30 s of clean
  speech (ideally from the microphone used in real life) — or just talk to the
  assistant: unknown voices land in the **pending** section grouped by voice,
  and you assign a whole group to a person with two clicks.
- **Roles**: each person is `admin`, `user` (default) or `guest`; the role
  travels with the transcription so the agent can enforce policy. Treat it as
  convenience authorization, not strong authentication — critical actions
  should require a second factor.
- **Profile adaptation**: every confident recognition (score ≥
  `SPEAKER_ADAPT_MIN_SCORE`, default 0.60) folds into an adaptive vector
  blended 1:1 with the enrolled samples, so recognition improves with use but
  cannot drift away from the enrolled anchor. Disable with
  `SPEAKER_ADAPT_ENABLED=false`.
- **Identity delivery** (`SPEAKER_TEXT_MODE`, switchable at runtime in the
  panel):

  | Mode | Transcript text | Extra fields in the Wyoming `Transcript` event |
  |---|---|---|
  | `prefix` (default) | `Krzysztof: zgaś światło` | `utterance_id` (when voice unknown) |
  | `field` | `zgaś światło` | `speaker`, `speaker_score`, `speaker_role`, `utterance_id` |
  | `both` | `Krzysztof: zgaś światło` | all of the above |

  Start with `prefix` — zero pipeline changes; an LLM agent simply sees who
  speaks in the text.

Privacy note: unknown-voice clips (guests included) are stored on disk in a
ring buffer (`PENDING_MAX_CLIPS`, default 40) until pruned, claimed or deleted.

## Configuration

Key environment variables (more in `.env.example` and `compose.yml`):

| Variable | Default | Purpose |
|---|---|---|
| `API_TOKEN` | unset | management API auth; required if 8580 leaves localhost |
| `UI_BIND` | `127.0.0.1` | bind address of the management UI/API |
| `SPEAKER_ID_ENABLED` | `false` | turn speaker identification on |
| `SPEAKER_MATCH_THRESHOLD` | `0.35` | cosine threshold; raise to reduce false matches |
| `VAD_ENABLED` | `true` | Silero-VAD silence/noise filtering + speech-span cropping |
| `WHISPER_INITIAL_PROMPT` | unset | vocabulary hint sent as the whisper initial prompt |

`WHISPER_INITIAL_PROMPT` biases decoding toward household vocabulary (the
assistant's name, people, rooms, devices), which fixes misheard proper names
on degraded far-field audio. Use nouns/proper names only, e.g.
`Agata (asystentka domowa), Krzysztof, salon, telewizor.` — example commands
in the prompt get echoed back as hallucinations when the clip is mostly noise.

Recommended VAD preset for Home Assistant (already in `compose.yml`):

```env
VAD_THRESHOLD=0.54
VAD_MIN_SPEECH_DURATION_MS=180
VAD_MIN_SILENCE_DURATION_MS=120
VAD_SPEECH_PAD_MS=50
VAD_MIN_TOTAL_SPEECH_MS=70
VAD_MIN_MAX_SEGMENT_MS=45
VAD_MIN_SPEECH_RMS=0.007
VAD_MIN_SPEECH_TO_NOISE_RATIO=2.6
```

With VAD active, audio is also cropped to the detected speech span (±0.1 s)
before transcription — the mostly-silent padding around short voice commands
otherwise makes the model hallucinate, especially in non-English languages.

Tuning: hallucinations on silence → raise `VAD_THRESHOLD`; quiet speech-like
noise passes → raise `VAD_MIN_SPEECH_RMS` / `VAD_MIN_SPEECH_TO_NOISE_RATIO`;
short commands get cut → lower `VAD_MIN_TOTAL_SPEECH_MS` and
`VAD_MIN_MAX_SEGMENT_MS`; clipped starts/ends → raise `VAD_SPEECH_PAD_MS`.

## Management API

Everything the panel does is plain HTTP on port `8580`:

| Endpoint | Purpose |
|---|---|
| `GET /health` | readiness/liveness, VAD + speaker-ID status |
| `GET`/`POST /settings` | read / switch `speaker_text_mode` at runtime |
| `GET /speakers`, `POST /speakers` | list / add people |
| `POST /speakers/{name}/samples` | upload a voice sample |
| `POST /speakers/{name}/role` | set `admin`/`user`/`guest` |
| `GET /pending`, `GET /pending/{id}/audio`, `DELETE /pending/{id}` | unknown voices, grouped by voice |
| `POST /speakers/{name}/samples/from-utterance/{id}` | claim a clip (+ its voice group) as samples |
| `POST /speakers/{name}/samples/from-latest` | claim the newest clip (+ group) — the LLM tool |
| `GET /pending/latest-voice` | `should_ask` verdict for the current unknown voice |
| `GET /history?limit=50` | recognition log |
| `GET /export` / `POST /import` | backup / restore people, samples, roles, settings |
| `POST /inference`, `POST /v1/audio/transcriptions` | HTTP transcription (whisper.cpp / OpenAI shape; needs a process with the model loaded — not the default Docker UI process) |

## Running from source

```bash
cp .env.example .env
UV_CACHE_DIR=/tmp/uv-cache uv venv && UV_CACHE_DIR=/tmp/uv-cache uv sync

# Wyoming server (Home Assistant talks to this)
UV_CACHE_DIR=/tmp/uv-cache uv run python -m transcribe_wyoming --uri tcp://0.0.0.0:10300 --language pl

# management UI/API (add --no-load-model to skip loading the ASR model)
UV_CACHE_DIR=/tmp/uv-cache uv run python server.py --host 0.0.0.0 --port 8580 --language pl

# tests
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest discover -s tests
```

Requirements: Python 3.11+, `uv`; GPU preferred, CPU fallback supported.

## Limitations

- no partial/streaming transcripts, no `zeroconf`
- no diarization: the whole utterance is attributed to one (dominant) voice
- requires a reachable whisper.cpp server (the STT engine runs there)

## License

Apache License 2.0. Based on the Wyoming protocol; see
[wyoming-faster-whisper](https://github.com/rhasspy/wyoming-faster-whisper).
