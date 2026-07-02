# Release Notes

## 2026-07-02 (v0.4.0)

### Voice-anchored "who are you?" enrollment for LLM agents

- New `POST /speakers/{name}/samples/from-latest` and HA service
  `wyoming_transcribe.claim_latest` (`name`, optional `include_cluster`,
  `max_age_seconds` default 300): claims the newest pending clip and its voice
  cluster. The unknown person's *answer* ("jestem Anna") is itself buffered and
  anchors the claim, so the voice matching (clustering) picks the right person
  even when another voice interjected — and no utterance_id has to travel through
  the pipeline. A stale-anchor guard (409) prevents claiming an unrelated clip
  when the answer was too short to buffer.
- `PENDING_MIN_SECONDS` default lowered 1.0 → 0.6 s so short introduction answers
  are buffered (ECAPA embedding minimum is 0.4 s).
- README: ready-to-use HA script (Assist-exposed tool) + system-prompt snippet for
  the flow, with documented limitations (very short answers; third-person answers).
- HA automation shipped separately in the user's HA: UI notification on
  `wyoming_transcribe_new_pending` as a fallback when nobody answers.

## 2026-07-02 (v0.3.0)

### Native HA panel (no more iframe)

- The full management UI (speakers with roles and mic recording, unrecognized
  voices, recognition log, settings, enrollment backup) is now a native HA custom
  panel (`frontend/panel.js`) rendered by the Home Assistant frontend.
- All panel traffic goes through a new authenticated, admin-only proxy view
  (`/api/wyoming_transcribe/proxy/{path}`) that adds the API token server-side:
  the token never reaches the browser, port 8580 only needs to be reachable from
  the HA host, and the panel works remotely (Nabu Casa/external URL).
- Microphone recording is enabled in secure contexts (HTTPS), with a visible
  notice on plain http (same browser rule as Assist's microphone); file upload
  and pending-voice assignment work regardless.
- The server-side page on port 8580 remains as a frozen fallback for non-HA
  setups; the HA panel is the primary UI going forward. Integration 0.3.0.

## 2026-07-02 (v0.2.0)

### Home Assistant: services + event for the "who are you?" flow

- Integration services `wyoming_transcribe.claim_utterance` (name, utterance_id,
  include_cluster) and `wyoming_transcribe.set_role` — automations and LLM tools can
  enroll pending voices and manage roles natively, without curl/tokens in prompts.
- Event `wyoming_transcribe_new_pending` fired when a new unrecognized voice lands in
  the buffer (polled every 60 s); integration bumped to 0.2.0.

### Recognition log ("Dziennik rozpoznań")

- Every transcription appends who/score/role (or the pending `utterance_id`) to a
  shared JSONL ring log (`HISTORY_MAX_ENTRIES`, default 200; `HISTORY_ENABLED`);
  `GET /history` + a UI card with per-entry playback of still-pending clips. Makes
  threshold tuning and misidentification review data-driven instead of guesswork.

### Enrollment backup

- `GET /export` downloads a tar.gz of people, samples, roles and settings (pending
  buffer and the operational log are excluded); `POST /import` restores it with
  archive-path validation. Both wired into the UI settings card. Live-verified:
  delete person -> voice unknown; import -> recognition (0.85) returns.

### Emotion spike (negative result)

- The tokenizer has `<|emo:angry/happy/neutral/sad/undefined|>` tokens, but the
  diarize fine-tune predicts `emo:undefined` with p=1.0 regardless of input — the
  capability is dead in this checkpoint, so no `emotion` field was added.

## 2026-07-02

### QA review fixes (10 findings, multi-agent review)

- **Security**: the unauthenticated management API (port `8580`) is published on
  `127.0.0.1` only; README documents SSH tunneling for remote UI access.
- **Speaker misattribution in recordings > 30 s**: diarization speaker indices restart
  in every 30 s window; they are now remapped onto one global speaker space by ECAPA
  voiceprint matching (`SPEAKER_CHAIN_THRESHOLD`, default `0.40`). Without a confident
  match a window's speaker gets a fresh index — never silently merged.
- **Event-loop blocking**: model inference runs via `asyncio.to_thread` in both the
  Wyoming handler and the FastAPI endpoints (`/inference`, `/v1/audio/transcriptions`,
  `/load`), serialized by an inference lock; concurrent satellites and health probes
  stay responsive during transcription.
- **Robustness**: the Wyoming finalize path always answers with a `Transcript` (empty
  on error) instead of dying silently; HA's pipeline no longer hangs until timeout.
- **Container supervision**: `docker-entrypoint.sh` supervises both processes (the old
  EXIT trap was dead code after `exec`); if either dies the container exits and
  Docker's restart policy brings the pair back. The compose healthcheck is restored.
- **Silent transcript truncation**: hitting the 400-token generation cap is detected;
  the window is retried as two shorter ones (warning logged).
- **UI**: the Transcribe form is disabled with an explanatory notice when the process
  runs with `--no-load-model` (it always returned 503 in the Docker deployment).
- **SRT/VTT**: one cue per diarized segment with speaker labels and real timestamps
  (previously a single whole-file cue).
- `requirements.txt` synced with `pyproject.toml` (adds `torchaudio`, `silero-vad`,
  `speechbrain`).
- Dockerfile installs dependencies before copying source — code changes no longer
  invalidate the torch-sized layer (rebuild: seconds instead of minutes).

### Speaker identity for the HA pipeline

- New `SPEAKER_TEXT_MODE` (`prefix` | `field` | `both`, default `prefix`), switchable
  at runtime from the new UI settings card; shared with the Wyoming process via
  `.settings.json` in the enrollment dir, applied on the next transcription.
- In `field`/`both` mode the Wyoming `Transcript` event carries `speaker` (enrolled
  name of the dominant speaker or `null`) and `speaker_score`; `verbose_json` exposes
  them in all modes.
- Identification is now done once per diarized speaker on the concatenation of all
  their segments (voice commands are often split into sub-0.4 s segments too short
  for a voiceprint on their own), and the dominant speaker (most speech time) is
  reported per utterance.

### Management API auth

- Optional `API_TOKEN` env: when set, every endpoint except `/` and `/health` requires
  `X-API-Token` or `Authorization: Bearer`; the UI stores the token in the browser.

### Unknown voices & "who are you?" enrollment

- Unrecognized dominant speakers are buffered in `<enrollment_dir>/.pending/` (clip +
  transcript + ECAPA voiceprint; ring buffer `PENDING_MAX_CLIPS`, min length
  `PENDING_MIN_SECONDS`); the Transcript event carries `utterance_id` in every mode.
- `GET /pending` groups clips by voice (greedy voiceprint clustering,
  `PENDING_CLUSTER_THRESHOLD`); `POST /speakers/{name}/samples/from-utterance/{id}`
  claims a clip — with `include_cluster=true` (default) the whole voice group — as
  enrollment samples, auto-creating the person. This is the LLM tool for the
  "who are you?" flow.
- New UI card "Nierozpoznane głosy": listen to pending clips, verify who speaks,
  assign a group to a person, or delete. Sample/pending playback now fetches with
  auth headers (works with API_TOKEN enabled).

### Speaker roles

- Each person has a role (`admin`/`user`/`guest`, default `user`; `.meta.json` in the
  person's dir), set from the UI or `POST /speakers/{name}/role`. The recognized
  dominant speaker's role is delivered as `speaker_role` in the Wyoming event
  (field/both modes) and in `verbose_json`.

### Home Assistant integration (HACS)

- New custom integration `custom_components/wyoming_transcribe`: config flow
  (host/port/token), the management UI embedded as an admin-only sidebar iframe
  panel, and sensors (model status, enrolled speakers, pending voices).
- `hacs.json` added; compose supports `UI_BIND` (default loopback; set `0.0.0.0`
  together with `API_TOKEN` to expose the UI for HA/browsers).

### Startup

- The full request path is warmed at startup: audio conversion/resample (librosa/soxr
  lazy init — measured 24 s in the container), Silero VAD load, and a full-length 30 s
  generate pass. First request after a cold start: **0.7 s** (previously ~25 s).

### Verification

- `93` unit tests pass (new suites: settings store, window merging, truncation split,
  token auth, speaker-field events).
- Live-tested on `docker compose up` (RTX 3090): cross-window speaker chaining on 50 s
  audio, `describe` answered in 0.00 s during a 29 s transcription, container restart
  after killing the UI process, mode switching without restart, token auth.

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

### Second review pass

- Validate diarize structural tokens at model load (fail fast at startup) instead of crashing the first request; the language token now falls back to the configured default language, then English.
- `verbose_json` segments now expose the identified speaker `name`/`score`.
- `embed_batch` caps clips at 15 s to bound padded-batch memory, and `embed` delegates to it so enrollment and match-time embeddings are computed identically.
- `/speakers` reports enrolled people via the `enrolled` field (the registry's `speakers` list stays for the Wyoming process).

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
