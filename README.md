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
| `GET /history?limit=50` | recognition log: recent transcriptions with who/score/role or `utterance_id` |
| `GET /export` / `POST /import` | download / restore a tar.gz backup of people, samples, roles and settings |

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

> **Upgrade note (2026-07-02):** port `8580` (management API/UI) is now published
> on `127.0.0.1` by default for security. If you previously reached the UI from
> another machine — or use the HACS integration from a separate HA host — set
> `UI_BIND=0.0.0.0` **and** `API_TOKEN=<secret>` in `.env`, then
> `docker compose up -d`. A "connection refused" on 8580 after upgrading means
> exactly this.

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

This repo ships a Home Assistant custom integration (`custom_components/cohere_transcribe_diarize`)
with the **full management UI as a native HA sidebar panel** (speakers with roles and
sample recording, unrecognized voices, recognition log, settings, backup) plus status
sensors (model status, enrolled speakers, pending unrecognized voices).

Since 0.3.0 the panel is not an iframe: it is rendered by the HA frontend and talks to
the server through an **authenticated proxy** (`/api/cohere_transcribe_diarize/proxy/...`,
admin-only). Consequences:

- the API token lives only in the integration's config — never in the browser;
- port `8580` must be reachable **from the HA host only**, not from every browser;
- the panel works remotely (Nabu Casa / external URL) like any other HA page;
- microphone recording in the panel needs HA served over **HTTPS** (same browser rule
  as Assist's microphone); on plain http the panel says so and you can still upload
  files or assign real utterances from the pending section.

The panel is the primary UI going forward; the server-side page on port `8580` stays
as a frozen fallback for non-HA setups.

The integration also gives automations and LLM tools native HA hooks:

- **Event `cohere_transcribe_diarize_new_pending`** — fired when a new unrecognized voice
  lands in the pending buffer (data: `utterance_id`, `text`, `seconds`, `created`).
  Example automation: send an actionable notification "Nowy głos: *zgaś światło* —
  kto to?" with buttons that call the claim service per household member.
- **Service `cohere_transcribe_diarize.claim_latest`** (`name`, optional `include_cluster`,
  `max_age_seconds`) — voice-anchored enrollment of the newest unrecognized
  utterance; the LLM tool for the "who are you?" flow (no utterance_id needed).
- **Service `cohere_transcribe_diarize.claim_utterance`** (`name`, `utterance_id`,
  `include_cluster` default true) — like above but for an explicit clip, e.g. from
  the `cohere_transcribe_diarize_new_pending` event data.
- **Service `cohere_transcribe_diarize.set_role`** (`name`, `role`: admin/user/guest).

Note: pending clips are polled every 15 s, so the event can lag up to 15 s
behind the utterance.

Install:

1. On the server: set `API_TOKEN=<secret>` and `UI_BIND=0.0.0.0` in `.env`, then
   `docker compose up -d` (never expose port 8580 without a token; if HA runs on the
   same host, `UI_BIND` can stay `127.0.0.1`).
2. HACS → Integrations → ⋮ → **Custom repositories** → add this repo URL, category
   *Integration* → install **Wyoming Transcribe** → restart HA.
3. Settings → Devices & services → **Add integration** → *Wyoming Transcribe* → enter
   the Docker host's address, port `8580` and the API token. That is the only place
   the token is ever entered.
4. A "Wyoming Transcribe" entry appears in the sidebar (admin-only) with the full
   management UI.

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

### Profile adaptation (recognition improves with use)

Every **confident** recognition (score ≥ `SPEAKER_ADAPT_MIN_SCORE`, default 0.60)
folds the utterance's voiceprint into the person's adaptive vector (running mean,
responsiveness capped at 50 utterances, stored in `<person>/.adapt.json`). Matching
then uses the enrolled-sample mean blended **1:1** with the adaptive vector, so:

- profiles track real usage conditions (far-field mics, rooms, voice drift, colds)
  and false negatives shrink over time;
- the enrolled samples remain an immovable anchor — adaptation can pull a profile
  only halfway, and an ambiguous utterance (another profile within 0.10 of the
  best score) is never used, so profiles cannot be poisoned or drift into each
  other;
- deleting the person (or their `.adapt.json`) resets adaptation; backups include it.

Disable with `SPEAKER_ADAPT_ENABLED=false`. The speaker card in the panel shows how
many recognitions have fed each profile ("adaptacja: N rozpoznań").

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
(their concatenated segments, min. `PENDING_MIN_SECONDS`, default 0.6 s) in
`<enrollment_dir>/.pending/` together with the transcript and an ECAPA voiceprint.
The `Transcript` event then carries an `utterance_id` (in **every** text mode).

Three ways to resolve a pending voice:

1. **UI (manual verification)** — the "Nierozpoznane głosy" card lists pending clips
   *grouped by voice* (one group = one person, matched by voiceprint similarity,
   `PENDING_CLUSTER_THRESHOLD`, default 0.40). Play each clip, check who it is, and
   assign the group to an existing or new person. Assigning a group moves all its
   clips to that person as enrollment samples at once, so the profile is immediately
   built from several samples instead of one weak clip.
2. **LLM pipeline ("who are you?") — the voice-anchored flow.** The conversation
   agent does not need any `utterance_id`: when the unknown person answers with
   their name, that answer is itself buffered and becomes the newest pending clip,
   and the voice clustering ties it to everything else that person said. Claiming
   "the newest clip + its voice cluster" therefore assigns the *right* voice even
   if another person interjected in between. That is exactly what
   `POST /speakers/{name}/samples/from-latest` and the HA service
   `cohere_transcribe_diarize.claim_latest` do (with a `max_age_seconds` guard, default
   300 s, refusing stale anchors).

   To avoid interrogating every one-off visitor, the companion service
   `cohere_transcribe_diarize.check_latest_voice` (backed by `GET /pending/latest-voice`)
   reports how much the *current unknown voice* has already talked to the system
   (its cluster: utterance count, total seconds, age of the newest clip) and
   returns a ready `should_ask` verdict — ask only "regulars" (defaults: ≥ 3
   utterances, ≥ 8 s of speech, newest clip ≤ 300 s old).

   Recipe for a Home Assistant LLM conversation agent — two scripts exposed to
   Assist as tools:

   ```yaml
   script:
     sprawdz_nieznany_glos:
       alias: "Sprawdź nierozpoznany głos"
       description: >-
         Wywołaj, gdy wypowiedź ma prefiks "Mówca N:". Zwraca should_ask —
         czy warto zapytać tę osobę, kim jest (pyta tylko "bywalców", nie
         jednorazowych gości).
       sequence:
         - service: cohere_transcribe_diarize.check_latest_voice
           response_variable: voice
         - stop: ""
           response_variable: voice

     przypisz_glos:
       alias: "Przypisz nierozpoznany głos do osoby"
       description: >-
         Wywołaj po tym, jak nierozpoznana osoba przedstawi się z imienia.
         Podaj anchor_utterance_id zwrócone przez sprawdz_nieznany_glos —
         wtedy przypisywany jest dokładnie ten głos, nawet jeśli w
         międzyczasie odezwał się ktoś inny.
       fields:
         name:
           description: "Imię osoby, np. Anna"
           required: true
         anchor_utterance_id:
           description: "utterance_id ze sprawdz_nieznany_glos"
           required: false
       sequence:
         - service: cohere_transcribe_diarize.claim_latest
           data:
             name: "{{ name }}"
             anchor_utterance_id: "{{ anchor_utterance_id | default('') }}"
   ```

   System-prompt snippet (includes the anti-overzealousness rules):

   ```text
   Wypowiedzi mają prefiks z imieniem mówcy ("Krzysztof: ...") albo "Mówca N:",
   gdy głos jest nierozpoznany. Zasady dla "Mówca N:":
   1. Najpierw normalnie obsłuż polecenie.
   2. Nie pytaj o tożsamość przy krótkich wypowiedziach (mniej niż ~5 słów).
   3. Zanim zapytasz, wywołaj narzędzie sprawdz_nieznany_glos; pytaj tylko gdy
      should_ask jest true. Zapamiętaj zwrócone utterance_id. Nie pytaj
      częściej niż raz na rozmowę.
   4. Pytaj: "Nie rozpoznaję Twojego głosu — kim jesteś? Przedstaw się pełnym
      zdaniem." Proś, by przedstawiła się sama osoba, której głosu nie rozpoznano.
   5. Gdy osoba się przedstawi, wywołaj przypisz_glos z jej imieniem i tym
      utterance_id jako anchor_utterance_id.
   ```

   Known limitations (by design): a very short answer (< ~0.6 s, e.g. just "Anna")
   may not be buffered — the `max_age_seconds` guard then rejects the claim instead
   of assigning a wrong clip, and the agent should ask again for a full sentence.
   If a *different unknown* person answers on someone's behalf, their voice would
   be enrolled under that name — hence the prompt asks the person to introduce
   themselves; misassignments are visible and reversible in the panel.

   The `cohere_transcribe_diarize_new_pending` event carries `voice_utterances` (cluster
   size), so a notification automation can likewise alert only about regulars
   (condition: `{{ trigger.event.data.voice_utterances >= 3 }}`).
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
- `role` per person in `GET /speakers` (and in the `enrolled_speakers` sensor's
  `roles` attribute in the HA integration).

**What a role does by itself: nothing.** The STT server only *labels* who spoke;
it has no idea what "locks" or "lights" are in your Home Assistant. Deciding what
each role may do — and enforcing it — belongs where actions are executed: in your
conversation agent / voice pipeline. Unrecognized voices carry no role at all
(`speaker_role: null`) and are handled like any other utterance.

#### How to use roles in practice

**Pattern 1 — `prefix` mode (default), policy in the agent prompt.** In this mode
the role is *not* in the text — only the name is (`Krzysztof: zgaś światło`). Keep
the role policy in your LLM agent's system prompt, keyed by name:

```text
Wypowiedzi mają prefiks z imieniem mówcy ("Krzysztof: ...") albo "Mówca N:" gdy
głos jest nierozpoznany. Zasady:
- Krzysztof (admin): pełna kontrola domu, w tym zamki, alarm i konfiguracja.
- Anna (user): sterowanie światłem, muzyką i temperaturą; bez zamków i alarmu.
- goście / "Mówca N": odpowiadaj tylko na pytania informacyjne, nie wykonuj akcji.
Gdy wypowiedź przekracza uprawnienia mówcy, odmów i powiedz dlaczego.
```

Simple and works today; the cost is updating the prompt when people change.

**Pattern 2 — `field`/`both` mode, policy keyed by role.** A custom pipeline
component (or an agent that receives the Wyoming event data) reads `speaker` and
`speaker_role` from the `Transcript` event and injects one line into the LLM
context, e.g. `mówca: Krzysztof (rola: admin)`. The prompt then needs only the
per-role policy (three lines), not a per-person list:

```text
Kontekst zawiera "mówca: <imię> (rola: <rola>)". Zasady wg roli:
- admin: wszystkie akcje; user: bez zamków/alarmu/konfiguracji;
- guest lub brak roli: tylko odpowiedzi informacyjne, żadnych akcji.
```

**Pattern 3 — automations keyed on role.** The `enrolled_speakers` sensor exposes a
`roles` attribute (name → role), and an agent tool can also `GET /speakers` to check
a role dynamically before performing a sensitive action.

**Security note:** a voice can be imitated or replayed; treat roles as convenience
authorization for everyday comfort (lights, media, blinds), not as strong
authentication. Critical actions (door locks, alarm, purchases) should require a
second factor regardless of role — e.g. a PIN in the conversation or confirmation
from a companion-app notification.

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
