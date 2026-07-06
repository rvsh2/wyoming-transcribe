# LLM agent recipes ("who are you?" flow and roles)

> **Note:** the literal `Speaker` prefix below must match your deployment's
> `SPEAKER_LABEL` (the shipped Polish compose sets `SPEAKER_LABEL=Mówca`,
> so replace `Speaker N:` with `Mówca N:` in every snippet).


Detailed recipes for wiring the speaker-identification features into a Home
Assistant LLM conversation agent.

## Voice-anchored enrollment ("who are you?")

When the dominant speaker of an utterance does not match any enrolled person,
the Wyoming process buffers that speaker's audio (min. `PENDING_MIN_SECONDS`,
default 0.6 s) in `<enrollment_dir>/.pending/` together with the transcript and
an ECAPA voiceprint. The `Transcript` event then carries an `utterance_id` (in
every text mode).

The conversation agent does not need any `utterance_id`: when the unknown
person answers with their name, that answer is itself buffered and becomes the
newest pending clip, and voice clustering ties it to everything else that
person said. Claiming "the newest clip + its voice cluster" therefore assigns
the *right* voice even if another person interjected in between. That is
exactly what `POST /speakers/{name}/samples/from-latest` and the HA service
`wyoming_transcribe.claim_latest` do (with a `max_age_seconds` guard,
default 300 s, refusing stale anchors).

To avoid interrogating every one-off visitor, the companion service
`wyoming_transcribe.check_latest_voice` (backed by
`GET /pending/latest-voice`) reports how much the *current unknown voice* has
already talked to the system (its cluster: utterance count, total seconds, age
of the newest clip) and returns a ready `should_ask` verdict — ask only
"regulars" (defaults: ≥ 3 utterances, ≥ 8 s of speech, newest clip ≤ 300 s
old).

Recipe — two scripts exposed to Assist as tools. Script names are arbitrary
(these match the reference deployment); write the descriptions and the system
prompt in your household's language — the LLM reads them:

```yaml
script:
  sprawdz_nieznany_glos:
    alias: "Check unrecognized voice"
    description: >-
      Call when an utterance is prefixed "Speaker N:" (unknown speaker).
      Returns should_ask — whether this person is worth asking who they
      are (only "regulars" qualify, not one-off visitors).
    sequence:
      - service: wyoming_transcribe.check_latest_voice
        response_variable: voice
      - stop: ""
        response_variable: voice

  przypisz_glos:
    alias: "Assign unrecognized voice to a person"
    description: >-
      Call after the unrecognized person introduces themselves by name.
      Pass the anchor_utterance_id returned by sprawdz_nieznany_glos —
      then exactly that voice is enrolled, even if someone else spoke
      in the meantime.
    fields:
      name:
        description: "Person's first name, e.g. Anna"
        required: true
      anchor_utterance_id:
        description: "utterance_id from sprawdz_nieznany_glos"
        required: false
    sequence:
      - service: wyoming_transcribe.claim_latest
        data:
          name: "{{ name }}"
          anchor_utterance_id: "{{ anchor_utterance_id | default('') }}"
```

System-prompt snippet (includes the anti-overzealousness rules):

```text
Utterances are prefixed with the speaker's name ("Krzysztof: ...") or with
"Speaker N:" when the voice is unrecognized. Rules for "Speaker N:":
1. Handle the request normally first.
2. Do not ask about identity for short utterances (fewer than ~5 words).
3. Before asking, call the sprawdz_nieznany_glos tool; ask only when
   should_ask is true. Remember the returned utterance_id. Never ask more
   than once per conversation.
4. Ask: "I don't recognize your voice — who are you? Please introduce
   yourself with a full sentence." The person whose voice was not
   recognized must introduce themselves personally.
5. When the person introduces themselves, call przypisz_glos with their
   name and that utterance_id as anchor_utterance_id.
```

Known limitations (by design): a very short answer (< ~0.6 s, e.g. just
"Anna") may not be buffered — the `max_age_seconds` guard then rejects the
claim instead of assigning a wrong clip, and the agent should ask again for a
full sentence. If a *different unknown* person answers on someone's behalf,
their voice would be enrolled under that name — hence the prompt asks the
person to introduce themselves; misassignments are visible and reversible in
the panel.

The `wyoming_transcribe_new_pending` event carries `voice_utterances`
(cluster size), so a notification automation can likewise alert only about
regulars (condition: `{{ trigger.event.data.voice_utterances >= 3 }}`).

## Using roles in practice

Every enrolled person has a role: `admin`, `user` (default) or `guest`. The
STT server only *labels* who spoke; enforcing what each role may do belongs in
your conversation agent / voice pipeline. Unrecognized voices carry no role
(`speaker_role: null`).

**Pattern 1 — `prefix` mode (default), policy in the agent prompt.** The role
is *not* in the text — only the name is (`Krzysztof: zgaś światło`). Keep the
role policy in your LLM agent's system prompt, keyed by name:

```text
Utterances are prefixed with the speaker's name ("Krzysztof: ...") or with
"Speaker N:" when the voice is unrecognized. Rules:
- Krzysztof (admin): full control of the house, including locks, alarm and
  configuration.
- Anna (user): lights, music and temperature; no locks or alarm.
- guests / "Speaker N": answer informational questions only, perform no actions.
When a request exceeds the speaker's permissions, refuse and say why.
```

Simple and works today; the cost is updating the prompt when people change.

**Pattern 2 — `field`/`both` mode, policy keyed by role.** A custom pipeline
component (or an agent that receives the Wyoming event data) reads `speaker`
and `speaker_role` from the `Transcript` event and injects one line into the
LLM context, e.g. `speaker: Krzysztof (role: admin)`. The prompt then needs
only the per-role policy, not a per-person list:

```text
The context contains "speaker: <name> (role: <role>)". Rules by role:
- admin: all actions; user: no locks/alarm/configuration;
- guest or no role: informational answers only, no actions.
```

**Pattern 3 — automations keyed on role.** The `enrolled_speakers` sensor
exposes a `roles` attribute (name → role), and an agent tool can also
`GET /speakers` to check a role dynamically before performing a sensitive
action.

**Security note:** a voice can be imitated or replayed; treat roles as
convenience authorization for everyday comfort (lights, media, blinds), not as
strong authentication. Critical actions (door locks, alarm, purchases) should
require a second factor regardless of role — e.g. a PIN in the conversation or
confirmation from a companion-app notification.
