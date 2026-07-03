# LLM agent recipes ("who are you?" flow and roles)

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
`cohere_transcribe_diarize.claim_latest` do (with a `max_age_seconds` guard,
default 300 s, refusing stale anchors).

To avoid interrogating every one-off visitor, the companion service
`cohere_transcribe_diarize.check_latest_voice` (backed by
`GET /pending/latest-voice`) reports how much the *current unknown voice* has
already talked to the system (its cluster: utterance count, total seconds, age
of the newest clip) and returns a ready `should_ask` verdict — ask only
"regulars" (defaults: ≥ 3 utterances, ≥ 8 s of speech, newest clip ≤ 300 s
old).

Recipe — two scripts exposed to Assist as tools:

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

Known limitations (by design): a very short answer (< ~0.6 s, e.g. just
"Anna") may not be buffered — the `max_age_seconds` guard then rejects the
claim instead of assigning a wrong clip, and the agent should ask again for a
full sentence. If a *different unknown* person answers on someone's behalf,
their voice would be enrolled under that name — hence the prompt asks the
person to introduce themselves; misassignments are visible and reversible in
the panel.

The `cohere_transcribe_diarize_new_pending` event carries `voice_utterances`
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
Wypowiedzi mają prefiks z imieniem mówcy ("Krzysztof: ...") albo "Mówca N:" gdy
głos jest nierozpoznany. Zasady:
- Krzysztof (admin): pełna kontrola domu, w tym zamki, alarm i konfiguracja.
- Anna (user): sterowanie światłem, muzyką i temperaturą; bez zamków i alarmu.
- goście / "Mówca N": odpowiadaj tylko na pytania informacyjne, nie wykonuj akcji.
Gdy wypowiedź przekracza uprawnienia mówcy, odmów i powiedz dlaczego.
```

Simple and works today; the cost is updating the prompt when people change.

**Pattern 2 — `field`/`both` mode, policy keyed by role.** A custom pipeline
component (or an agent that receives the Wyoming event data) reads `speaker`
and `speaker_role` from the `Transcript` event and injects one line into the
LLM context, e.g. `mówca: Krzysztof (rola: admin)`. The prompt then needs only
the per-role policy, not a per-person list:

```text
Kontekst zawiera "mówca: <imię> (rola: <rola>)". Zasady wg roli:
- admin: wszystkie akcje; user: bez zamków/alarmu/konfiguracji;
- guest lub brak roli: tylko odpowiedzi informacyjne, żadnych akcji.
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
