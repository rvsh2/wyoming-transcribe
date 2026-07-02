/* Wyoming Transcribe — Home Assistant custom panel.
 *
 * The full management UI (speakers, pending voices, recognition log,
 * settings, backup) rendered inside the HA frontend. All API calls go
 * through the integration's authenticated proxy
 * (/api/wyoming_transcribe/proxy/...), so the API token never reaches
 * the browser and port 8580 only needs to be reachable from the HA host.
 *
 * This panel is the primary UI; the server-side page on port 8580 stays
 * available as a frozen fallback for non-HA setups.
 */

const STYLES = `
  * { margin: 0; padding: 0; box-sizing: border-box; }
  :host {
    display: block;
    font-family: 'Inter', -apple-system, system-ui, sans-serif;
    background: linear-gradient(135deg, #0f172a 0%, #172033 52%, #1f2937 100%);
    color: #e0e0e0;
    min-height: 100%;
    padding: 0.65rem;
  }
  .container { width: min(1200px, 100%); margin: 0 auto; }
  h1 { font-size: 1.8rem; font-weight: 700; color: #fffffe; margin: 0.5rem 0 1rem; }
  .card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1.25rem; margin: 1rem 0;
  }
  .card h2 { color: #8fa3b8; margin-bottom: 0.5rem; font-size: 1.15rem; }
  code {
    background: rgba(143,163,184,0.14); color: #9ec7b8;
    padding: 2px 6px; border-radius: 4px; font-size: 0.9rem;
  }
  label { font-weight: 600; color: #fffffe; }
  select, input[type="text"], input[type="file"] {
    width: 100%; padding: 0.7rem 0.9rem;
    background: rgba(0,0,0,0.25); color: #fffffe;
    border: 1px solid rgba(255,255,255,0.12); border-radius: 10px;
  }
  button {
    border: 0; border-radius: 999px; padding: 0.7rem 1.2rem;
    background: #5f7389; color: #fffffe; font-weight: 700; cursor: pointer;
  }
  button:disabled { opacity: 0.6; cursor: not-allowed; }
  .button-row { display: flex; flex-wrap: wrap; gap: 0.6rem; align-items: center; }
  .button-secondary { background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12); }
  .button-danger { background: #a24c59; }
  .btn-small { padding: 0.45rem 0.85rem; font-size: 0.85rem; }
  .hint { color: #94a1b2; font-size: 0.92rem; }
  .form-row { display: grid; gap: 0.5rem; margin-top: 0.75rem; }
  .status-ok { color: #2cb67d; font-weight: 600; }
  .status-warn { color: #ffb4be; font-weight: 600; }
  .speaker-add { display: grid; grid-template-columns: 1fr auto; gap: 0.5rem; margin: 0.75rem 0; }
  .speaker-card {
    padding: 1rem; border-radius: 12px;
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    margin-top: 0.75rem;
  }
  .speaker-head { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap; }
  .speaker-name { font-weight: 700; color: #fffffe; font-size: 1.05rem; }
  .speaker-meta { color: #94a1b2; font-size: 0.85rem; }
  .sample-row {
    display: flex; align-items: center; gap: 0.6rem; padding: 0.45rem 0;
    border-top: 1px solid rgba(255,255,255,0.06); flex-wrap: wrap;
  }
  .sample-row audio { height: 36px; }
  .sample-meta { color: #94a1b2; font-size: 0.82rem; }
  .spacer { flex: 1; }
  .empty-hint { color: #94a1b2; font-size: 0.9rem; padding: 0.5rem 0; }
  .upload-label { display: inline-block; }
  .upload-label input { display: none; }
  .recording-live { color: #ffb4be; font-weight: 700; }
`;

class WyomingTranscribePanel extends HTMLElement {
  constructor() {
    super();
    this._hass = null;
    this._initialized = false;
    this._roles = ["admin", "user", "guest"];
    this._speakerNames = [];
    this._recorder = null;
    this._recorderStream = null;
    this._recorderChunks = [];
    this._recorderButton = null;
    this._recorderName = null;
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._initialized && this.isConnected) this._initialize();
  }

  connectedCallback() {
    if (!this._initialized && this._hass) this._initialize();
  }

  disconnectedCallback() {
    this._stopStream();
  }

  /* ------------------------------------------------------------------ api */

  async _api(path, options = {}) {
    const response = await this._hass.fetchWithAuth(
      `/api/wyoming_transcribe/proxy/${path}`,
      options
    );
    let payload = {};
    try { payload = await response.json(); } catch (e) { /* empty or binary */ }
    if (!response.ok) {
      throw new Error(payload.detail || `Request failed (${response.status})`);
    }
    return payload;
  }

  async _apiBlob(path) {
    const response = await this._hass.fetchWithAuth(
      `/api/wyoming_transcribe/proxy/${path}`
    );
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.blob();
  }

  /* ------------------------------------------------------------- rendering */

  _initialize() {
    this._initialized = true;
    const root = this.attachShadow({ mode: "open" });
    root.innerHTML = `
      <style>${STYLES}</style>
      <div class="container">
        <h1>Wyoming Transcribe</h1>

        <div class="card">
          <h2>Status serwera</h2>
          <div id="server-status" class="hint">Łączenie...</div>
        </div>

        <div class="card">
          <h2>Mówcy — rozpoznawanie głosu</h2>
          <p class="hint">Zdefiniuj osoby i dodaj próbki głosu (zalecane 10–30 s czystej mowy).
            Najprościej jednak przypisywać prawdziwe wypowiedzi z sekcji „Nierozpoznane głosy” poniżej.</p>
          <div class="speaker-add">
            <input id="new-speaker-name" type="text" placeholder="Imię nowej osoby" autocomplete="off">
            <button id="add-speaker" type="button">Dodaj osobę</button>
          </div>
          <div id="mic-hint" class="hint" style="display:none; padding: 0.5rem 0.75rem; border: 1px solid #d9822b; border-radius: 8px; margin-bottom: 0.5rem;">
            Nagrywanie mikrofonem wymaga dostępu do Home Assistant po <strong>HTTPS</strong>
            (tak samo jak mikrofon w Assist). Przy zwykłym http możesz wgrywać pliki
            albo przypisywać wypowiedzi z sekcji „Nierozpoznane głosy”.
          </div>
          <div id="speakers-status" class="hint"></div>
          <div id="speakers-list"></div>
        </div>

        <div class="card">
          <h2>Nierozpoznane głosy</h2>
          <p class="hint">Wypowiedzi, których głos nie pasował do żadnej osoby (jedna grupa = ten
            sam głos). Odsłuchaj i przypisz — domyślnie przypisywana jest cała grupa. Te same
            nagrania może przypisywać usługa <code>wyoming_transcribe.claim_utterance</code>.</p>
          <div id="pending-status" class="hint"></div>
          <div id="pending-list"></div>
        </div>

        <div class="card">
          <h2>Dziennik rozpoznań</h2>
          <p class="hint">Ostatnie transkrypcje z decyzją identyfikacji (kto + pewność).</p>
          <div id="history-status" class="hint"></div>
          <div id="history-list"></div>
          <div class="button-row form-row">
            <button id="history-refresh" class="button-secondary btn-small" type="button">Odśwież</button>
          </div>
        </div>

        <div class="card">
          <h2>Ustawienia</h2>
          <div class="form-row">
            <label for="speaker-text-mode">Przekazywanie tożsamości mówcy</label>
            <select id="speaker-text-mode">
              <option value="prefix">Tylko prefiks w tekście („Krzysztof: …”)</option>
              <option value="field">Tylko pole „speaker” w evencie Wyoming (czysty tekst)</option>
              <option value="both">Prefiks w tekście i pole „speaker”</option>
            </select>
            <div class="hint">Zmiana obowiązuje od następnej transkrypcji (bez restartu).</div>
            <div class="button-row">
              <button id="save-settings" type="button">Zapisz ustawienia</button>
            </div>
            <div id="settings-status" class="hint"></div>
          </div>
          <div class="form-row">
            <label>Kopia zapasowa głosów</label>
            <div class="button-row">
              <button id="export-backup" class="button-secondary" type="button">Pobierz kopię (tar.gz)</button>
              <label class="button-secondary upload-label" style="border-radius:999px;padding:0.7rem 1.2rem;font-weight:700;cursor:pointer;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);">
                Przywróć z kopii<input id="import-backup" type="file" accept=".tar.gz,.tgz,application/gzip">
              </label>
            </div>
            <div class="hint">Kopia obejmuje osoby, próbki, role i ustawienia
              (bez bufora nierozpoznanych i dziennika).</div>
            <div id="backup-status" class="hint"></div>
          </div>
        </div>
      </div>
    `;

    this._el = (id) => root.getElementById(id);

    this._el("add-speaker").addEventListener("click", () => this._addSpeaker());
    this._el("new-speaker-name").addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); this._addSpeaker(); }
    });
    this._el("history-refresh").addEventListener("click", () => this._loadHistory());
    this._el("save-settings").addEventListener("click", () => this._saveSettings());
    this._el("export-backup").addEventListener("click", () => this._exportBackup());
    this._el("import-backup").addEventListener("change", () => this._importBackup());

    if (!(window.isSecureContext && navigator.mediaDevices)) {
      this._el("mic-hint").style.display = "block";
    }

    this._loadStatus();
    this._loadSpeakers().then(() => this._loadPending()).then(() => this._loadHistory());
    this._loadSettings();
  }

  _escape(value) {
    const div = document.createElement("div");
    div.textContent = value == null ? "" : String(value);
    return div.innerHTML;
  }

  _audioPlayer(path) {
    const wrapper = document.createElement("span");
    const button = document.createElement("button");
    button.className = "button-secondary btn-small";
    button.type = "button";
    button.textContent = "▶ Odtwórz";
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const blob = await this._apiBlob(path);
        const audio = document.createElement("audio");
        audio.controls = true;
        audio.autoplay = true;
        audio.src = URL.createObjectURL(blob);
        wrapper.replaceChildren(audio);
      } catch (error) {
        button.disabled = false;
        button.textContent = `Błąd (${error.message})`;
      }
    });
    wrapper.appendChild(button);
    return wrapper;
  }

  /* --------------------------------------------------------------- status */

  async _loadStatus() {
    const target = this._el("server-status");
    try {
      const health = await this._api("health");
      const ready = health.ready
        ? '<span class="status-ok">● gotowy</span>'
        : '<span class="status-warn">● bez modelu ASR (tryb zarządzania)</span>';
      const speakerId = health.speaker_id || {};
      target.innerHTML =
        `${ready} · model <code>${this._escape(health.model)}</code>` +
        ` · rozpoznawanie mówców: ${speakerId.enabled ? "włączone" : "wyłączone"}` +
        (speakerId.speakers && speakerId.speakers.length
          ? ` (${this._escape(speakerId.speakers.join(", "))})`
          : "");
    } catch (error) {
      target.innerHTML = `<span class="status-warn">Brak połączenia: ${this._escape(error.message)}</span>`;
    }
  }

  /* -------------------------------------------------------------- speakers */

  _setSpeakersStatus(message, isError) {
    const target = this._el("speakers-status");
    target.textContent = message || "";
    target.style.color = isError ? "#ffb4be" : "#94a1b2";
  }

  async _loadSpeakers() {
    try {
      const data = await this._api("speakers");
      this._roles = data.roles || this._roles;
      const speakers = data.speakers || [];
      this._speakerNames = speakers.map((s) => s.name);
      const status = data.speaker_id || {};
      const enrolled = (status.enrolled || []).join(", ") || "brak";
      this._setSpeakersStatus(
        status.enabled
          ? `Rozpoznawanie włączone (próg ${status.threshold}). Osoby z próbkami: ${enrolled}.`
          : `Rozpoznawanie wyłączone (SPEAKER_ID_ENABLED=false). Osoby z próbkami: ${enrolled}.`,
        false
      );
      const list = this._el("speakers-list");
      list.innerHTML = "";
      if (!speakers.length) {
        list.innerHTML = '<div class="empty-hint">Brak zdefiniowanych osób. Dodaj pierwszą powyżej.</div>';
        return;
      }
      for (const speaker of speakers) list.appendChild(this._speakerCard(speaker));
    } catch (error) {
      this._setSpeakersStatus(`Błąd wczytywania: ${error.message}`, true);
    }
  }

  _speakerCard(speaker) {
    const card = document.createElement("div");
    card.className = "speaker-card";

    const head = document.createElement("div");
    head.className = "speaker-head";
    const totalSeconds = speaker.samples.reduce((sum, s) => sum + (s.seconds || 0), 0);
    const title = document.createElement("div");
    title.innerHTML =
      `<span class="speaker-name">${this._escape(speaker.name)}</span>` +
      `<div class="speaker-meta">${speaker.samples.length} próbek · ${totalSeconds.toFixed(1)} s łącznie</div>`;
    head.appendChild(title);

    const actions = document.createElement("div");
    actions.className = "button-row";

    const roleSelect = document.createElement("select");
    roleSelect.style.width = "auto";
    for (const role of this._roles) {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = `rola: ${role}`;
      roleSelect.appendChild(option);
    }
    roleSelect.value = speaker.role || "user";
    roleSelect.addEventListener("change", async () => {
      const form = new FormData();
      form.append("role", roleSelect.value);
      try {
        await this._api(`speakers/${encodeURIComponent(speaker.name)}/role`, { method: "POST", body: form });
        this._setSpeakersStatus(`Rola „${speaker.name}” ustawiona na ${roleSelect.value}.`, false);
      } catch (error) {
        this._setSpeakersStatus(`Nie udało się zmienić roli: ${error.message}`, true);
      }
    });
    actions.appendChild(roleSelect);

    const recordButton = document.createElement("button");
    recordButton.className = "button-secondary btn-small";
    recordButton.type = "button";
    if (window.isSecureContext && navigator.mediaDevices) {
      recordButton.textContent = "Nagraj";
      recordButton.addEventListener("click", () => this._toggleRecording(speaker.name, recordButton));
    } else {
      recordButton.textContent = "Nagraj (wymaga HTTPS)";
      recordButton.disabled = true;
      recordButton.title = "Mikrofon działa tylko przy dostępie do HA po HTTPS. Użyj wgrywania pliku lub sekcji „Nierozpoznane głosy”.";
    }
    actions.appendChild(recordButton);

    const uploadLabel = document.createElement("label");
    uploadLabel.className = "button-secondary btn-small upload-label";
    uploadLabel.style.cssText = "border-radius:999px;font-weight:700;cursor:pointer;padding:0.45rem 0.85rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);";
    uploadLabel.textContent = "Wgraj plik";
    const uploadInput = document.createElement("input");
    uploadInput.type = "file";
    uploadInput.accept = "audio/*";
    uploadInput.addEventListener("change", () => {
      if (uploadInput.files.length) this._uploadSample(speaker.name, uploadInput.files[0]);
    });
    uploadLabel.appendChild(uploadInput);
    actions.appendChild(uploadLabel);

    const deleteButton = document.createElement("button");
    deleteButton.className = "button-danger btn-small";
    deleteButton.type = "button";
    deleteButton.textContent = "Usuń osobę";
    deleteButton.addEventListener("click", async () => {
      if (!confirm(`Usunąć osobę "${speaker.name}" wraz ze wszystkimi próbkami?`)) return;
      try {
        await this._api(`speakers/${encodeURIComponent(speaker.name)}`, { method: "DELETE" });
        await this._loadSpeakers();
      } catch (error) {
        this._setSpeakersStatus(`Nie udało się usunąć: ${error.message}`, true);
      }
    });
    actions.appendChild(deleteButton);

    head.appendChild(actions);
    card.appendChild(head);

    if (!speaker.samples.length) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "Brak próbek dla tej osoby.";
      card.appendChild(empty);
    }
    for (const sample of speaker.samples) {
      const row = document.createElement("div");
      row.className = "sample-row";
      row.appendChild(this._audioPlayer(
        `speakers/${encodeURIComponent(speaker.name)}/samples/${encodeURIComponent(sample.id)}`
      ));
      const meta = document.createElement("span");
      meta.className = "sample-meta";
      meta.textContent = `${(sample.seconds || 0).toFixed(1)} s`;
      const spacer = document.createElement("div");
      spacer.className = "spacer";
      const del = document.createElement("button");
      del.className = "button-danger btn-small";
      del.type = "button";
      del.textContent = "Usuń";
      del.addEventListener("click", async () => {
        try {
          await this._api(
            `speakers/${encodeURIComponent(speaker.name)}/samples/${encodeURIComponent(sample.id)}`,
            { method: "DELETE" }
          );
          await this._loadSpeakers();
        } catch (error) {
          this._setSpeakersStatus(`Nie udało się usunąć próbki: ${error.message}`, true);
        }
      });
      row.appendChild(meta);
      row.appendChild(spacer);
      row.appendChild(del);
      card.appendChild(row);
    }
    return card;
  }

  async _addSpeaker() {
    const input = this._el("new-speaker-name");
    const name = input.value.trim();
    if (!name) {
      this._setSpeakersStatus("Podaj imię osoby.", true);
      return;
    }
    const form = new FormData();
    form.append("name", name);
    try {
      await this._api("speakers", { method: "POST", body: form });
      input.value = "";
      await this._loadSpeakers();
    } catch (error) {
      this._setSpeakersStatus(`Nie udało się dodać: ${error.message}`, true);
    }
  }

  async _uploadSample(name, file) {
    const form = new FormData();
    form.append("file", file);
    this._setSpeakersStatus(`Wgrywanie próbki dla ${name}...`, false);
    try {
      await this._api(`speakers/${encodeURIComponent(name)}/samples`, { method: "POST", body: form });
      this._setSpeakersStatus(`Dodano próbkę dla ${name}.`, false);
      await this._loadSpeakers();
    } catch (error) {
      this._setSpeakersStatus(`Nie udało się wgrać próbki: ${error.message}`, true);
    }
  }

  _recorderMime() {
    const types = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"];
    if (typeof MediaRecorder === "undefined") return "";
    return types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
  }

  _stopStream() {
    if (this._recorderStream) {
      this._recorderStream.getTracks().forEach((track) => track.stop());
      this._recorderStream = null;
    }
  }

  async _toggleRecording(name, button) {
    if (this._recorder && this._recorder.state === "recording") {
      this._recorder.stop();
      return;
    }
    try {
      this._recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this._recorderChunks = [];
      this._recorder = new MediaRecorder(
        this._recorderStream,
        this._recorderMime() ? { mimeType: this._recorderMime() } : undefined
      );
      this._recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size) this._recorderChunks.push(event.data);
      });
      this._recorder.addEventListener("stop", () => {
        const blob = this._recorderChunks.length
          ? new Blob(this._recorderChunks, { type: this._recorderChunks[0].type })
          : null;
        this._stopStream();
        button.textContent = "Nagraj";
        button.classList.remove("button-danger");
        if (blob) {
          this._uploadSample(name, new File([blob], "recording.webm", { type: blob.type }));
        }
      });
      this._recorder.start();
      button.textContent = "Zatrzymaj";
      button.classList.add("button-danger");
      this._setSpeakersStatus(`Nagrywanie dla ${name}... kliknij „Zatrzymaj”, by zapisać.`, false);
    } catch (error) {
      this._stopStream();
      this._setSpeakersStatus(`Brak dostępu do mikrofonu: ${error.message}`, true);
    }
  }

  /* --------------------------------------------------------------- pending */

  async _loadPending() {
    const status = this._el("pending-status");
    try {
      const data = await this._api("pending");
      this._renderPending(data.clusters || []);
    } catch (error) {
      status.textContent = `Błąd wczytywania: ${error.message}`;
    }
  }

  _renderPending(clusters) {
    const list = this._el("pending-list");
    const status = this._el("pending-status");
    list.innerHTML = "";
    const count = clusters.reduce((total, cluster) => total + cluster.clips.length, 0);
    status.textContent = count
      ? `${count} nagrań w ${clusters.length} grupach (grupa = ten sam głos).`
      : "Brak oczekujących nagrań — wszystkie głosy rozpoznane lub bufor pusty.";
    for (const cluster of clusters) list.appendChild(this._pendingCluster(cluster));
  }

  _assignSelect() {
    const select = document.createElement("select");
    select.style.width = "auto";
    for (const name of this._speakerNames) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    const fresh = document.createElement("option");
    fresh.value = "__new__";
    fresh.textContent = "+ nowa osoba…";
    select.appendChild(fresh);
    return select;
  }

  _resolveAssignTarget(select) {
    if (select.value === "__new__") {
      const name = prompt("Imię nowej osoby:");
      return name && name.trim() ? name.trim() : null;
    }
    return select.value || null;
  }

  async _claim(name, utteranceId, includeCluster) {
    const form = new FormData();
    form.append("include_cluster", includeCluster ? "true" : "false");
    const status = this._el("pending-status");
    try {
      const result = await this._api(
        `speakers/${encodeURIComponent(name)}/samples/from-utterance/${encodeURIComponent(utteranceId)}`,
        { method: "POST", body: form }
      );
      status.textContent = `Przypisano ${result.claimed.length} nagrań do „${name}”.`;
    } catch (error) {
      status.textContent = `Nie udało się przypisać: ${error.message}`;
    }
    await this._loadPending();
    await this._loadSpeakers();
    await this._loadHistory();
  }

  _pendingCluster(cluster) {
    const card = document.createElement("div");
    card.className = "speaker-card";

    const head = document.createElement("div");
    head.className = "speaker-head";
    const title = document.createElement("div");
    const clipCount = cluster.clips.length;
    title.innerHTML =
      `<span class="speaker-name">Nieznany głos</span>` +
      `<div class="speaker-meta">${clipCount} ${clipCount === 1 ? "nagranie" : "nagrań"}</div>`;
    head.appendChild(title);

    const actions = document.createElement("div");
    actions.className = "button-row";
    const target = this._assignSelect();
    const assignAll = document.createElement("button");
    assignAll.className = "btn-small";
    assignAll.type = "button";
    assignAll.textContent = "Przypisz grupę";
    assignAll.addEventListener("click", () => {
      const name = this._resolveAssignTarget(target);
      if (name) this._claim(name, cluster.clips[0].id, true);
    });
    actions.appendChild(target);
    actions.appendChild(assignAll);
    head.appendChild(actions);
    card.appendChild(head);

    for (const clip of cluster.clips) {
      const row = document.createElement("div");
      row.className = "sample-row";
      row.appendChild(this._audioPlayer(`pending/${encodeURIComponent(clip.id)}/audio`));
      const meta = document.createElement("span");
      meta.className = "sample-meta";
      const when = clip.created ? new Date(clip.created * 1000).toLocaleString() : "";
      const text = clip.text ? ` · „${clip.text}”` : "";
      meta.textContent = `${(clip.seconds || 0).toFixed(1)} s · ${when}${text}`;
      const spacer = document.createElement("div");
      spacer.className = "spacer";
      const assignOne = document.createElement("button");
      assignOne.className = "button-secondary btn-small";
      assignOne.type = "button";
      assignOne.textContent = "Tylko to";
      assignOne.addEventListener("click", () => {
        const name = this._resolveAssignTarget(target);
        if (name) this._claim(name, clip.id, false);
      });
      const remove = document.createElement("button");
      remove.className = "button-danger btn-small";
      remove.type = "button";
      remove.textContent = "Usuń";
      remove.addEventListener("click", async () => {
        try {
          await this._api(`pending/${encodeURIComponent(clip.id)}`, { method: "DELETE" });
        } catch (error) {
          this._el("pending-status").textContent = `Nie udało się usunąć: ${error.message}`;
        }
        await this._loadPending();
      });
      row.appendChild(meta);
      row.appendChild(spacer);
      row.appendChild(assignOne);
      row.appendChild(remove);
      card.appendChild(row);
    }
    return card;
  }

  /* --------------------------------------------------------------- history */

  async _loadHistory() {
    const status = this._el("history-status");
    try {
      const data = await this._api("history?limit=50");
      this._renderHistory(data.entries || []);
    } catch (error) {
      status.textContent = `Błąd wczytywania: ${error.message}`;
    }
  }

  _renderHistory(entries) {
    const list = this._el("history-list");
    const status = this._el("history-status");
    list.innerHTML = "";
    if (!entries.length) {
      status.textContent = "Brak wpisów — dziennik wypełnia się z każdą transkrypcją.";
      return;
    }
    const recognized = entries.filter((e) => e.speaker).length;
    status.textContent =
      `${entries.length} ostatnich transkrypcji · rozpoznane: ${recognized} · nieznane: ${entries.length - recognized}`;
    for (const entry of entries) {
      const row = document.createElement("div");
      row.className = "sample-row";
      const meta = document.createElement("span");
      meta.className = "sample-meta";
      const when = entry.ts ? new Date(entry.ts * 1000).toLocaleString() : "";
      let who = "nieznany";
      if (entry.speaker) {
        const score = entry.score != null ? ` (${entry.score.toFixed(2)})` : "";
        const role = entry.role ? ` · ${entry.role}` : "";
        who = `${entry.speaker}${score}${role}`;
      }
      meta.textContent = `${when} · ${who} · „${entry.text || ""}”`;
      const spacer = document.createElement("div");
      spacer.className = "spacer";
      row.appendChild(meta);
      row.appendChild(spacer);
      if (entry.utterance_id) {
        row.appendChild(this._audioPlayer(`pending/${encodeURIComponent(entry.utterance_id)}/audio`));
      }
      list.appendChild(row);
    }
  }

  /* -------------------------------------------------------------- settings */

  async _loadSettings() {
    try {
      const data = await this._api("settings");
      if (data.speaker_text_mode) {
        this._el("speaker-text-mode").value = data.speaker_text_mode;
      }
      this._el("settings-status").textContent = "";
    } catch (error) {
      this._el("settings-status").textContent = `Błąd wczytywania ustawień: ${error.message}`;
    }
  }

  async _saveSettings() {
    const form = new FormData();
    form.append("speaker_text_mode", this._el("speaker-text-mode").value);
    try {
      await this._api("settings", { method: "POST", body: form });
      this._el("settings-status").textContent = "Zapisano — obowiązuje od następnej transkrypcji.";
    } catch (error) {
      this._el("settings-status").textContent = `Błąd zapisu: ${error.message}`;
    }
  }

  async _exportBackup() {
    const status = this._el("backup-status");
    status.textContent = "Przygotowywanie kopii...";
    try {
      const blob = await this._apiBlob("export");
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "speakers-backup.tar.gz";
      link.click();
      URL.revokeObjectURL(link.href);
      status.textContent = "Kopia pobrana.";
    } catch (error) {
      status.textContent = `Błąd eksportu: ${error.message}`;
    }
  }

  async _importBackup() {
    const input = this._el("import-backup");
    const status = this._el("backup-status");
    if (!input.files.length) return;
    if (!confirm("Przywrócić kopię? Istniejące pliki o tych samych nazwach zostaną nadpisane.")) {
      input.value = "";
      return;
    }
    const form = new FormData();
    form.append("file", input.files[0]);
    try {
      const result = await this._api("import", { method: "POST", body: form });
      status.textContent = `Przywrócono ${result.files} plików.`;
      await this._loadSpeakers();
      await this._loadPending();
      await this._loadHistory();
    } catch (error) {
      status.textContent = `Błąd importu: ${error.message}`;
    }
    input.value = "";
  }
}

customElements.define("wyoming-transcribe-panel", WyomingTranscribePanel);
