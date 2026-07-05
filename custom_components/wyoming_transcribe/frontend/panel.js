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
          <h2>Server Status</h2>
          <div id="server-status" class="hint">Connecting...</div>
        </div>

        <div class="card">
          <h2>Speakers — Voice Recognition</h2>
          <p class="hint">Define people and add voice samples (10–30 s of clean speech recommended).
            The easiest way, though, is to assign real utterances from the "Unrecognized Voices" section below.</p>
          <div class="speaker-add">
            <input id="new-speaker-name" type="text" placeholder="New person's name" autocomplete="off">
            <button id="add-speaker" type="button">Add person</button>
          </div>
          <div id="mic-hint" class="hint" style="display:none; padding: 0.5rem 0.75rem; border: 1px solid #d9822b; border-radius: 8px; margin-bottom: 0.5rem;">
            Microphone recording requires accessing Home Assistant over <strong>HTTPS</strong>
            (same as the Assist microphone). Over plain http you can upload files
            or assign utterances from the "Unrecognized Voices" section.
          </div>
          <div id="speakers-status" class="hint"></div>
          <div id="speakers-list"></div>
        </div>

        <div class="card">
          <h2>Unrecognized Voices</h2>
          <p class="hint">Utterances whose voice did not match any known person (one group = the
            same voice). Listen and assign — by default the whole group is assigned. The same
            recordings can be assigned by the <code>wyoming_transcribe.claim_utterance</code> service.</p>
          <div id="pending-status" class="hint"></div>
          <div id="pending-list"></div>
        </div>

        <div class="card">
          <h2>Recognition Log</h2>
          <p class="hint">Recent transcriptions with the identification decision (who + confidence).</p>
          <div id="history-status" class="hint"></div>
          <div id="history-list"></div>
          <div class="button-row form-row">
            <button id="history-refresh" class="button-secondary btn-small" type="button">Refresh</button>
          </div>
        </div>

        <div class="card">
          <h2>Settings</h2>
          <div class="form-row">
            <label for="speaker-text-mode">Speaker identity delivery</label>
            <select id="speaker-text-mode">
              <option value="prefix">Text prefix only ("Krzysztof: …")</option>
              <option value="field">Only the "speaker" field in the Wyoming event (plain text)</option>
              <option value="both">Text prefix and the "speaker" field</option>
            </select>
            <div class="hint">Changes take effect from the next transcription (no restart needed).</div>
            <div class="button-row">
              <button id="save-settings" type="button">Save settings</button>
            </div>
            <div id="settings-status" class="hint"></div>
          </div>
          <div class="form-row">
            <label>Voice Backup</label>
            <div class="button-row">
              <button id="export-backup" class="button-secondary" type="button">Download backup (tar.gz)</button>
              <label class="button-secondary upload-label" style="border-radius:999px;padding:0.7rem 1.2rem;font-weight:700;cursor:pointer;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);">
                Restore from backup<input id="import-backup" type="file" accept=".tar.gz,.tgz,application/gzip">
              </label>
            </div>
            <div class="hint">The backup includes people, samples, roles and settings
              (excluding the unrecognized-voice buffer and the log).</div>
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
    button.textContent = "▶ Play";
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
        button.textContent = `Error (${error.message})`;
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
        ? '<span class="status-ok">● ready</span>'
        : '<span class="status-warn">● no ASR model (management mode)</span>';
      const speakerId = health.speaker_id || {};
      target.innerHTML =
        `${ready} · model <code>${this._escape(health.model)}</code>` +
        ` · speaker recognition: ${speakerId.enabled ? "enabled" : "disabled"}` +
        (speakerId.speakers && speakerId.speakers.length
          ? ` (${this._escape(speakerId.speakers.join(", "))})`
          : "");
    } catch (error) {
      target.innerHTML = `<span class="status-warn">No connection: ${this._escape(error.message)}</span>`;
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
      const enrolled = (status.enrolled || []).join(", ") || "none";
      this._setSpeakersStatus(
        status.enabled
          ? `Recognition enabled (threshold ${status.threshold}). People with samples: ${enrolled}.`
          : `Recognition disabled (SPEAKER_ID_ENABLED=false). People with samples: ${enrolled}.`,
        false
      );
      const list = this._el("speakers-list");
      list.innerHTML = "";
      if (!speakers.length) {
        list.innerHTML = '<div class="empty-hint">No people defined yet. Add the first one above.</div>';
        return;
      }
      for (const speaker of speakers) list.appendChild(this._speakerCard(speaker));
    } catch (error) {
      this._setSpeakersStatus(`Failed to load: ${error.message}`, true);
    }
  }

  _speakerCard(speaker) {
    const card = document.createElement("div");
    card.className = "speaker-card";

    const head = document.createElement("div");
    head.className = "speaker-head";
    const totalSeconds = speaker.samples.reduce((sum, s) => sum + (s.seconds || 0), 0);
    const adapted = speaker.adapted
      ? ` · adaptation: ${speaker.adapted} recognitions`
      : "";
    const title = document.createElement("div");
    title.innerHTML =
      `<span class="speaker-name">${this._escape(speaker.name)}</span>` +
      `<div class="speaker-meta">${speaker.samples.length} samples · ${totalSeconds.toFixed(1)} s total${adapted}</div>`;
    head.appendChild(title);

    const actions = document.createElement("div");
    actions.className = "button-row";

    const roleSelect = document.createElement("select");
    roleSelect.style.width = "auto";
    for (const role of this._roles) {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = `role: ${role}`;
      roleSelect.appendChild(option);
    }
    roleSelect.value = speaker.role || "user";
    roleSelect.addEventListener("change", async () => {
      const form = new FormData();
      form.append("role", roleSelect.value);
      try {
        await this._api(`speakers/${encodeURIComponent(speaker.name)}/role`, { method: "POST", body: form });
        this._setSpeakersStatus(`Role for "${speaker.name}" set to ${roleSelect.value}.`, false);
      } catch (error) {
        this._setSpeakersStatus(`Failed to change role: ${error.message}`, true);
      }
    });
    actions.appendChild(roleSelect);

    const recordButton = document.createElement("button");
    recordButton.className = "button-secondary btn-small";
    recordButton.type = "button";
    if (window.isSecureContext && navigator.mediaDevices) {
      recordButton.textContent = "Record";
      recordButton.addEventListener("click", () => this._toggleRecording(speaker.name, recordButton));
    } else {
      recordButton.textContent = "Record (requires HTTPS)";
      recordButton.disabled = true;
      recordButton.title = "The microphone only works when HA is accessed over HTTPS. Use file upload or the \"Unrecognized Voices\" section.";
    }
    actions.appendChild(recordButton);

    const uploadLabel = document.createElement("label");
    uploadLabel.className = "button-secondary btn-small upload-label";
    uploadLabel.style.cssText = "border-radius:999px;font-weight:700;cursor:pointer;padding:0.45rem 0.85rem;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.12);";
    uploadLabel.textContent = "Upload file";
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
    deleteButton.textContent = "Delete person";
    deleteButton.addEventListener("click", async () => {
      if (!confirm(`Delete person "${speaker.name}" along with all their samples?`)) return;
      try {
        await this._api(`speakers/${encodeURIComponent(speaker.name)}`, { method: "DELETE" });
        await this._loadSpeakers();
      } catch (error) {
        this._setSpeakersStatus(`Failed to delete: ${error.message}`, true);
      }
    });
    actions.appendChild(deleteButton);

    head.appendChild(actions);
    card.appendChild(head);

    if (!speaker.samples.length) {
      const empty = document.createElement("div");
      empty.className = "empty-hint";
      empty.textContent = "No samples for this person.";
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
      del.textContent = "Delete";
      del.addEventListener("click", async () => {
        try {
          await this._api(
            `speakers/${encodeURIComponent(speaker.name)}/samples/${encodeURIComponent(sample.id)}`,
            { method: "DELETE" }
          );
          await this._loadSpeakers();
        } catch (error) {
          this._setSpeakersStatus(`Failed to delete sample: ${error.message}`, true);
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
      this._setSpeakersStatus("Enter a person's name.", true);
      return;
    }
    const form = new FormData();
    form.append("name", name);
    try {
      await this._api("speakers", { method: "POST", body: form });
      input.value = "";
      await this._loadSpeakers();
    } catch (error) {
      this._setSpeakersStatus(`Failed to add: ${error.message}`, true);
    }
  }

  async _uploadSample(name, file) {
    const form = new FormData();
    form.append("file", file);
    this._setSpeakersStatus(`Uploading sample for ${name}...`, false);
    try {
      await this._api(`speakers/${encodeURIComponent(name)}/samples`, { method: "POST", body: form });
      this._setSpeakersStatus(`Added sample for ${name}.`, false);
      await this._loadSpeakers();
    } catch (error) {
      this._setSpeakersStatus(`Failed to upload sample: ${error.message}`, true);
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
        button.textContent = "Record";
        button.classList.remove("button-danger");
        if (blob) {
          this._uploadSample(name, new File([blob], "recording.webm", { type: blob.type }));
        }
      });
      this._recorder.start();
      button.textContent = "Stop";
      button.classList.add("button-danger");
      this._setSpeakersStatus(`Recording for ${name}... click "Stop" to save.`, false);
    } catch (error) {
      this._stopStream();
      this._setSpeakersStatus(`Microphone access failed: ${error.message}`, true);
    }
  }

  /* --------------------------------------------------------------- pending */

  async _loadPending() {
    const status = this._el("pending-status");
    try {
      const data = await this._api("pending");
      this._renderPending(data.clusters || []);
    } catch (error) {
      status.textContent = `Failed to load: ${error.message}`;
    }
  }

  _renderPending(clusters) {
    const list = this._el("pending-list");
    const status = this._el("pending-status");
    list.innerHTML = "";
    const count = clusters.reduce((total, cluster) => total + cluster.clips.length, 0);
    status.textContent = count
      ? `${count} recordings in ${clusters.length} groups (group = same voice).`
      : "No pending recordings — all voices recognized or the buffer is empty.";
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
    fresh.textContent = "+ new person…";
    select.appendChild(fresh);
    return select;
  }

  _resolveAssignTarget(select) {
    if (select.value === "__new__") {
      const name = prompt("New person's name:");
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
      status.textContent = `Assigned ${result.claimed.length} recordings to "${name}".`;
    } catch (error) {
      status.textContent = `Failed to assign: ${error.message}`;
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
      `<span class="speaker-name">Unknown voice</span>` +
      `<div class="speaker-meta">${clipCount} ${clipCount === 1 ? "recording" : "recordings"}</div>`;
    head.appendChild(title);

    const actions = document.createElement("div");
    actions.className = "button-row";
    const target = this._assignSelect();
    const assignAll = document.createElement("button");
    assignAll.className = "btn-small";
    assignAll.type = "button";
    assignAll.textContent = "Assign group";
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
      const text = clip.text ? ` · "${clip.text}"` : "";
      meta.textContent = `${(clip.seconds || 0).toFixed(1)} s · ${when}${text}`;
      const spacer = document.createElement("div");
      spacer.className = "spacer";
      const assignOne = document.createElement("button");
      assignOne.className = "button-secondary btn-small";
      assignOne.type = "button";
      assignOne.textContent = "Only this one";
      assignOne.addEventListener("click", () => {
        const name = this._resolveAssignTarget(target);
        if (name) this._claim(name, clip.id, false);
      });
      const remove = document.createElement("button");
      remove.className = "button-danger btn-small";
      remove.type = "button";
      remove.textContent = "Delete";
      remove.addEventListener("click", async () => {
        try {
          await this._api(`pending/${encodeURIComponent(clip.id)}`, { method: "DELETE" });
        } catch (error) {
          this._el("pending-status").textContent = `Failed to delete: ${error.message}`;
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
      status.textContent = `Failed to load: ${error.message}`;
    }
  }

  _renderHistory(entries) {
    const list = this._el("history-list");
    const status = this._el("history-status");
    list.innerHTML = "";
    if (!entries.length) {
      status.textContent = "No entries yet — the log fills up with each transcription.";
      return;
    }
    const recognized = entries.filter((e) => e.speaker).length;
    status.textContent =
      `${entries.length} recent transcriptions · recognized: ${recognized} · unknown: ${entries.length - recognized}`;
    for (const entry of entries) {
      const row = document.createElement("div");
      row.className = "sample-row";
      const meta = document.createElement("span");
      meta.className = "sample-meta";
      const when = entry.ts ? new Date(entry.ts * 1000).toLocaleString() : "";
      let who = "unknown";
      if (entry.speaker) {
        const score = entry.score != null ? ` (${entry.score.toFixed(2)})` : "";
        const role = entry.role ? ` · ${entry.role}` : "";
        who = `${entry.speaker}${score}${role}`;
      }
      meta.textContent = `${when} · ${who} · "${entry.text || ""}"`;
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
      this._el("settings-status").textContent = `Failed to load settings: ${error.message}`;
    }
  }

  async _saveSettings() {
    const form = new FormData();
    form.append("speaker_text_mode", this._el("speaker-text-mode").value);
    try {
      await this._api("settings", { method: "POST", body: form });
      this._el("settings-status").textContent = "Saved — takes effect from the next transcription.";
    } catch (error) {
      this._el("settings-status").textContent = `Failed to save: ${error.message}`;
    }
  }

  async _exportBackup() {
    const status = this._el("backup-status");
    status.textContent = "Preparing backup...";
    try {
      const blob = await this._apiBlob("export");
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "speakers-backup.tar.gz";
      link.click();
      URL.revokeObjectURL(link.href);
      status.textContent = "Backup downloaded.";
    } catch (error) {
      status.textContent = `Export failed: ${error.message}`;
    }
  }

  async _importBackup() {
    const input = this._el("import-backup");
    const status = this._el("backup-status");
    if (!input.files.length) return;
    if (!confirm("Restore backup? Existing files with the same names will be overwritten.")) {
      input.value = "";
      return;
    }
    const form = new FormData();
    form.append("file", input.files[0]);
    try {
      const result = await this._api("import", { method: "POST", body: form });
      status.textContent = `Restored ${result.files} files.`;
      await this._loadSpeakers();
      await this._loadPending();
      await this._loadHistory();
    } catch (error) {
      status.textContent = `Import failed: ${error.message}`;
    }
    input.value = "";
  }
}

customElements.define("wyoming-transcribe-panel", WyomingTranscribePanel);
