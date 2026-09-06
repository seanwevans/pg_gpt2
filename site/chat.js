/*
 * Chat front-end for a pg_gpt2 instance.
 *
 * This file is served from GitHub Pages, which is static hosting: there is no
 * model here. Generation happens on a pg_gpt2 endpoint (see server/app.py in
 * the repository) that the visitor points this page at. The endpoint URL comes
 * from, in order: the ?endpoint= query parameter, localStorage, then the
 * optional config.json shipped alongside this page.
 */

const STORAGE_KEY = "pg_gpt2.settings.v1";

const el = (id) => document.getElementById(id);
const ui = {
  dot: el("dot"),
  statusText: el("status-text"),
  statusModel: el("status-model"),
  setup: el("setup"),
  transcript: el("transcript"),
  composer: el("composer"),
  prompt: el("prompt"),
  send: el("send"),
  settings: el("settings"),
  endpoint: el("endpoint"),
  model: el("model"),
  reconnect: el("reconnect"),
};

const sliders = ["max_tokens", "temperature", "top_k", "top_p"];

const state = {
  endpoint: "",
  model: "",
  max_tokens: 32,
  temperature: 0.8,
  top_k: 40,
  top_p: 0.95,
  messages: [],
  busy: false,
};

/* ---------------------------------------------------------------- settings */

function loadSettings() {
  let stored = {};
  try {
    stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
  } catch {
    stored = {};
  }
  Object.assign(state, stored);

  const fromQuery = new URLSearchParams(location.search).get("endpoint");
  if (fromQuery) state.endpoint = fromQuery;
}

function saveSettings() {
  const { endpoint, model, max_tokens, temperature, top_k, top_p } = state;
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ endpoint, model, max_tokens, temperature, top_k, top_p })
    );
  } catch {
    /* private browsing: settings just do not persist */
  }
}

async function loadDefaultConfig() {
  // Optional: a deployment can ship a config.json next to index.html so that
  // visitors do not have to type an endpoint at all.
  if (state.endpoint) return;
  try {
    const res = await fetch("./config.json", { cache: "no-store" });
    if (!res.ok) return;
    const cfg = await res.json();
    if (cfg.endpoint) state.endpoint = cfg.endpoint;
    if (cfg.model) state.model = cfg.model;
  } catch {
    /* no config.json: the visitor supplies the endpoint */
  }
}

function syncControls() {
  ui.endpoint.value = state.endpoint;
  for (const key of sliders) {
    el(key).value = state[key];
    el(`${key}_v`).textContent = state[key];
  }
}

/* ------------------------------------------------------------------ status */

function setStatus(kind, text, detail = "") {
  ui.dot.className = `dot ${kind}`;
  ui.statusText.textContent = text;
  ui.statusModel.textContent = detail;
}

function baseUrl() {
  return state.endpoint.replace(/\/+$/, "");
}

async function probe() {
  if (!state.endpoint) {
    setStatus("bad", "No endpoint configured");
    ui.setup.hidden = false;
    ui.settings.open = true;
    ui.send.disabled = true;
    return;
  }

  ui.setup.hidden = true;
  setStatus("busy", "Connecting…");

  try {
    const res = await fetch(`${baseUrl()}/healthz`, { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const health = await res.json();

    el("max_tokens").max = String(health.max_tokens || 128);
    if (state.max_tokens > health.max_tokens) {
      state.max_tokens = health.max_tokens;
      syncControls();
    }

    setStatus(
      "ok",
      `Connected — pg_llm ${health.pg_llm_version}`,
      `· default model ${health.default_model}`
    );
    ui.send.disabled = false;
    await loadModels(health.default_model);
  } catch (err) {
    setStatus("bad", `Cannot reach ${baseUrl()}`, `· ${err.message}`);
    ui.send.disabled = true;
    ui.settings.open = true;
  }
}

async function loadModels(defaultModel) {
  try {
    const res = await fetch(`${baseUrl()}/v1/models`, { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();
    ui.model.innerHTML = "";

    const auto = document.createElement("option");
    auto.value = "";
    auto.textContent = `(server default: ${defaultModel})`;
    ui.model.append(auto);

    for (const model of data.models || []) {
      const option = document.createElement("option");
      option.value = model.id;
      const shape = `${model.n_layer}L·${model.d_model}d`;
      option.textContent = model.ready ? `${model.id} (${shape})` : `${model.id} — not loaded`;
      option.disabled = !model.ready;
      ui.model.append(option);
    }
    ui.model.value = state.model || "";
  } catch {
    /* model list is a nicety; the endpoint default still works */
  }
}

/* -------------------------------------------------------------- transcript */

function addMessage(role, text = "") {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;

  const who = document.createElement("span");
  who.className = "who";
  who.textContent = role === "user" ? "you" : role === "error" ? "error" : "pg_gpt2";

  const body = document.createElement("span");
  body.className = "body";
  body.textContent = text;

  wrapper.append(who, body);
  ui.transcript.append(wrapper);
  ui.transcript.scrollTop = ui.transcript.scrollHeight;
  return body;
}

/* ------------------------------------------------------------- generation */

function requestBody() {
  const body = {
    messages: state.messages,
    max_tokens: Number(state.max_tokens),
    temperature: Number(state.temperature),
    top_k: Number(state.top_k),
    top_p: Number(state.top_p),
  };
  if (state.model) body.model = state.model;
  return body;
}

/*
 * The endpoint streams Server-Sent Events, but SSE over POST is not something
 * EventSource can do, so read the response body and split frames by hand.
 */
async function streamCompletion(body, onDelta) {
  const res = await fetch(`${baseUrl()}/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const payload = await res.json();
      if (payload.detail) detail = payload.detail;
    } catch { /* keep the status line */ }
    throw new Error(detail);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let completion = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);

      let event = "message";
      const dataLines = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
      }
      if (!dataLines.length) continue;

      const payload = JSON.parse(dataLines.join("\n"));
      if (event === "token") {
        completion = payload.completion;
        onDelta(completion);
      } else if (event === "done") {
        completion = payload.completion ?? completion;
      } else if (event === "error") {
        throw new Error(payload.message);
      }
    }
  }
  return completion;
}

async function submit(text) {
  if (state.busy) return;
  state.busy = true;
  ui.send.disabled = true;
  ui.prompt.value = "";

  addMessage("user", text);
  state.messages.push({ role: "user", content: text });

  const body = addMessage("model", "");
  body.classList.add("cursor");

  const started = performance.now();
  try {
    const completion = await streamCompletion(requestBody(), (partial) => {
      body.textContent = partial;
      ui.transcript.scrollTop = ui.transcript.scrollHeight;
    });

    const trimmed = (completion || "").trim();
    body.textContent = trimmed || "(no output)";
    state.messages.push({ role: "assistant", content: trimmed });

    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    setStatus("ok", `Connected`, `· last completion ${seconds}s`);
  } catch (err) {
    body.remove();
    addMessage("error", err.message);
    setStatus("bad", "Generation failed", `· ${err.message}`);
  } finally {
    body.classList.remove("cursor");
    state.busy = false;
    ui.send.disabled = !state.endpoint;
    ui.prompt.focus();
  }
}

/* ------------------------------------------------------------------- wiring */

ui.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.prompt.value.trim();
  if (text) submit(text);
});

ui.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    ui.composer.requestSubmit();
  }
});

ui.endpoint.addEventListener("change", () => {
  state.endpoint = ui.endpoint.value.trim();
  saveSettings();
  probe();
});

ui.model.addEventListener("change", () => {
  state.model = ui.model.value;
  saveSettings();
});

ui.reconnect.addEventListener("click", probe);

for (const key of sliders) {
  el(key).addEventListener("input", (event) => {
    state[key] = event.target.value;
    el(`${key}_v`).textContent = event.target.value;
    saveSettings();
  });
}

(async function init() {
  loadSettings();
  await loadDefaultConfig();
  syncControls();
  await probe();
})();
