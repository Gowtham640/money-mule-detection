"use strict";

/*
 * Money-mule live dashboard.
 *
 * Data flow:  GET {API_ORIGIN}/api/live/state (snapshot, carries `seq`)
 *             + WS {API_ORIGIN}/ws/live (delta events, each with `seq`)
 *             -> store (single source of truth on the page) -> renderers
 *
 * The dashboard can run on a different port than FastAPI. API_ORIGIN is
 * window.MM_API_ORIGIN (from /config.js), ?api=, or hostname:8088.
 *
 * The page never invents data. Every number, row, node and edge comes from a
 * backend payload; the page only aggregates for display (sorting, top-N).
 */

const MAX_TX = 200;
const MAX_ALERTS = 120;
const RENDER_INTERVAL_MS = 150;
const CHART_INTERVAL_MS = 1000;
const INVESTIGATION_REFRESH_MS = 2000;
const WS_IDLE_TIMEOUT_MS = 40000;

const store = {
  seq: 0,
  config: {},
  accounts: new Map(),
  edges: new Map(),
  degree: new Map(),
  transactions: [],
  alerts: [],
  runs: new Map(),
  metrics: null,
  history: { runs: [], role_totals: {} },
  patternMemory: { stored: 0, last: null },
  attackTypes: [],
  models: {},
  health: {},
  lastCompletedRun: null,
};

const ui = {
  dirty: new Set(),
  renderTimer: null,
  lastRender: 0,
  chartTimer: null,
  lastChartRender: 0,
  htmlCache: new WeakMap(),
  charts: {},
  selectedForBan: new Set(),
  investigation: { id: null, data: null, refreshTimer: null, lastFetch: 0 },
  subgraph: { runId: null, replaying: false, token: 0 },
  sirenMuted: false,
  audio: null,
};

const $ = (id) => document.getElementById(id);

function resolveApiOrigin() {
  const params = new URLSearchParams(location.search);
  const fromQuery = params.get("api");
  if (fromQuery) return fromQuery.replace(/\/$/, "");
  if (typeof window.MM_API_ORIGIN === "string" && window.MM_API_ORIGIN) {
    return window.MM_API_ORIGIN.replace(/\/$/, "");
  }
  if (location.port && location.port !== "8088") {
    return `${location.protocol}//${location.hostname}:8088`;
  }
  return "";
}

const API_ORIGIN = resolveApiOrigin();

function apiUrl(path) {
  return API_ORIGIN ? `${API_ORIGIN}${path}` : path;
}

function wsUrl() {
  if (API_ORIGIN) {
    const url = new URL(API_ORIGIN);
    const scheme = url.protocol === "https:" ? "wss" : "ws";
    return `${scheme}://${url.host}/ws/live`;
  }
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws/live`;
}

// ------------------------------------------------------------------ formatting
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
const fmtInt = (n) => (n === null || n === undefined ? "--" : Number(n).toLocaleString("en-IN"));
const fmtMoney = (n) => (n === null || n === undefined ? "--" : `₹${Number(n).toLocaleString("en-IN", { maximumFractionDigits: 0 })}`);
const fmtScore = (x, d = 3) => (x === null || x === undefined ? "--" : Number(x).toFixed(d));
const fmtPct = (x) => (x === null || x === undefined ? "--" : `${(Number(x) * 100).toFixed(1)}%`);
const fmtMs = (ms) => (ms === null || ms === undefined ? "--" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(2)} s`);
const fmtTime = (iso) => (iso ? new Date(iso).toLocaleTimeString([], { hour12: false }) : "--");

function setText(id, value) {
  const el = $(id);
  if (el && el.textContent !== String(value)) el.textContent = value;
}

function setHtml(el, html) {
  if (!el) return;
  if (ui.htmlCache.get(el) === html) return;
  ui.htmlCache.set(el, html);
  el.innerHTML = html;
}

function statusPill(status) {
  const label = { normal: "normal", early: "warning", fraud: "flagged", banned: "banned" }[status] || status;
  return `<span class="status-tag status-tag--${escapeHtml(status)}">${escapeHtml(label)}</span>`;
}

function accountLink(id) {
  return `<button class="link-btn" data-account="${escapeHtml(id)}" type="button">${escapeHtml(id)}</button>`;
}

function table(columns, rows, emptyText) {
  if (!rows.length) return `<div class="empty-state">${escapeHtml(emptyText)}</div>`;
  const head = columns.map((c) => `<th>${escapeHtml(c.label)}</th>`).join("");
  const body = rows
    .map((row) => `<tr${row.__class ? ` class="${row.__class}"` : ""}>${columns.map((c) => `<td>${c.render ? c.render(row) : escapeHtml(row[c.key])}</td>`).join("")}</tr>`)
    .join("");
  return `<table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

// ------------------------------------------------------------------- network
async function api(path, options = {}) {
  const response = await fetch(apiUrl(path), {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch (_) {
    payload = null;
  }
  if (!response.ok) {
    const detail = payload && payload.detail ? payload.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  $("toast-stack").appendChild(toast);
  setTimeout(() => toast.remove(), 4200);
  const stack = $("toast-stack");
  while (stack.children.length > 5) stack.firstChild.remove();
}

// ---------------------------------------------------------------- websocket
const conn = {
  socket: null,
  retry: 0,
  hydrated: false,
  buffer: [],
  reconnectTimer: null,
  pingTimer: null,
  lastMessageAt: 0,
  firstLoad: true,
};

function setConnStatus(state, detail = "") {
  const dot = $("conn-dot");
  dot.className = `status-dot ${state === "live" ? "live" : state === "offline" ? "offline" : "pending"}`;
  const label = { live: "● LIVE", reconnecting: "RECONNECTING", connecting: "CONNECTING", offline: "OFFLINE" }[state] || state;
  setText("conn-text", detail ? `${label} · ${detail}` : label);
  const enabled = state === "live";
  ["spawn-attack-btn", "spawner-btn", "reset-btn"].forEach((id) => {
    $(id).disabled = !enabled;
  });
}

function connect() {
  clearTimeout(conn.reconnectTimer);
  setConnStatus(conn.retry ? "reconnecting" : "connecting");
  const socket = new WebSocket(wsUrl());
  conn.socket = socket;
  conn.hydrated = false;
  conn.buffer = [];

  socket.onopen = async () => {
    conn.lastMessageAt = Date.now();
    try {
      // WS is open and buffering; now take the snapshot and replay newer events.
      const snapshot = await api("/api/live/state");
      if (conn.socket !== socket) return;
      applySnapshot(snapshot);
      conn.hydrated = true;
      const pending = conn.buffer;
      conn.buffer = [];
      pending.forEach(handleMessage);
      conn.retry = 0;
      setConnStatus("live");
      if (conn.firstLoad) {
        conn.firstLoad = false;
        $("global-loader").classList.add("hidden");
      }
    } catch (error) {
      setText("loader-text", `Snapshot failed: ${error.message}. Retrying...`);
      socket.close();
    }
  };

  socket.onmessage = (event) => {
    conn.lastMessageAt = Date.now();
    let message;
    try {
      message = JSON.parse(event.data);
    } catch (_) {
      return;
    }
    if (!conn.hydrated) conn.buffer.push(message);
    else handleMessage(message);
  };

  socket.onclose = () => {
    if (conn.socket !== socket) return;
    conn.socket = null;
    conn.hydrated = false;
    conn.retry += 1;
    const delay = Math.min(1000 * 2 ** Math.min(conn.retry - 1, 3), 10000) + Math.random() * 300;
    setConnStatus(conn.retry > 4 ? "offline" : "reconnecting", `retry in ${(delay / 1000).toFixed(1)} s`);
    if (conn.firstLoad) setText("loader-text", "Backend unreachable — retrying...");
    conn.reconnectTimer = setTimeout(connect, delay);
  };
}

function startKeepAlive() {
  clearInterval(conn.pingTimer);
  conn.pingTimer = setInterval(() => {
    const socket = conn.socket;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (Date.now() - conn.lastMessageAt > WS_IDLE_TIMEOUT_MS) {
      socket.close();
      return;
    }
    socket.send(JSON.stringify({ type: "ping" }));
  }, 15000);
}

function handleMessage(message) {
  if (message.type === "batch") {
    message.events.forEach(applyEvent);
  } else if (message.type === "hello" || message.type === "pong") {
    return;
  } else if (message.type === "heartbeat") {
    return;
  } else {
    applyEvent(message);
  }
  scheduleRender();
}

// ------------------------------------------------------------------ snapshot
function applySnapshot(snapshot) {
  store.seq = snapshot.seq;
  store.config = snapshot.config;
  store.metrics = snapshot.metrics;
  store.models = snapshot.models || {};
  store.health = snapshot.health || {};
  store.history = snapshot.history;
  store.patternMemory = snapshot.pattern_memory;
  store.attackTypes = snapshot.attack_types;
  store.accounts = new Map(snapshot.accounts.map((a) => [a.account_id, a]));
  store.edges = new Map();
  store.degree = new Map();
  snapshot.graph.edges.forEach((edge) => setEdge(edge));
  store.transactions = snapshot.recent_transactions.slice(-MAX_TX);
  store.alerts = snapshot.alerts.slice(-MAX_ALERTS);
  store.runs = new Map(snapshot.attack_runs.map((run) => [run.attack_run_id, run]));
  store.lastCompletedRun = snapshot.attack_runs.find((run) => run.status === "completed") || null;

  populateAttackTypes();
  populateAccountOptions();
  rebuildLiveGraph();
  const latest = snapshot.attack_runs[0];
  if (latest && (!ui.subgraph.runId || !store.runs.has(ui.subgraph.runId))) {
    loadRunIntoSubgraph(latest.attack_run_id, { animate: false });
  }
  markDirty("all");
  scheduleCharts(true);
}

// -------------------------------------------------------------------- events
function setEdge(edge) {
  const existed = store.edges.has(edge.id);
  store.edges.set(edge.id, edge);
  if (!existed) {
    store.degree.set(edge.source, (store.degree.get(edge.source) || 0) + 1);
    store.degree.set(edge.target, (store.degree.get(edge.target) || 0) + 1);
  }
}

function deleteEdge(id) {
  const edge = store.edges.get(id);
  if (!edge) return null;
  store.edges.delete(id);
  for (const acc of [edge.source, edge.target]) {
    const next = (store.degree.get(acc) || 1) - 1;
    if (next <= 0) store.degree.delete(acc);
    else store.degree.set(acc, next);
  }
  return edge;
}

function mergeAccount(data) {
  const prev = store.accounts.get(data.account_id);
  const next = prev ? { ...prev, ...data } : data;
  store.accounts.set(data.account_id, next);
  return { prev, next };
}

function applyEvent(event) {
  if (event.seq !== null && event.seq !== undefined) {
    if (event.seq <= store.seq) return; // already folded into the snapshot
    store.seq = event.seq;
  }
  const d = event.data;
  switch (event.type) {
    case "transaction_created": {
      store.transactions.push(d);
      if (store.transactions.length > MAX_TX) store.transactions.splice(0, store.transactions.length - MAX_TX);
      const risky = ["early", "fraud"].includes(d.sender_status) || ["early", "fraud"].includes(d.receiver_status);
      const flagged = d.sender_status === "fraud" || d.receiver_status === "fraud";
      graphs.live?.pulse(d.sender, d.receiver, d.is_attack ? "attack" : flagged ? "flagged" : risky ? "risky" : "normal", d.is_attack ? 1300 : 900);
      touchInvestigation([d.sender, d.receiver]);
      markDirty("feeds");
      break;
    }
    case "graph_edge_upserted":
      setEdge(d);
      graphs.live?.upsertEdge(d);
      syncActive([d.source, d.target]);
      break;
    case "graph_edges_expired": {
      const touched = [];
      d.changed.forEach((edge) => {
        setEdge(edge);
        graphs.live?.upsertEdge(edge);
      });
      d.removed.forEach((id) => {
        const edge = deleteEdge(id);
        graphs.live?.removeEdge(id);
        if (edge) touched.push(edge.source, edge.target);
      });
      syncActive(touched);
      break;
    }
    case "account_created":
      mergeAccount(d);
      graphs.live?.upsertNode(d.account_id, { status: d.status, fresh: true, active: false });
      appendAccountOption(d.account_id);
      markDirty("accounts");
      break;
    case "account_updated": {
      const { prev, next } = mergeAccount(d);
      if (!prev || prev.status !== next.status) {
        graphs.live?.upsertNode(d.account_id, { status: next.status });
        if (graphs.attack?.hasNode(d.account_id) && !ui.subgraph.replaying) graphs.attack.upsertNode(d.account_id, { status: next.status });
      }
      touchInvestigation([d.account_id]);
      markDirty("accounts");
      break;
    }
    case "accounts_decayed":
      d.accounts.forEach((a) => {
        const { prev, next } = mergeAccount(a);
        if (prev && prev.status !== next.status) {
          graphs.live?.upsertNode(a.account_id, { status: next.status });
          if (graphs.attack?.hasNode(a.account_id) && !ui.subgraph.replaying) graphs.attack.upsertNode(a.account_id, { status: next.status });
        }
      });
      markDirty("accounts");
      break;
    case "early_warning":
    case "early_warning_cleared":
      markDirty("accounts");
      break;
    case "role_updated":
      mergeAccount({ account_id: d.account_id, role: d.new_role });
      markDirty("accounts");
      break;
    case "fraud_detected":
      showToast(`${d.account_id} flagged as money mule (${d.role}) — fused ${fmtScore(d.fused_score, 2)}`, "error");
      flash();
      markDirty("accounts");
      break;
    case "alert_created":
      store.alerts.push(d);
      if (store.alerts.length > MAX_ALERTS) store.alerts.splice(0, store.alerts.length - MAX_ALERTS);
      markDirty("alerts");
      break;
    case "accounts_banned":
      d.account_ids.forEach((id) => ui.selectedForBan.delete(id));
      showToast(`Banned ${d.account_ids.length} account(s): ${d.account_ids.join(", ")}`, "success");
      markDirty("accounts");
      break;
    case "attack_started":
      store.runs.set(d.attack_run_id, d);
      showToast(`${d.attack_run_id} started: ${d.attack_name} (${d.planned_steps} transactions)`, "info");
      setText("attack-alert-strip", `${d.attack_run_id} · ${d.attack_name} is injecting ${d.planned_steps} transactions through the live pipeline.`);
      if (!ui.subgraph.replaying) loadRunIntoSubgraph(d.attack_run_id, { animate: false, fresh: true });
      markDirty("runs");
      break;
    case "attack_step": {
      const run = store.runs.get(d.attack_run_id);
      if (run) run.transaction_count = d.step;
      if (ui.subgraph.runId === d.attack_run_id && !ui.subgraph.replaying) addSubgraphStep(d, true);
      markDirty("runs");
      break;
    }
    case "attack_updated":
      store.runs.set(d.attack_run_id, { ...store.runs.get(d.attack_run_id), ...d });
      markDirty("runs");
      break;
    case "attack_detected":
      store.runs.set(d.attack_run_id, { ...store.runs.get(d.attack_run_id), ...d });
      setText("attack-alert-strip", `${d.attack_run_id} DETECTED ${fmtMs(d.detection_latency_ms)} after start — first flagged account ${d.detected_by}.`);
      $("attack-alert-strip").classList.add("is-alert");
      playSiren();
      markDirty("runs");
      break;
    case "attack_completed": {
      const merged = { ...store.runs.get(d.attack_run_id), ...d };
      store.runs.set(d.attack_run_id, merged);
      if (d.status === "completed") {
        store.lastCompletedRun = merged;
        store.history.runs.push({
          attack_run_id: d.attack_run_id,
          attack_type: d.attack_type,
          attack_name: d.attack_name,
          detection_latency_ms: d.detection_latency_ms,
          first_warning_latency_ms: d.first_warning_latency_ms,
          precision: d.precision,
          recall: d.recall,
          threshold_after: d.threshold_after,
          role_counts: d.role_counts,
        });
        if (store.history.runs.length > 40) store.history.runs.shift();
        showToast(`${d.attack_run_id} evaluated: ${d.detected_accounts.length}/${d.labeled_participants.length} mules detected, ${d.false_positive_accounts.length} false positive(s)`, "success");
      } else {
        showToast(`${d.attack_run_id} failed: ${d.error || "unknown error"}`, "error");
      }
      $("attack-alert-strip").classList.remove("is-alert");
      setText("attack-alert-strip", `${d.attack_run_id} ${d.status}. Detected ${d.detected_accounts?.length ?? 0} of ${d.labeled_participants?.length ?? 0} injected mules.`);
      $("replay-btn").disabled = false;
      markDirty("runs");
      scheduleCharts();
      break;
    }
    case "threshold_updated":
      if (store.metrics) store.metrics.detection_threshold = d.new;
      markDirty("metrics");
      break;
    case "pattern_memory_updated":
      store.patternMemory = { stored: d.stored, last: d };
      markDirty("runs");
      break;
    case "metrics_updated":
      store.metrics = d;
      markDirty("metrics");
      scheduleCharts();
      break;
    case "system_status":
      if (store.metrics) store.metrics.spawner_running = d.spawner_running;
      markDirty("metrics");
      break;
    case "system_reset":
      showToast("Database was reset — reloading state from the backend", "info");
      conn.socket?.close();
      break;
    default:
      break;
  }
}

// --------------------------------------------------------- render scheduling
function markDirty(...keys) {
  keys.forEach((key) => {
    if (key === "all") ["metrics", "feeds", "alerts", "accounts", "runs"].forEach((k) => ui.dirty.add(k));
    else ui.dirty.add(key);
  });
  scheduleRender();
}

function scheduleRender() {
  if (ui.renderTimer || !ui.dirty.size) return;
  const wait = Math.max(0, RENDER_INTERVAL_MS - (performance.now() - ui.lastRender));
  ui.renderTimer = setTimeout(() => {
    requestAnimationFrame(() => {
      ui.renderTimer = null;
      ui.lastRender = performance.now();
      const dirty = new Set(ui.dirty);
      ui.dirty.clear();
      if (dirty.has("metrics")) renderMetrics();
      if (dirty.has("feeds")) renderFeeds();
      if (dirty.has("alerts")) renderAlerts();
      if (dirty.has("accounts") || dirty.has("metrics")) renderAccountPanels();
      if (dirty.has("runs")) renderRuns();
    });
  }, wait);
}

function scheduleCharts(force = false) {
  if (ui.chartTimer) return;
  const wait = force ? 0 : Math.max(0, CHART_INTERVAL_MS - (performance.now() - ui.lastChartRender));
  ui.chartTimer = setTimeout(() => {
    ui.chartTimer = null;
    ui.lastChartRender = performance.now();
    renderCharts();
  }, wait);
}

// ------------------------------------------------------------------ renderers
function renderMetrics() {
  const m = store.metrics;
  if (!m) return;
  const bySource = m.tx_by_source || {};
  setText("m-tx-total", fmtInt(m.tx_total));
  setText("m-tx-split", `${fmtInt(bySource.normal || 0)} normal · ${fmtInt(bySource.attack || 0)} attack${bySource.manual ? ` · ${fmtInt(bySource.manual)} manual` : ""}`);
  setText("m-tps", fmtScore(m.tps, 2));
  setText("m-active", fmtInt(m.accounts_active));
  setText("m-window", `${fmtInt(m.accounts_in_window)} active in ${store.config.feature_window_sec}s window`);
  setText("m-early", fmtInt(m.early_warning_count));
  setText("m-warning-thr", `warning threshold ${fmtScore(m.warning_threshold)}`);
  setText("m-flagged", fmtInt(m.flagged_count));
  setText("m-det-thr", `fused score ≥ ${fmtScore(m.detection_threshold)}`);
  setText("m-banned", fmtInt(m.banned_count));
  const at = m.alerts_by_type || {};
  setText("m-alerts", fmtInt(m.alerts_total));
  setText("m-alert-split", `${fmtInt(at.fraud_detected || 0)} fraud · ${fmtInt(at.early_warning || 0)} warning · ${fmtInt(at.behavioral_drift || 0)} drift`);
  setText("m-attacks", fmtInt(m.attacks_total));
  setText("m-attacks-running", `${m.attacks_running} running · ${m.attacks_completed} evaluated`);
  setText("m-attacks-detected", fmtInt(m.attacks_detected));
  setText("m-detection-rate", m.detection_rate === null ? "no evaluated runs yet" : `${fmtPct(m.detection_rate)} of evaluated runs`);
  setText("m-latency", fmtMs(m.avg_detection_latency_ms));
  setText("m-warning-latency", `first warning avg ${fmtMs(m.avg_first_warning_latency_ms)}`);
  setText("m-pr", m.precision === null ? "--" : `${fmtScore(m.precision, 2)} / ${fmtScore(m.recall, 2)}`);
  setText("m-f1", `F1 ${fmtScore(m.f1, 2)} · TP ${m.true_positives} FP ${m.false_positives} FN ${m.false_negatives}`);
  setText("m-pipeline", m.pipeline_ms_p50 === null ? "--" : `${fmtScore(m.pipeline_ms_p50, 1)} ms`);
  setText("m-model-ms", `p95 ${fmtScore(m.pipeline_ms_p95, 1)} · RF ${fmtScore(m.rf_ms, 1)} · GNN ${fmtScore(m.gnn_ms, 1)} ms`);
  setText("warning-threshold", fmtScore(m.warning_threshold));
  setText("detection-threshold", fmtScore(m.detection_threshold));
  setText("adaptive-threshold", fmtScore(m.detection_threshold));
  setText("adaptive-note", `Calibrated start ${fmtScore(m.calibrated_threshold, 2)} (≤6 false-positive accounts/hour on a held-out stream). After each attack is evaluated, missed mules lower it and false positives since the previous evaluation raise it, within ${fmtScore(m.threshold_band[0], 2)}–${fmtScore(m.threshold_band[1], 2)}.`);
  setText("flagged-note", `Flagged when the fused score (0.5·RF + 0.3·rules + 0.2·GNN) reaches the adaptive threshold, currently ${fmtScore(m.detection_threshold)}. Flags persist until an analyst bans the account.`);
  setText("rule-high", fmtInt(m.rule_high_count));
  setText("rule-medium", fmtInt(m.rule_medium_count));
  setText("rule-scored", fmtInt(m.rule_scored_count));
  setText("drift-chip", `${fmtInt(m.drift_alert_count)} above 3σ`);
  setText("attack-running-chip", `Running: ${m.attacks_running}`);
  setText("spawner-btn", m.spawner_running ? "Pause Traffic" : "Resume Traffic");
  setText("live-graph-info", `${fmtInt(m.edges_in_window)} edges · ${fmtInt(m.accounts_in_window)} accounts active in the last ${store.config.feature_window_sec}s · ${fmtInt(m.accounts_total)} nodes`);
  renderModelChips();
}

function renderModelChips() {
  const models = store.models || {};
  const chip = (label, status) => {
    const ok = status === "loaded" || status === "ready" || status === "ok";
    return `<span class="model-chip ${ok ? "ok" : "bad"}" title="${escapeHtml(status)}">${label} ${ok ? "✓" : "✗"}</span>`;
  };
  setHtml($("model-chips"), [
    chip("RF", models.rf_model),
    chip("GNN", models.gnn_model),
    chip("SHAP", models.shap),
    chip("DB", "ok"),
  ].join(""));
  setText("gnn-chip", models.gnn_model === "loaded" ? "GNN live" : `GNN: ${models.gnn_model || "--"}`);
  setText("rf-auc", fmtScore(models.rf_eval?.roc_auc));
  setText("gnn-auc", fmtScore(models.gnn_eval?.roc_auc));
  setText("model-device", models.model_device || "--");
}

function txRow(tx) {
  const tags = [];
  if (tx.is_attack) tags.push(`<span class="tag tag--attack" title="Injected by the attack simulator">${escapeHtml(tx.attack_run_id)} #${tx.attack_step}</span>`);
  if (tx.source === "manual") tags.push(`<span class="tag">manual</span>`);
  const worst = tx.sender_status === "fraud" || tx.receiver_status === "fraud" ? "fraud" : tx.sender_status === "early" || tx.receiver_status === "early" ? "early" : "normal";
  return `<div class="feed-row feed-row--${worst === "fraud" ? "fraud" : worst === "early" ? "warning" : "normal"}">
    <strong>${accountLink(tx.sender)} → ${accountLink(tx.receiver)}</strong>
    <span>${fmtMoney(tx.amount)} · ${escapeHtml(tx.channel)} · ${fmtTime(tx.timestamp)} ${tags.join(" ")}</span>
    <span class="feed-meta">${escapeHtml(tx.transaction_id)} · scores ${fmtScore(tx.sender_score, 2)} / ${fmtScore(tx.receiver_score, 2)}${tx.processing_ms ? ` · ${fmtScore(tx.processing_ms, 1)} ms` : ""}</span>
  </div>`;
}

function renderFeeds() {
  const recent = store.transactions.slice(-40).reverse();
  setHtml($("tx-feed"), recent.length ? recent.map(txRow).join("") : `<div class="empty-state">Waiting for the first transaction...</div>`);
  const risky = store.transactions
    .filter((tx) => ["early", "fraud"].includes(tx.sender_status) || ["early", "fraud"].includes(tx.receiver_status))
    .slice(-40)
    .reverse();
  setHtml($("risky-feed"), risky.length ? risky.map(txRow).join("") : `<div class="empty-state">No risky transactions in the recent stream.</div>`);
  setText("tx-feed-meta", `last ${recent.length} of ${fmtInt(store.metrics?.tx_total)}`);
  setText("risky-feed-meta", `${risky.length} in last ${store.transactions.length}`);
}

function renderAlerts() {
  const alerts = store.alerts.slice(-40).reverse();
  setHtml($("alert-feed"), alerts.length ? alerts.map((a) => `
    <div class="feed-row feed-row--${a.severity === "critical" || a.severity === "high" ? "fraud" : a.severity === "medium" ? "warning" : "normal"}">
      <strong><span class="sev sev--${escapeHtml(a.severity)}">${escapeHtml(a.severity)}</span> ${escapeHtml(a.alert_type.replace(/_/g, " "))} · ${accountLink(a.account_id)}</strong>
      <span>${escapeHtml(a.message)}</span>
      <span class="feed-meta">#${a.alert_id} · ${fmtTime(a.timestamp)} · ${escapeHtml(a.transaction_id || "")}</span>
    </div>`).join("") : `<div class="empty-state">No alerts yet.</div>`);
  setText("alerts-meta", `${fmtInt(store.metrics?.alerts_total)} total`);
}

function renderAccountPanels() {
  const accounts = [...store.accounts.values()];
  const early = accounts.filter((a) => a.status === "early").sort((a, b) => b.early_risk - a.early_risk);
  setText("early-chip", `${early.length} suspicious`);
  setHtml($("early-table"), table([
    { label: "Account", render: (r) => accountLink(r.account_id) },
    { label: "Warning", render: (r) => fmtScore(r.early_risk) },
    { label: "Signals", render: (r) => Object.keys(r.signals || {}).length },
    { label: "Fused", render: (r) => fmtScore(r.fused_score) },
    { label: "Reasons", render: (r) => escapeHtml((r.reasons || []).join(", ") || "—") },
  ], early.slice(0, 30), "No account is above the adaptive warning threshold right now."));

  const flagged = accounts.filter((a) => a.status === "fraud").sort((a, b) => (b.flagged_at || "").localeCompare(a.flagged_at || ""));
  setText("flagged-chip", `${flagged.length} flagged`);
  [...ui.selectedForBan].forEach((id) => {
    if (store.accounts.get(id)?.status !== "fraud") ui.selectedForBan.delete(id);
  });
  setHtml($("flagged-table"), table([
    { label: "", render: (r) => `<input type="checkbox" data-ban="${escapeHtml(r.account_id)}" ${ui.selectedForBan.has(r.account_id) ? "checked" : ""}>` },
    { label: "Account", render: (r) => accountLink(r.account_id) },
    { label: "Fused", render: (r) => fmtScore(r.fused_score) },
    { label: "RF", render: (r) => fmtScore(r.rf_score) },
    { label: "GNN", render: (r) => fmtScore(r.gnn_score) },
    { label: "Rules", render: (r) => `${r.rule_score ?? "--"}/${store.config.max_rule_score}` },
    { label: "Role", render: (r) => escapeHtml(r.role) },
    { label: "Flagged", render: (r) => fmtTime(r.flagged_at) },
  ], flagged, "No accounts flagged. Spawn an attack to see detection happen."));
  $("ban-btn").disabled = ui.selectedForBan.size === 0;
  setText("ban-btn", ui.selectedForBan.size ? `Ban Selected (${ui.selectedForBan.size})` : "Ban Selected");

  const scored = accounts.filter((a) => a.rule_score !== null && a.rule_score !== undefined && a.status !== "banned");
  const byRule = [...scored].sort((a, b) => b.rule_score - a.rule_score || (b.fused_score || 0) - (a.fused_score || 0)).slice(0, 15);
  setHtml($("rule-table"), table([
    { label: "Account", render: (r) => accountLink(r.account_id) },
    { label: "Points", render: (r) => `${r.rule_score}/${store.config.max_rule_score}` },
    { label: "Rules fired", render: (r) => escapeHtml((r.rules_fired || []).join(", ").replace(/_/g, " ") || "—") },
    { label: "Status", render: (r) => statusPill(r.status) },
  ], byRule, "No accounts scored yet."));

  const byFused = [...scored].sort((a, b) => (b.fused_score || 0) - (a.fused_score || 0)).slice(0, 15);
  setHtml($("ml-table"), table([
    { label: "Account", render: (r) => accountLink(r.account_id) },
    { label: "RF", render: (r) => fmtScore(r.rf_score) },
    { label: "GNN", render: (r) => fmtScore(r.gnn_score) },
    { label: "Rule norm", render: (r) => fmtScore(r.rule_score / store.config.max_rule_score) },
    { label: "Fused", render: (r) => `<strong>${fmtScore(r.fused_score)}</strong>` },
    { label: "Status", render: (r) => statusPill(r.status) },
  ], byFused, "No accounts scored yet."));

  const drift = accounts.filter((a) => a.drift_score !== null && a.drift_score !== undefined && a.drift_score >= 1)
    .sort((a, b) => b.drift_score - a.drift_score).slice(0, 15);
  setHtml($("drift-table"), table([
    { label: "Account", render: (r) => accountLink(r.account_id) },
    { label: "Drift", render: (r) => `${fmtScore(r.drift_score, 2)}σ` },
    { label: "Changed most", render: (r) => escapeHtml((r.drift_top || []).join(", ").replace(/_/g, " ") || "mixed") },
    { label: "Status", render: (r) => statusPill(r.status) },
  ], drift, "No account has drifted from its baseline window."));
}

function runStatusChips(run) {
  const chips = [];
  if (run.first_warning_latency_ms !== null && run.first_warning_latency_ms !== undefined) chips.push(`<span class="tag tag--warn">warning +${fmtMs(run.first_warning_latency_ms)}</span>`);
  if (run.detected_at) chips.push(`<span class="tag tag--fraud">detected +${fmtMs(run.detection_latency_ms)}</span>`);
  chips.push(`<span class="tag tag--status">${escapeHtml(run.status)}</span>`);
  return chips.join(" ");
}

function renderRuns() {
  const runs = [...store.runs.values()].sort((a, b) => (b.started_at || "").localeCompare(a.started_at || ""));
  const active = runs.filter((r) => r.status === "running" || r.status === "evaluating");
  const shown = active.length ? active : runs.slice(0, 1);
  setHtml($("attack-progress"), shown.map((run) => {
    const pct = Math.round((100 * (run.transaction_count || 0)) / Math.max(run.planned_steps || 1, 1));
    return `<div class="attack-run">
      <div class="attack-run__head"><strong>${escapeHtml(run.attack_run_id)} · ${escapeHtml(run.attack_name)}</strong><span>${run.transaction_count || 0}/${run.planned_steps} tx</span></div>
      <div class="progress"><div class="progress__bar ${run.detected_at ? "is-detected" : ""}" style="width:${pct}%"></div></div>
      <div class="attack-run__chips">${runStatusChips(run)}</div>
    </div>`;
  }).join("") || `<div class="empty-state">No attack runs yet.</div>`);

  setText("history-meta", `${runs.length} run(s) loaded · stored in attack_runs`);
  setHtml($("attack-history"), table([
    { label: "Run", render: (r) => `<button class="link-btn" data-run="${escapeHtml(r.attack_run_id)}" type="button">${escapeHtml(r.attack_run_id)}</button>` },
    { label: "Pattern", render: (r) => escapeHtml(r.attack_name) },
    { label: "Started", render: (r) => fmtTime(r.started_at) },
    { label: "Tx", render: (r) => `${r.transaction_count}/${r.planned_steps}` },
    { label: "Mules detected", render: (r) => (r.detected_accounts ? `${r.detected_accounts.length}/${r.labeled_participants.length}` : "--") },
    { label: "First warning", render: (r) => fmtMs(r.first_warning_latency_ms) },
    { label: "Detection", render: (r) => fmtMs(r.detection_latency_ms) },
    { label: "Precision", render: (r) => fmtScore(r.precision, 2) },
    { label: "Recall", render: (r) => fmtScore(r.recall, 2) },
    { label: "Threshold", render: (r) => (r.threshold_after !== null && r.threshold_after !== undefined ? `${fmtScore(r.threshold_before)}→${fmtScore(r.threshold_after)}` : "--") },
    { label: "Status", render: (r) => escapeHtml(r.status) },
  ], runs.slice(0, 30), "No attacks spawned yet."));

  const last = store.lastCompletedRun;
  if (last && last.detected_accounts) {
    setText("summary-pattern", `${last.attack_run_id} · ${last.attack_name} — evaluated against the simulator's injected mule list.`);
    setText("summary-injected", last.labeled_participants.length);
    setText("summary-detected", last.detected_accounts.length);
    setText("summary-missed", last.missed_accounts.length);
    setText("summary-warning", fmtMs(last.first_warning_latency_ms));
    setText("summary-latency", fmtMs(last.detection_latency_ms));
    setText("summary-fp", last.false_positive_accounts.length);
    setText("adaptive-precision", fmtScore(last.precision, 2));
    setText("adaptive-recall", fmtScore(last.recall, 2));
    const group = (title, ids, tone) => `<div class="structured-item"><div class="structured-item__head"><strong>${title}</strong><span class="structured-item__count">${ids.length}</span></div><div class="token-stream">${ids.map((id) => `<button class="token-pill token-pill--${tone}" data-account="${escapeHtml(id)}" type="button">${escapeHtml(id)}</button>`).join("") || '<span class="empty-state">none</span>'}</div></div>`;
    setHtml($("summary-lists"), [
      group("Detected mules", last.detected_accounts, "danger"),
      group("Missed mules", last.missed_accounts, "warning"),
      group("False positives since previous evaluation", last.false_positive_accounts, "neutral"),
      group("Already in early warning when the attack started", last.warned_before_start || [], "success"),
    ].join(""));
  }
  const pm = store.patternMemory || {};
  setText("pattern-stored", fmtInt(pm.stored));
  if (pm.last) {
    setText("pattern-similarity", fmtPct(pm.last.similarity));
    const s = pm.last.similarity;
    const verdict = s > 0.7 ? "Repeat attack structure detected." : s > 0.4 ? "Variant of a known pattern." : "New fraud pattern.";
    setText("pattern-message", `${pm.last.attack_run_id} (${pm.last.attack_type.replace(/_/g, " ")}): ${verdict}${pm.last.matched_run_id ? ` Closest stored: ${pm.last.matched_run_id}.` : ""}`);
  }
}

// --------------------------------------------------------------------- charts
const CHART_COLORS = {
  teal: "rgba(43, 217, 176, 0.75)",
  amber: "rgba(255, 188, 77, 0.8)",
  red: "rgba(255, 77, 109, 0.8)",
  magenta: "rgba(255, 77, 255, 0.75)",
  blue: "rgba(47, 125, 255, 0.75)",
  grey: "rgba(150, 177, 162, 0.6)",
};

function baseChartOptions(extra = {}) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { labels: { color: "#96b1a2", boxWidth: 12 } } },
    scales: {
      x: { ticks: { color: "#96b1a2", maxRotation: 0, autoSkip: true, maxTicksLimit: 8 }, grid: { color: "rgba(172,210,191,0.06)" } },
      y: { ticks: { color: "#96b1a2" }, grid: { color: "rgba(172,210,191,0.08)" }, beginAtZero: true },
    },
    ...extra,
  };
}

function upsertChart(id, type, data, options) {
  if (!window.Chart) return;
  const existing = ui.charts[id];
  if (existing) {
    existing.data.labels = data.labels;
    data.datasets.forEach((ds, i) => {
      if (existing.data.datasets[i]) Object.assign(existing.data.datasets[i], ds);
      else existing.data.datasets.push(ds);
    });
    existing.data.datasets.length = data.datasets.length;
    existing.update("none");
    return;
  }
  ui.charts[id] = new Chart($(id), { type, data, options });
}

function renderCharts() {
  const m = store.metrics;
  if (!m) return;
  const dist = m.risk_distribution;
  upsertChart("risk-chart", "bar", {
    labels: dist.edges.slice(0, -1).map((e) => e.toFixed(1)),
    datasets: [
      { label: "Early-warning score", data: dist.early, backgroundColor: CHART_COLORS.amber },
      { label: "Fused detection score", data: dist.fused, backgroundColor: CHART_COLORS.red },
    ],
  }, baseChartOptions({ scales: { x: { ticks: { color: "#96b1a2" } }, y: { type: "logarithmic", ticks: { color: "#96b1a2" } } } }));

  const vs = m.volume_series;
  upsertChart("volume-chart", "bar", {
    labels: vs.normal.map((_, i) => new Date((vs.start + i * vs.bucket_sec) * 1000).toLocaleTimeString([], { hour12: false })),
    datasets: [
      { label: "Normal", data: vs.normal, backgroundColor: CHART_COLORS.teal, stack: "v" },
      { label: "Attack (injected)", data: vs.attack, backgroundColor: CHART_COLORS.magenta, stack: "v" },
    ],
  }, baseChartOptions({ scales: { x: { stacked: true, ticks: { color: "#96b1a2", maxTicksLimit: 6 } }, y: { stacked: true, ticks: { color: "#96b1a2" } } } }));

  const channels = Object.keys(m.channel_counts).sort();
  upsertChart("channel-chart", "doughnut", {
    labels: channels,
    datasets: [{ data: channels.map((c) => m.channel_counts[c]), backgroundColor: [CHART_COLORS.teal, CHART_COLORS.blue, CHART_COLORS.amber, CHART_COLORS.magenta, CHART_COLORS.grey] }],
  }, { responsive: true, maintainAspectRatio: false, animation: false, plugins: { legend: { position: "right", labels: { color: "#96b1a2" } } } });

  const types = Object.keys(m.attack_types).sort();
  upsertChart("attack-type-chart", "bar", {
    labels: types.map((t) => t.replace(/_/g, " ")),
    datasets: [
      { label: "Runs", data: types.map((t) => m.attack_types[t].runs), backgroundColor: CHART_COLORS.blue },
      { label: "Detected", data: types.map((t) => m.attack_types[t].detected), backgroundColor: CHART_COLORS.red },
    ],
  }, baseChartOptions());

  const runs = store.history.runs;
  upsertChart("threshold-chart", "line", {
    labels: runs.map((r) => r.attack_run_id),
    datasets: [
      { label: "Detection threshold", data: runs.map((r) => r.threshold_after), borderColor: CHART_COLORS.amber, backgroundColor: CHART_COLORS.amber, tension: 0.25 },
      { label: "Recall", data: runs.map((r) => r.recall), borderColor: CHART_COLORS.teal, backgroundColor: CHART_COLORS.teal, tension: 0.25 },
      { label: "Precision", data: runs.map((r) => r.precision), borderColor: CHART_COLORS.red, backgroundColor: CHART_COLORS.red, tension: 0.25 },
    ],
  }, baseChartOptions({ scales: { x: { ticks: { color: "#96b1a2" } }, y: { min: 0, max: 1, ticks: { color: "#96b1a2" } } } }));

  upsertChart("latency-chart", "bar", {
    labels: runs.map((r) => r.attack_run_id),
    datasets: [
      { label: "First warning (ms)", data: runs.map((r) => r.first_warning_latency_ms), backgroundColor: CHART_COLORS.amber },
      { label: "Detection (ms)", data: runs.map((r) => r.detection_latency_ms), backgroundColor: CHART_COLORS.red },
    ],
  }, baseChartOptions());

  const roles = Object.keys(m.role_counts).sort();
  upsertChart("role-chart", "bar", {
    labels: roles,
    datasets: [{ label: "Flagged accounts", data: roles.map((r) => m.role_counts[r]), backgroundColor: CHART_COLORS.red }],
  }, baseChartOptions({ indexAxis: "y", plugins: { legend: { display: false } } }));
}

// ---------------------------------------------------------------------- graphs
const graphs = { live: null, attack: null };

function initGraphs() {
  if (!window.THREE || !window.LiveGraph3D) {
    $("live-network").innerHTML = '<div class="empty-state">Three.js failed to load (no internet?). Data panels still work.</div>';
    return;
  }
  graphs.live = new LiveGraph3D($("live-network"), { layout: "sphere", autoRotate: false, nodeRadius: 6 });
  graphs.attack = new LiveGraph3D($("attack-network"), { layout: "cluster", background: false, cameraDistance: 520, nodeRadius: 7, fog: 0 });
  const tooltip = $("graph-tooltip");
  graphs.live.on("hover", (hit) => {
    if (!hit) {
      tooltip.classList.add("hidden");
      return;
    }
    const a = store.accounts.get(hit.id);
    if (!a) return;
    tooltip.innerHTML = `<strong>${escapeHtml(a.account_id)}</strong> ${statusPill(a.status)}<br>
      warning ${fmtScore(a.early_risk)} · fused ${fmtScore(a.fused_score)}<br>
      RF ${fmtScore(a.rf_score)} · GNN ${fmtScore(a.gnn_score)} · rules ${a.rule_score ?? "--"}<br>
      ${a.status === "fraud" ? `role ${escapeHtml(a.role)}<br>` : ""}${escapeHtml((a.reasons || []).slice(0, 2).join(", "))}`;
    tooltip.style.left = `${hit.x + 14}px`;
    tooltip.style.top = `${hit.y + 14}px`;
    tooltip.classList.remove("hidden");
  });
  graphs.live.on("click", (id) => openInvestigation(id, true));
  graphs.attack.on("click", (id) => openInvestigation(id, true));
}

function rebuildLiveGraph() {
  if (!graphs.live) return;
  graphs.live.clear();
  store.accounts.forEach((a) => graphs.live.upsertNode(a.account_id, { status: a.status, active: store.degree.has(a.account_id), force: true }));
  store.edges.forEach((edge) => graphs.live.upsertEdge(edge));
}

function syncActive(ids) {
  if (!graphs.live) return;
  ids.forEach((id) => graphs.live.upsertNode(id, { active: store.degree.has(id) }));
}

function addSubgraphStep(step, animate) {
  const g = graphs.attack;
  if (!g) return;
  for (const id of [step.sender, step.receiver]) {
    g.upsertNode(id, { status: step.statuses?.[id] || store.accounts.get(id)?.status || "normal", active: true });
  }
  g.upsertEdge({ id: step.transaction_id, source: step.sender, target: step.receiver, count: 1, attack_count: 1 });
  if (animate) g.pulse(step.sender, step.receiver, "attack", 650);
  const run = store.runs.get(ui.subgraph.runId);
  setText("subgraph-meta", `${g.edges.size}/${run?.planned_steps ?? "?"} tx · ${g.ids.length} accounts`);
}

async function loadRunIntoSubgraph(runId, { animate = true, fresh = false } = {}) {
  const g = graphs.attack;
  ui.subgraph.runId = runId;
  const token = ++ui.subgraph.token;
  const run = store.runs.get(runId);
  setText("subgraph-title", run ? `${run.attack_run_id} · ${run.attack_name}` : runId);
  $("replay-btn").disabled = false;
  if (!g) return;
  g.clear();
  setText("subgraph-meta", "0 tx");
  if (fresh) return; // a just-started run fills in from live attack_step events
  let detail;
  try {
    detail = await api(`/api/attacks/${encodeURIComponent(runId)}`);
  } catch (error) {
    showToast(`Could not load ${runId}: ${error.message}`, "error");
    return;
  }
  if (token !== ui.subgraph.token) return;
  // Detector status of each account right after each stored transaction.
  const statusAfter = new Map();
  detail.detections.forEach((row) => {
    if (!statusAfter.has(row.transaction_id)) statusAfter.set(row.transaction_id, {});
    statusAfter.get(row.transaction_id)[row.account_id] = row.status;
  });
  ui.subgraph.replaying = animate;
  for (const tx of detail.transactions) {
    if (token !== ui.subgraph.token) return;
    addSubgraphStep({ ...tx, statuses: animate ? statusAfter.get(tx.transaction_id) : null }, animate);
    if (animate) await new Promise((resolve) => setTimeout(resolve, 600));
  }
  ui.subgraph.replaying = false;
  if (token === ui.subgraph.token) g.ids.forEach((id) => g.upsertNode(id, { status: store.accounts.get(id)?.status || "normal" }));
}

// -------------------------------------------------------------- investigation
function touchInvestigation(ids) {
  const current = ui.investigation.id;
  if (!current || !ids.includes(current)) return;
  if (ui.investigation.refreshTimer) return;
  const wait = Math.max(0, INVESTIGATION_REFRESH_MS - (Date.now() - ui.investigation.lastFetch));
  ui.investigation.refreshTimer = setTimeout(() => {
    ui.investigation.refreshTimer = null;
    openInvestigation(current, false);
  }, wait);
}

async function openInvestigation(accountId, scroll) {
  const id = String(accountId || "").trim().toUpperCase();
  if (!id) return;
  ui.investigation.id = id;
  ui.investigation.lastFetch = Date.now();
  if (scroll) $("investigation-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  let detail;
  try {
    detail = await api(`/api/accounts/${encodeURIComponent(id)}`);
  } catch (error) {
    showToast(`Investigation failed: ${error.message}`, "error");
    return;
  }
  if (ui.investigation.id !== id) return;
  ui.investigation.data = detail;
  $("investigation-input").value = id;
  renderInvestigation(detail);
}

function renderInvestigation(d) {
  $("investigation-empty").classList.add("hidden");
  $("investigation-body").classList.remove("hidden");
  const a = d.account;
  setText("inv-account", a.account_id);
  setHtml($("inv-status"), statusPill(a.status));
  setText("inv-fused", `${fmtScore(d.scores.fused)} / ${fmtScore(d.scores.threshold)}`);
  setText("inv-role", a.role);
  setText("inv-rf", fmtScore(d.scores.rf));
  setText("inv-gnn", fmtScore(d.scores.gnn));
  setText("inv-rule", `${d.rules.score}/${d.rules.max_score}`);
  setText("inv-early", `${fmtScore(d.scores.early_risk)} / ${fmtScore(d.scores.warning_threshold)}`);
  setHtml($("inv-profile"), [
    `created ${escapeHtml(new Date(a.created_at).toLocaleDateString())} (${escapeHtml(a.origin)})`,
    `device ${escapeHtml(a.device_id)} shared by ${a.device_members.length}`,
    `home channel ${escapeHtml(a.home_channel)}`,
    `balance ${fmtMoney(a.balance)}`,
    `${fmtInt(d.totals.n)} stored tx · sent ${fmtMoney(d.totals.sent)} · received ${fmtMoney(d.totals.received)}`,
    a.flagged_at ? `flagged ${fmtTime(a.flagged_at)}` : "not flagged",
  ].map((s) => `<span>${s}</span>`).join(""));

  const explanation = d.explanation_at_detection?.available ? d.explanation_at_detection : d.explanation_now;
  const reasons = explanation?.available ? explanation.explanations : [];
  setHtml($("inv-explanations"), reasons.length
    ? reasons.map((r) => `<li>${escapeHtml(r)}</li>`).join("") + `<li class="small-note">SHAP on the Random Forest, ${d.explanation_at_detection?.available ? "computed when the account was flagged" : "computed on the current window"}.</li>`
    : `<li>${escapeHtml(explanation?.reason || "No window activity to explain.")}</li>`);
  setHtml($("inv-rules"), Object.entries(d.rules.flags).map(([k, fired]) => `<span class="token-pill ${fired ? "token-pill--danger" : ""}">${escapeHtml(k.replace(/_/g, " "))}${fired ? " ✓" : ""}</span>`).join(""));
  const signals = Object.entries(d.early_warning.signals || {});
  setHtml($("inv-signals"), signals.length ? signals.map(([k, v]) => `<span class="token-pill token-pill--warning">${escapeHtml(k.replace(/_/g, " "))} ${fmtScore(v, 2)}</span>`).join("") : '<span class="empty-state">no active signals</span>');

  const features = Object.entries(d.features);
  const shapByFeature = new Map((explanation?.features || []).map((f) => [f.feature, f.shap]));
  setHtml($("inv-features"), table([
    { label: "Feature", render: (r) => escapeHtml(r[0].replace(/_/g, " ")) },
    { label: "Now", render: (r) => escapeHtml(typeof r[1] === "number" ? Number(r[1].toFixed(3)).toLocaleString("en-IN") : r[1]) },
    { label: "Baseline", render: (r) => (d.drift.baseline && d.drift.baseline[r[0]] !== undefined ? escapeHtml(Number(d.drift.baseline[r[0]].toFixed(3)).toLocaleString("en-IN")) : "--") },
    { label: "SHAP", render: (r) => (shapByFeature.has(r[0]) ? fmtScore(shapByFeature.get(r[0])) : "--") },
  ], features, "No features"));

  setHtml($("inv-transactions"), table([
    { label: "Tx", render: (r) => escapeHtml(r.transaction_id) },
    { label: "Time", render: (r) => fmtTime(r.timestamp) },
    { label: "Dir", render: (r) => (r.sender === a.account_id ? "out →" : "← in") },
    { label: "Counterparty", render: (r) => accountLink(r.sender === a.account_id ? r.receiver : r.sender) },
    { label: "Amount", render: (r) => fmtMoney(r.amount) },
    { label: "Channel", render: (r) => escapeHtml(r.channel) },
    { label: "Source", render: (r) => (r.is_attack ? `<span class="tag tag--attack">${escapeHtml(r.attack_run_id)}</span>` : escapeHtml(r.source)) },
  ], d.transactions.slice(0, 25), "No stored transactions."));

  setHtml($("inv-neighbors"), table([
    { label: "Account", render: (r) => accountLink(r.account_id) },
    { label: "Dir", render: (r) => (r.direction === "out" ? "sent to" : "received from") },
    { label: "Tx", key: "count" },
    { label: "Amount", render: (r) => fmtMoney(r.amount) },
    { label: "Status", render: (r) => statusPill(r.status) },
  ], d.window_neighbors, "No counterparties inside the current window."));

  setHtml($("inv-alerts"), table([
    { label: "#", key: "alert_id" },
    { label: "Time", render: (r) => fmtTime(r.timestamp) },
    { label: "Type", render: (r) => escapeHtml(r.alert_type.replace(/_/g, " ")) },
    { label: "Severity", render: (r) => `<span class="sev sev--${escapeHtml(r.severity)}">${escapeHtml(r.severity)}</span>` },
  ], d.alerts, "No alerts for this account."));

  const gt = d.simulation_ground_truth.attack_runs;
  setHtml($("inv-ground-truth"), gt.length
    ? gt.map((r) => `${escapeHtml(r.attack_run_id)} (${escapeHtml(r.attack_name)}) — ${r.labeled_mule ? "injected as a mule" : "touched as a counterparty"}`).join("<br>")
    : "Not part of any simulated attack run.");

  const history = d.risk_history;
  upsertChart("inv-risk-chart", "line", {
    labels: history.map((h) => fmtTime(h.timestamp)),
    datasets: [
      { label: "Fused", data: history.map((h) => h.fused_score), borderColor: CHART_COLORS.red, backgroundColor: CHART_COLORS.red, tension: 0.2, pointRadius: 2 },
      { label: "RF", data: history.map((h) => h.rf_score), borderColor: CHART_COLORS.blue, backgroundColor: CHART_COLORS.blue, tension: 0.2, pointRadius: 1 },
      { label: "GNN", data: history.map((h) => h.gnn_score), borderColor: CHART_COLORS.magenta, backgroundColor: CHART_COLORS.magenta, tension: 0.2, pointRadius: 1 },
      { label: "Early warning", data: history.map((h) => h.early_risk), borderColor: CHART_COLORS.amber, backgroundColor: CHART_COLORS.amber, tension: 0.2, pointRadius: 1 },
    ],
  }, baseChartOptions({ scales: { x: { ticks: { color: "#96b1a2", maxTicksLimit: 6 } }, y: { min: 0, max: 1, ticks: { color: "#96b1a2" } } } }));

  const categories = explanation?.available ? explanation.categories : [];
  upsertChart("inv-shap-chart", "bar", {
    labels: categories.map((c) => c.category),
    datasets: [{ label: "|SHAP| contribution (×100)", data: categories.map((c) => c.contribution), backgroundColor: CHART_COLORS.red }],
  }, baseChartOptions({ indexAxis: "y", plugins: { legend: { display: false } } }));
}

// -------------------------------------------------------------------- actions
function populateAttackTypes() {
  const select = $("attack-type");
  const current = select.value;
  select.innerHTML = store.attackTypes.map((t) => `<option value="${escapeHtml(t.key)}">${escapeHtml(t.name)}</option>`).join("");
  if (current && store.attackTypes.some((t) => t.key === current)) select.value = current;
  describeAttack();
}

function describeAttack() {
  const type = store.attackTypes.find((t) => t.key === $("attack-type").value);
  setText("attack-description", type ? type.description : "");
}

function populateAccountOptions() {
  $("account-options").innerHTML = [...store.accounts.keys()].map((id) => `<option value="${escapeHtml(id)}">`).join("");
}

function appendAccountOption(id) {
  const option = document.createElement("option");
  option.value = id;
  $("account-options").appendChild(option);
}

async function spawnAttack() {
  const button = $("spawn-attack-btn");
  const attackType = $("attack-type").value;
  button.disabled = true;
  try {
    const response = await api("/api/attacks/spawn", { method: "POST", body: JSON.stringify({ attack_type: attackType }) });
    showToast(`Backend accepted ${response.attack_run.attack_run_id}`, "info");
    graphs.live?.focus(response.attack_run.participants, 1100);
  } catch (error) {
    showToast(`Spawn failed: ${error.message}`, "error");
  } finally {
    setTimeout(() => {
      button.disabled = !(conn.socket && conn.hydrated);
    }, 400);
  }
}

async function banSelected() {
  const ids = [...ui.selectedForBan];
  if (!ids.length) return;
  $("ban-btn").disabled = true;
  try {
    await api("/api/accounts/ban", { method: "POST", body: JSON.stringify({ account_ids: ids }) });
  } catch (error) {
    showToast(`Ban failed: ${error.message}`, "error");
  }
}

async function toggleSpawner() {
  const running = store.metrics?.spawner_running;
  try {
    await api("/api/spawner", { method: "POST", body: JSON.stringify({ running: !running }) });
  } catch (error) {
    showToast(`Spawner control failed: ${error.message}`, "error");
  }
}

async function resetDatabase() {
  if (!window.confirm("Delete every stored transaction, alert and attack run, and re-seed accounts?")) return;
  try {
    await api("/api/admin/reset", { method: "POST", body: JSON.stringify({ confirm: true }) });
  } catch (error) {
    showToast(`Reset failed: ${error.message}`, "error");
  }
}

function flash() {
  document.body.classList.add("is-attack-flash");
  clearTimeout(ui.flashTimer);
  ui.flashTimer = setTimeout(() => document.body.classList.remove("is-attack-flash"), 900);
}

function playSiren() {
  if (ui.sirenMuted) return;
  try {
    ui.audio = ui.audio || new (window.AudioContext || window.webkitAudioContext)();
    const ctx = ui.audio;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sawtooth";
    osc.frequency.setValueAtTime(680, ctx.currentTime);
    for (let i = 0; i < 4; i += 1) {
      osc.frequency.linearRampToValueAtTime(980, ctx.currentTime + i * 0.5 + 0.25);
      osc.frequency.linearRampToValueAtTime(680, ctx.currentTime + i * 0.5 + 0.5);
    }
    gain.gain.setValueAtTime(0.0001, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.05, ctx.currentTime + 0.05);
    gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 2);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 2.05);
  } catch (_) {
    // audio is optional
  }
}

function bindEvents() {
  $("spawn-attack-btn").addEventListener("click", spawnAttack);
  $("attack-type").addEventListener("change", describeAttack);
  $("ban-btn").addEventListener("click", banSelected);
  $("spawner-btn").addEventListener("click", toggleSpawner);
  $("reset-btn").addEventListener("click", resetDatabase);
  $("replay-btn").addEventListener("click", () => ui.subgraph.runId && loadRunIntoSubgraph(ui.subgraph.runId, { animate: true }));
  $("mute-siren-btn").addEventListener("click", () => {
    ui.sirenMuted = !ui.sirenMuted;
    setText("mute-siren-btn", ui.sirenMuted ? "Unmute Siren" : "Mute Siren");
  });
  $("select-all-flagged-btn").addEventListener("click", () => {
    store.accounts.forEach((a) => {
      if (a.status === "fraud") ui.selectedForBan.add(a.account_id);
    });
    markDirty("accounts");
  });
  $("investigation-go").addEventListener("click", () => openInvestigation($("investigation-input").value, false));
  $("investigation-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter") openInvestigation(event.target.value, false);
  });
  document.addEventListener("click", (event) => {
    const accountBtn = event.target.closest("[data-account]");
    if (accountBtn) {
      openInvestigation(accountBtn.dataset.account, true);
      return;
    }
    const runBtn = event.target.closest("[data-run]");
    if (runBtn) {
      loadRunIntoSubgraph(runBtn.dataset.run, { animate: true });
      $("attack-network-frame").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
  document.addEventListener("change", (event) => {
    const box = event.target.closest("[data-ban]");
    if (!box) return;
    if (box.checked) ui.selectedForBan.add(box.dataset.ban);
    else ui.selectedForBan.delete(box.dataset.ban);
    markDirty("accounts");
  });
}

function boot() {
  bindEvents();
  initGraphs();
  connect();
  startKeepAlive();
}

boot();
