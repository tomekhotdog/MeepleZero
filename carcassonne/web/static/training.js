// training.js — the training view: a live dashboard for one AlphaZero
// TrainingRun. Loss curves, arena win-rates (with the 0.55 promotion gate),
// buffer growth, and a checkpoint timeline — all drawn on canvas, no chart lib.
//
// Data source is the API exactly as produced by web/app.py:
//   GET /api/runs          -> {runs: [{name, config, iterations_or_steps,
//                                       n_checkpoints, last_step, last_gate}]}
//   GET /api/runs/{name}   -> {config, checkpoints:[{step,file}],
//                              metrics:{learn:[...], gate:[...]}}
// learn rows: {step, loss, policy_loss, value_loss, buffer_games, buffer_examples}
// gate  rows: {iter, candidate_step, promoted, wr_best, wr_greedy}

import { resizeToDisplay } from "./board.js";

const POLL_MS = 10_000;
const GATE_LINE = 0.55; // the promotion win-rate threshold (AlphaGo-Zero gating)

// Design tokens mirrored from style.css so canvas drawing matches the chrome.
const C = {
  bone: "#e8e2d4",
  dim: "#9ba3ad",
  ai: "#5fd4c4",
  gold: "#c9a25e",
  grid: "rgba(155, 163, 173, 0.22)",
  axis: "rgba(155, 163, 173, 0.8)",
};

// --- state -------------------------------------------------------------------

const T = {
  name: null, // selected run name
  detail: null, // full /api/runs/{name} response
  timer: null,
};

const $ = (id) => document.getElementById(id);
const select = $("run-select");

// --- server calls ------------------------------------------------------------

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) {
    const err = new Error(`HTTP ${r.status}`);
    err.status = r.status;
    err.body = await r.json().catch(() => null);
    throw err;
  }
  return r.json();
}

// --- run picker --------------------------------------------------------------

function describeRun(r) {
  const step = r.last_step ?? "—";
  const gate = r.last_gate ? ` · gate wr ${(r.last_gate.wr_greedy * 100).toFixed(0)}%` : "";
  return `${r.name} · step ${step} · ${r.n_checkpoints} ckpt${gate}`;
}

async function loadRunList() {
  let list;
  try {
    ({ runs: list } = await api("/api/runs"));
  } catch (err) {
    showError(`Could not list training runs — ${err.message || err}`);
    return [];
  }
  select.replaceChildren();
  if (list.length === 0) {
    select.append(new Option("No training runs found", ""));
    showEmpty(
      "No training runs found. Point serve at a runs dir with --runs-dir <dir>."
    );
    return [];
  }
  const placeholder = new Option("Choose a run…", "");
  placeholder.disabled = true;
  placeholder.selected = true;
  select.append(placeholder);
  for (const r of list) select.append(new Option(describeRun(r), r.name));
  return list;
}

async function loadRun(name, { silent = false } = {}) {
  if (!silent) showError("");
  let detail;
  try {
    detail = await api(`/api/runs/${encodeURIComponent(name)}`);
  } catch (err) {
    if (!silent) {
      const msg = err.body?.explanation || err.body?.error || err.message || String(err);
      showError(`Could not load run — ${msg}`);
    }
    return;
  }
  T.name = name;
  T.detail = detail;
  $("empty-hint").style.display = "none";
  $("charts").hidden = false;
  renderSidebar(detail);
  markUpdated();
  drawAll();
}

// --- sidebar -----------------------------------------------------------------

function renderSidebar(detail) {
  const learn = detail.metrics.learn;
  const gate = detail.metrics.gate;
  const last = learn.length ? learn[learn.length - 1] : null;
  const lastGate = gate.length ? gate[gate.length - 1] : null;
  const promotions = gate.filter((g) => g.promoted).length;

  const rows = [
    ["steps", last ? last.step : "—"],
    ["loss", last ? last.loss.toFixed(4) : "—"],
    ["buffer games", last ? last.buffer_games : "—"],
    ["examples", last ? last.buffer_examples : "—"],
    ["checkpoints", detail.checkpoints.length],
    ["gates", `${gate.length} (${promotions} promoted)`],
    ["last wr vs best", lastGate ? pct(lastGate.wr_best) : "—"],
    ["last wr vs greedy", lastGate ? pct(lastGate.wr_greedy) : "—"],
  ];
  $("summary-body").replaceChildren(...rows.map(kvRow));
  $("summary-panel").hidden = false;

  const t = (detail.config && detail.config.train) || {};
  const a = (detail.config && detail.config.net_arch) || {};
  const cfg = [
    ["channels", a.channels],
    ["blocks", a.n_blocks],
    ["sims", t.mcts_sims],
    ["selfplay/iter", t.selfplay_games_per_iter],
    ["learn/iter", t.learn_steps_per_iter],
    ["gate every", t.gate_every],
    ["gate games", t.gate_games],
  ].filter(([, v]) => v !== undefined);
  $("config-body").replaceChildren(...cfg.map(kvRow));
  $("config-panel").hidden = cfg.length === 0;
}

function kvRow([k, v]) {
  const row = document.createElement("div");
  row.className = "kv";
  const key = document.createElement("span");
  key.className = "kv-k";
  key.textContent = k;
  const val = document.createElement("span");
  val.className = "kv-v";
  val.textContent = String(v);
  row.append(key, val);
  return row;
}

function pct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function markUpdated() {
  const now = new Date().toLocaleTimeString();
  $("updated").textContent = `updated ${now} · polling every ${POLL_MS / 1000}s`;
  $("live-panel").hidden = false;
}

// --- generic canvas line chart ----------------------------------------------

// opts: {series:[{pts:[[x,y]...], color, width, dashed}], hlines, markers,
//        xMin,xMax, yMin,yMax, yFmt, empty}
function drawChart(canvas, opts) {
  const ctx = canvas.getContext("2d");
  const { w, h } = resizeToDisplay(canvas, ctx);
  ctx.clearRect(0, 0, w, h);

  const padL = 40;
  const padR = 10;
  const padT = 10;
  const padB = 20;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;

  const series = opts.series || [];
  const hasData = series.some((s) => s.pts.length > 0) || (opts.markers || []).length > 0;
  if (!hasData) {
    ctx.fillStyle = C.dim;
    ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(opts.empty || "No data yet.", w / 2, h / 2);
    return;
  }

  const { xMin, xMax, yMin, yMax } = opts;
  const xSpan = xMax - xMin || 1;
  const ySpan = yMax - yMin || 1;
  const xAt = (x) => padL + ((x - xMin) / xSpan) * plotW;
  const yAt = (y) => padT + (1 - (y - yMin) / ySpan) * plotH;
  const yFmt = opts.yFmt || ((v) => String(Math.round(v)));

  // y grid + labels (min / mid / max).
  ctx.font = "9px ui-monospace, monospace";
  ctx.fillStyle = C.axis;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const frac of [0, 0.5, 1]) {
    const yv = yMin + frac * ySpan;
    const py = yAt(yv);
    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
    ctx.stroke();
    ctx.fillText(yFmt(yv), padL - 5, py);
  }

  // x labels (min / max).
  ctx.fillStyle = C.axis;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(String(Math.round(xMin)), padL, padT + plotH + 4);
  ctx.textAlign = "right";
  ctx.fillText(String(Math.round(xMax)), padL + plotW, padT + plotH + 4);

  // Reference horizontal lines (e.g. the 0.55 promotion gate).
  for (const hl of opts.hlines || []) {
    const py = yAt(hl.y);
    ctx.strokeStyle = hl.color;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, py);
    ctx.lineTo(padL + plotW, py);
    ctx.stroke();
    ctx.setLineDash([]);
    if (hl.label) {
      ctx.fillStyle = hl.color;
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(hl.label, padL + 2, py - 1);
    }
  }

  // Series lines.
  for (const s of series) {
    if (s.pts.length === 0) continue;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 1.5;
    ctx.setLineDash(s.dashed ? [4, 3] : []);
    ctx.beginPath();
    s.pts.forEach(([x, y], i) => {
      const px = xAt(x);
      const py = yAt(y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    // Dots for short series so single points are visible.
    if (s.pts.length <= 60) {
      ctx.fillStyle = s.color;
      for (const [x, y] of s.pts) {
        ctx.beginPath();
        ctx.arc(xAt(x), yAt(y), 2, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  }

  // Emphasis markers (e.g. promoted gates).
  for (const m of opts.markers || []) {
    ctx.fillStyle = m.color;
    ctx.beginPath();
    ctx.arc(xAt(m.x), yAt(m.y), m.r || 3.5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = C.bone;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}

// --- legends -----------------------------------------------------------------

function setLegend(id, items) {
  const el = $(id);
  el.replaceChildren(
    ...items.map(({ color, label, dashed }) => {
      const span = document.createElement("span");
      span.className = "legend-item";
      const dot = document.createElement("span");
      dot.className = "legend-dot";
      dot.style.background = dashed ? "transparent" : color;
      if (dashed) dot.style.borderTop = `2px dashed ${color}`;
      span.append(dot, document.createTextNode(label));
      return span;
    })
  );
}

// --- the four charts ---------------------------------------------------------

function drawAll() {
  if (!T.detail) return;
  drawLoss();
  drawArena();
  drawGames();
  drawCheckpoints();
}

function xRange(values, fallback = 1) {
  if (values.length === 0) return [0, fallback];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  return [lo, hi === lo ? lo + fallback : hi];
}

function drawLoss() {
  const learn = T.detail.metrics.learn;
  const steps = learn.map((r) => r.step);
  const [xMin, xMax] = xRange(steps);
  const all = learn.flatMap((r) => [r.loss, r.policy_loss, r.value_loss]);
  const yMax = all.length ? Math.max(...all) * 1.05 : 1;
  drawChart($("loss-chart"), {
    series: [
      { pts: learn.map((r) => [r.step, r.loss]), color: C.bone, width: 1.8 },
      { pts: learn.map((r) => [r.step, r.policy_loss]), color: C.dim },
      { pts: learn.map((r) => [r.step, r.value_loss]), color: C.ai },
    ],
    xMin,
    xMax,
    yMin: 0,
    yMax,
    yFmt: (v) => v.toFixed(2),
    empty: "No learner steps recorded yet.",
  });
  setLegend("loss-legend", [
    { color: C.bone, label: "total" },
    { color: C.dim, label: "policy" },
    { color: C.ai, label: "value" },
  ]);
  $("loss-axis").textContent = learn.length
    ? `step ${xMin} → ${xMax} · lower is better`
    : "";
}

function drawArena() {
  const gate = T.detail.metrics.gate;
  const iters = gate.map((r) => r.iter);
  const [xMin, xMax] = xRange(iters);
  const promoted = gate.filter((r) => r.promoted);
  drawChart($("arena-chart"), {
    series: [
      { pts: gate.map((r) => [r.iter, r.wr_best]), color: C.bone, width: 1.8 },
      { pts: gate.map((r) => [r.iter, r.wr_greedy]), color: C.ai },
    ],
    hlines: [{ y: GATE_LINE, color: C.gold, label: "0.55 gate" }],
    markers: promoted.map((r) => ({ x: r.iter, y: r.wr_best, color: C.ai })),
    xMin,
    xMax,
    yMin: 0,
    yMax: 1,
    yFmt: (v) => v.toFixed(1),
    empty: "No arena gates recorded yet.",
  });
  setLegend("arena-legend", [
    { color: C.bone, label: "vs best" },
    { color: C.ai, label: "vs greedy" },
    { color: C.gold, label: "0.55 gate", dashed: true },
  ]);
  $("arena-axis").textContent = gate.length
    ? `gate iter ${xMin} → ${xMax} · dots = promoted`
    : "";
}

function drawGames() {
  const learn = T.detail.metrics.learn;
  const steps = learn.map((r) => r.step);
  const [xMin, xMax] = xRange(steps);
  const games = learn.map((r) => r.buffer_games);
  const yMax = games.length ? Math.max(...games) * 1.1 || 1 : 1;
  drawChart($("games-chart"), {
    series: [{ pts: learn.map((r) => [r.step, r.buffer_games]), color: C.dim, width: 1.8 }],
    xMin,
    xMax,
    yMin: 0,
    yMax,
    yFmt: (v) => String(Math.round(v)),
    empty: "No buffer data yet.",
  });
  $("games-axis").textContent = learn.length
    ? `step ${xMin} → ${xMax} · games in replay buffer`
    : "";
}

function drawCheckpoints() {
  const ckpts = T.detail.checkpoints;
  const gate = T.detail.metrics.gate;
  const promotedSteps = new Set(gate.filter((g) => g.promoted).map((g) => g.candidate_step));
  const canvas = $("ckpt-chart");
  const ctx = canvas.getContext("2d");
  const { w, h } = resizeToDisplay(canvas, ctx);
  ctx.clearRect(0, 0, w, h);
  const padL = 40;
  const padR = 10;
  const y = h / 2;

  if (ckpts.length === 0) {
    ctx.fillStyle = C.dim;
    ctx.font = "12px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("No checkpoints yet.", w / 2, y);
    $("ckpt-axis").textContent = "";
    setLegend("ckpt-legend", []);
    return;
  }

  const steps = ckpts.map((c) => c.step);
  const lo = Math.min(...steps);
  const hi = Math.max(...steps);
  const span = hi - lo || 1;
  const plotW = w - padL - padR;
  const xAt = (s) => padL + ((s - lo) / span) * plotW;

  // Baseline.
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, y);
  ctx.lineTo(padL + plotW, y);
  ctx.stroke();

  for (const c of ckpts) {
    const promoted = promotedSteps.has(c.step);
    const px = xAt(c.step);
    ctx.strokeStyle = promoted ? C.ai : C.dim;
    ctx.lineWidth = promoted ? 2.5 : 1.5;
    ctx.beginPath();
    ctx.moveTo(px, y - (promoted ? 12 : 7));
    ctx.lineTo(px, y + (promoted ? 12 : 7));
    ctx.stroke();
  }

  ctx.fillStyle = C.axis;
  ctx.font = "9px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(String(lo), padL, y + 16);
  ctx.textAlign = "right";
  ctx.fillText(String(hi), padL + plotW, y + 16);

  setLegend("ckpt-legend", [
    { color: C.dim, label: "checkpoint" },
    { color: C.ai, label: "promoted" },
  ]);
  $("ckpt-axis").textContent = `${ckpts.length} checkpoints · step ${lo} → ${hi}`;
}

// --- polling -----------------------------------------------------------------

function startPolling() {
  stopPolling();
  T.timer = setInterval(() => {
    if (document.hidden || !T.name) return; // don't poll a backgrounded tab
    loadRun(T.name, { silent: true });
  }, POLL_MS);
}

function stopPolling() {
  if (T.timer) clearInterval(T.timer);
  T.timer = null;
}

// --- helpers -----------------------------------------------------------------

function showError(msg) {
  $("run-error").textContent = msg || "";
}

function showEmpty(msg) {
  const hint = $("empty-hint");
  hint.textContent = msg;
  hint.style.display = "";
  $("charts").hidden = true;
}

// --- interactions ------------------------------------------------------------

select.addEventListener("change", () => {
  if (select.value) loadRun(select.value);
});

window.addEventListener("resize", drawAll);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden && T.name) loadRun(T.name, { silent: true });
});

// --- boot --------------------------------------------------------------------

async function boot() {
  const list = await loadRunList();
  const wanted = new URLSearchParams(location.search).get("run");
  if (wanted && list.some((r) => r.name === wanted)) {
    select.value = wanted;
    await loadRun(wanted);
  }
  startPolling();
}

boot();
