// replay.js — the replay view: step through a saved game, board on the left,
// per-move AI search panel + whole-game win-probability chart on the right.
//
// Data source is the API exactly as produced by web/views.py:
//   GET /api/replays          -> {replays: [{name, agents, started_at, final_scores, winner, turns}]}
//   GET /api/replays/{name}   -> {header, moves, final_scores, winner, states}
// `states` holds one entry per position: states[i] is the board BEFORE move i,
// and states[moves.length] is the terminal position. moves[i] is applied to
// states[i] to reach states[i+1].

import { resizeToDisplay, drawTiles, drawFrontier, cellPx, screenToCellFloat } from "./board.js";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const AUTOPLAY_MS = 1000;

// --- state -------------------------------------------------------------------

const R = {
  tiledefs: null,
  data: null, // full /api/replays/{name} response
  index: 0, // current position: 0..moves.length
  winprob: [], // per-move {p: prob for player 0, known: bool}, length moves.length
  cam: { x: 0, y: 0, zoom: 1 },
  drag: null,
  playing: false,
  timer: null,
};

// --- DOM ---------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const board = $("board");
const bctx = board.getContext("2d");
const winCanvas = $("winprob");
const select = $("replay-select");

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

// --- picker ------------------------------------------------------------------

function describeReplay(m) {
  const agents = (m.agents ?? []).join(" vs ");
  const fs = m.final_scores ? `${m.final_scores[0]}–${m.final_scores[1]}` : "?";
  const win = m.winner === null || m.winner === undefined ? "draw" : `P${m.winner} won`;
  const when = (m.started_at ?? "").replace("T", " ").slice(0, 16);
  return `${agents} · ${when} · ${fs} · ${win}`;
}

async function loadReplayList() {
  let list;
  try {
    ({ replays: list } = await api("/api/replays"));
  } catch (err) {
    showError(`Could not list replays — ${err.message || err}`);
    return [];
  }
  select.replaceChildren();
  const placeholder = new Option("Choose a replay…", "");
  placeholder.disabled = true;
  placeholder.selected = true;
  select.append(placeholder);
  for (const m of list) select.append(new Option(describeReplay(m), m.name));
  if (list.length === 0) {
    select.replaceChildren(new Option("No replays found", ""));
  }
  return list;
}

async function loadReplay(name) {
  stopAutoplay();
  showError("");
  try {
    R.data = await api(`/api/replays/${encodeURIComponent(name)}`);
  } catch (err) {
    R.data = null;
    const msg = err.body?.explanation || err.body?.error || err.message || String(err);
    showError(`Could not load replay — ${msg}`);
    hidePanels();
    return;
  }
  R.cam = { x: 0, y: 0, zoom: 1 };
  buildWinProb();
  const M = R.data.moves.length;
  const slider = $("slider");
  slider.max = String(M);
  slider.value = "0";
  $("empty-hint").style.display = "none";
  $("scrubber").hidden = false;
  $("winprob-panel").hidden = false;
  document.querySelector(".p0-name").textContent = `P0 ${R.data.header.agents[0]}`;
  setIndex(0);
}

// --- position / scrubber -----------------------------------------------------

function moves() {
  return R.data ? R.data.moves : [];
}

function setIndex(i) {
  const M = moves().length;
  R.index = Math.max(0, Math.min(M, i));
  $("slider").value = String(R.index);
  $("position").textContent = R.index < M ? `move ${R.index + 1} / ${M}` : `final / ${M}`;
  renderMoveInfo();
  renderSearchPanel();
}

function step(delta) {
  setIndex(R.index + delta);
}

function toggleAutoplay() {
  if (R.playing) stopAutoplay();
  else startAutoplay();
}

function startAutoplay() {
  if (!R.data || R.index >= moves().length) {
    setIndex(0); // replay from the top if we're at the end
  }
  R.playing = true;
  $("play").textContent = "⏸";
  R.timer = setInterval(() => {
    if (R.index >= moves().length) {
      stopAutoplay();
      return;
    }
    step(1);
  }, AUTOPLAY_MS);
}

function stopAutoplay() {
  R.playing = false;
  $("play").textContent = "▶";
  if (R.timer) clearInterval(R.timer);
  R.timer = null;
}

// --- move info ---------------------------------------------------------------

function swatch(player) {
  const s = document.createElement("span");
  s.className = `swatch p${player}`;
  return s;
}

function moveLabel(mv) {
  const a = mv.action;
  let act;
  if (a === null) act = "no meeple";
  else if (a.type === "meeple") act = `${a.kind} f${a.feature}`;
  else act = "retrieve abbot";
  return `(${mv.x},${mv.y}) r${mv.rot} · ${act}`;
}

function renderMoveInfo() {
  const M = moves().length;
  const info = $("move-info");
  if (R.index >= M) {
    info.hidden = true;
    return;
  }
  info.hidden = false;
  const rec = moves()[R.index];
  const who = $("move-who");
  who.replaceChildren(
    swatch(rec.player),
    document.createTextNode(` ${rec.player === 0 ? "blue" : "red"} · ${R.data.header.agents[rec.player]}`)
  );
  $("move-detail").textContent = `${rec.tile} → ${moveLabel(rec.move)}`;

  // Score delta: diff the running scores across this move's two states.
  const before = R.data.states[R.index].scores;
  const after = R.data.states[R.index + 1].scores;
  const delta = $("move-delta");
  delta.replaceChildren();
  const gains = [0, 1].filter((p) => after[p] - before[p] !== 0);
  if (gains.length === 0) {
    delta.textContent = "no score this move";
    delta.className = "muted";
  } else {
    delta.className = "";
    for (const p of gains) {
      const d = after[p] - before[p];
      const chip = document.createElement("span");
      chip.className = `delta-chip p${p}`;
      chip.textContent = `${p === 0 ? "blue" : "red"} ${d >= 0 ? "+" : ""}${d}`;
      delta.append(chip);
    }
  }
}

// --- the search panel --------------------------------------------------------

function sameMove(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

function renderSearchPanel() {
  const panel = $("search-panel");
  const M = moves().length;
  const bars = $("search-bars");
  const note = $("search-note");
  const valueEl = $("search-value");
  const metaEl = $("search-meta");
  bars.replaceChildren();
  valueEl.textContent = "";
  metaEl.textContent = "";
  note.textContent = "";

  if (R.index >= M) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const rec = moves()[R.index];
  const annot = rec.annot;
  if (!annot || annot.top.length === 0) {
    note.textContent = "No search data for this move.";
    return;
  }

  const totalVisits = annot.top.reduce((s, c) => s + c.visits, 0);
  const totalPrior = annot.top.reduce((s, c) => s + c.prior, 0) || 1;
  const heuristic = totalVisits === 0;

  valueEl.textContent = `AI value: ${annot.value >= 0 ? "+" : ""}${annot.value.toFixed(2)}`;
  metaEl.textContent = `${annot.sims} sims · ${annot.think_ms} ms`;

  for (const c of annot.top) {
    const priorFrac = c.prior / totalPrior;
    const visitFrac = heuristic ? 0 : c.visits / totalVisits;
    const row = document.createElement("div");
    row.className = "cand";
    if (sameMove(c.move, rec.move)) row.classList.add("played");
    const label = document.createElement("div");
    label.className = "cand-label mono";
    label.textContent = moveLabel(c.move);
    row.append(label, barRow("prior", "prior", priorFrac), barRow("visits", "visits", visitFrac));
    bars.append(row);
  }

  note.textContent = heuristic
    ? "Prior weights are the greedy heuristic — no tree search (visits 0)."
    : "";
}

function barRow(tag, cls, frac) {
  const row = document.createElement("div");
  row.className = "bar-row";
  const t = document.createElement("span");
  t.className = "bar-tag";
  t.textContent = tag;
  const track = document.createElement("div");
  track.className = "bar-track";
  const fill = document.createElement("div");
  fill.className = `bar-fill ${cls}`;
  fill.style.width = `${(frac * 100).toFixed(1)}%`;
  track.append(fill);
  const pct = document.createElement("span");
  pct.className = "bar-pct mono";
  pct.textContent = `${Math.round(frac * 100)}%`;
  row.append(t, track, pct);
  return row;
}

// --- win-probability chart ---------------------------------------------------

// value is from the MOVING player's perspective in [-1, 1]; convert to a
// player-0 win probability. Moves without an annot carry the last known value
// forward and are drawn dimmed.
function buildWinProb() {
  const out = [];
  let last = 0.5;
  for (const rec of R.data.moves) {
    if (rec.annot) {
      const v = rec.player === 0 ? rec.annot.value : -rec.annot.value;
      last = 0.5 * (1 + v);
      out.push({ p: last, known: true });
    } else {
      out.push({ p: last, known: false });
    }
  }
  R.winprob = out;
}

function drawWinProb() {
  if (!R.data) return;
  const { w, h } = resizeToDisplay(winCanvas, winCanvas.getContext("2d"));
  const ctx = winCanvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);
  const padL = 22;
  const padR = 6;
  const padT = 8;
  const padB = 4;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const n = R.winprob.length;
  const xAt = (i) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (p) => padT + (1 - p) * plotH;

  // Midline (50%) and frame.
  ctx.strokeStyle = "rgba(155, 163, 173, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, yAt(0.5));
  ctx.lineTo(padL + plotW, yAt(0.5));
  ctx.stroke();

  // y labels 0 / .5 / 1 (blue advantage up).
  ctx.fillStyle = "rgba(155, 163, 173, 0.8)";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("1", padL - 4, yAt(1));
  ctx.fillText(".5", padL - 4, yAt(0.5));
  ctx.fillText("0", padL - 4, yAt(0));

  if (n > 0) {
    // Draw segment by segment: dim where either endpoint is interpolated.
    for (let i = 1; i < n; i++) {
      const a = R.winprob[i - 1];
      const b = R.winprob[i];
      ctx.strokeStyle = a.known && b.known ? "#5fd4c4" : "rgba(95, 212, 196, 0.28)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(xAt(i - 1), yAt(a.p));
      ctx.lineTo(xAt(i), yAt(b.p));
      ctx.stroke();
    }
    // Known points as small dots.
    ctx.fillStyle = "#5fd4c4";
    for (let i = 0; i < n; i++) {
      if (!R.winprob[i].known) continue;
      ctx.beginPath();
      ctx.arc(xAt(i), yAt(R.winprob[i].p), 1.6, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  // Current-position marker.
  const markIdx = Math.min(R.index, Math.max(0, n - 1));
  const mx = xAt(markIdx);
  ctx.strokeStyle = "rgba(232, 226, 212, 0.7)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(mx, padT);
  ctx.lineTo(mx, padT + plotH);
  ctx.stroke();
}

function winProbClick(e) {
  const n = R.winprob.length;
  if (n === 0) return;
  const rect = winCanvas.getBoundingClientRect();
  const padL = 22;
  const padR = 6;
  const plotW = rect.width - padL - padR;
  const frac = (e.clientX - rect.left - padL) / plotW;
  setIndex(Math.round(frac * (n - 1)));
}

// --- board rendering ---------------------------------------------------------

function render() {
  const { w, h } = resizeToDisplay(board, bctx);
  bctx.clearRect(0, 0, w, h);
  if (R.data && R.tiledefs) {
    const view = R.data.states[R.index];
    drawTiles(bctx, board, view, R.tiledefs, R.cam);
    drawFrontier(bctx, board, view, R.cam);
    drawWinProb();
  }
  requestAnimationFrame(render);
}

// --- board pan / zoom (view-only) --------------------------------------------

board.addEventListener("pointerdown", (e) => {
  R.drag = { px: e.clientX, py: e.clientY };
  board.setPointerCapture(e.pointerId);
  board.classList.add("dragging");
});

board.addEventListener("pointermove", (e) => {
  if (!R.drag) return;
  const c = cellPx(R.cam);
  R.cam.x -= (e.clientX - R.drag.px) / c;
  R.cam.y += (e.clientY - R.drag.py) / c; // screen y down = world south
  R.drag.px = e.clientX;
  R.drag.py = e.clientY;
});

const endDrag = () => {
  R.drag = null;
  board.classList.remove("dragging");
};
board.addEventListener("pointerup", endDrag);
board.addEventListener("pointercancel", endDrag);

board.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = board.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const before = screenToCellFloat(board, R.cam, px, py);
    R.cam.zoom = Math.min(2.5, Math.max(0.4, R.cam.zoom * Math.exp(-e.deltaY * 0.0012)));
    const after = screenToCellFloat(board, R.cam, px, py);
    R.cam.x += before.x - after.x; // keep the point under the cursor fixed
    R.cam.y += before.y - after.y;
  },
  { passive: false }
);

// --- interactions ------------------------------------------------------------

select.addEventListener("change", () => {
  if (select.value) loadReplay(select.value);
});
$("first").addEventListener("click", () => (stopAutoplay(), setIndex(0)));
$("prev").addEventListener("click", () => (stopAutoplay(), step(-1)));
$("next").addEventListener("click", () => (stopAutoplay(), step(1)));
$("last").addEventListener("click", () => (stopAutoplay(), setIndex(moves().length)));
$("play").addEventListener("click", toggleAutoplay);
$("slider").addEventListener("input", (e) => {
  stopAutoplay();
  setIndex(Number(e.target.value));
});
winCanvas.addEventListener("click", winProbClick);

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, select, textarea")) return;
  if (!R.data) return;
  if (e.key === "ArrowLeft") {
    stopAutoplay();
    step(-1);
  } else if (e.key === "ArrowRight") {
    stopAutoplay();
    step(1);
  } else if (e.key === "Home") {
    stopAutoplay();
    setIndex(0);
  } else if (e.key === "End") {
    stopAutoplay();
    setIndex(moves().length);
  } else if (e.key === " ") {
    e.preventDefault();
    toggleAutoplay();
  } else {
    return;
  }
});

window.addEventListener("resize", drawWinProb);

// --- helpers -----------------------------------------------------------------

function showError(msg) {
  $("replay-error").textContent = msg || "";
}

function hidePanels() {
  for (const id of ["scrubber", "move-info", "search-panel", "winprob-panel"]) {
    $(id).hidden = true;
  }
  $("empty-hint").style.display = "";
}

// --- boot --------------------------------------------------------------------

async function boot() {
  const defs = await api("/api/tiledefs");
  R.tiledefs = defs.tiles;
  requestAnimationFrame(render);
  const list = await loadReplayList();
  const wanted = new URLSearchParams(location.search).get("replay");
  if (wanted && list.some((m) => m.name === wanted)) {
    select.value = wanted;
    loadReplay(wanted);
  }
}

boot();
