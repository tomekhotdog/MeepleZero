// replay.js — the replay view: step through a saved game, board on the left,
// per-move AI search panel + whole-game win-probability chart on the right.
//
// Data source is the API exactly as produced by web/views.py:
//   GET /api/replays          -> {replays: [{name, agents, started_at, final_scores, winner, turns}]}
//   GET /api/replays/{name}   -> {header, moves, final_scores, winner, states}
// `states` holds one entry per position: states[i] is the board BEFORE move i,
// and states[moves.length] is the terminal position. moves[i] is applied to
// states[i] to reach states[i+1].

import {
  resizeToDisplay,
  drawTiles,
  drawFrontier,
  cellPx,
  screenToCellFloat,
  worldToScreen,
} from "./board.js";
import { TOKENS, featureAnchor } from "./tiles.js";
import {
  drawMoveRing,
  drawTileHighlight,
  drawFeatureOutline,
  collectMeepleHits,
  hitTestMeeple,
  createMeepleTip,
  showMeepleTip,
  hideMeepleTip,
} from "./overlays.js";

const AUTOPLAY_MS = 1000;

// --- state -------------------------------------------------------------------

const R = {
  tiledefs: null,
  data: null, // full /api/replays/{name} response
  index: 0, // current position: 0..moves.length
  moveRows: null, // <button> per move, built once per replay; index-aligned to moves
  winprob: [], // per-move {p: prob for player 0, known: bool}, length moves.length
  cam: { x: 0, y: 0, zoom: 1 },
  drag: null,
  playing: false,
  timer: null,
  meepleHits: [], // per-frame [{cx, cy, r, meeple}] for meeple hover hit-testing
  hoverMeeple: null, // the meeple dict under the cursor, or null
};

// --- DOM ---------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const board = $("board");
const bctx = board.getContext("2d");
const winCanvas = $("winprob");
const select = $("replay-select");
const stage = $("stage");
// The meeple-hover tooltip lives in the stage (same as the play view); the
// #meeple-tip CSS is global so it matches. replay.html is a separate page, so
// we build the element here.
const meepleTip = createMeepleTip(stage);

const playerColor = (p) => (p === 0 ? TOKENS.p0 : TOKENS.p1);

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
  buildMoveList();
  const M = R.data.moves.length;
  const slider = $("slider");
  slider.max = String(M);
  slider.value = "0";
  $("empty-hint").style.display = "none";
  $("scrubber").hidden = false;
  $("winprob-panel").hidden = false;
  $("winprob-axis").textContent = `move 0 → ${M} · higher = blue (P0) ahead`;
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
  clearMeepleHover(); // a moved board makes any hovered feature stale
  renderMoveInfo();
  renderSearchPanel();
  updateMoveHighlight();
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

// --- move list (left rail) ---------------------------------------------------

// Built once per replay; scrubbing only toggles `.current` + auto-scrolls, so
// the DOM isn't rebuilt every frame. Rows are chronological (move 1 at top),
// index-aligned to R.data.moves so R.index maps straight to R.moveRows[i].
function buildMoveList() {
  const list = $("move-list");
  const rows = [];
  const M = moves().length;
  for (let i = 0; i < M; i++) {
    const rec = R.data.moves[i];
    const before = R.data.states[i].scores;
    const after = R.data.states[i + 1].scores;
    const delta = after[rec.player] - before[rec.player];
    const scored = (R.data.states[i + 1].last_events ?? []).length > 0;

    const row = document.createElement("button");
    row.type = "button";
    row.className = "move-row";
    row.dataset.index = String(i);

    const num = document.createElement("span");
    num.className = "move-n";
    num.textContent = String(rec.n + 1); // 1-based, matching the scrubber's "move N / M"

    const tile = document.createElement("span");
    tile.className = "move-tile";
    tile.textContent = rec.tile;
    tile.title = rec.tile;

    const d = document.createElement("span");
    d.className = "move-d";
    if (delta > 0) d.textContent = `+${delta}`;

    row.append(num, swatch(rec.player), tile, d);
    if (scored) {
      const dot = document.createElement("span");
      dot.className = "event-dot";
      dot.setAttribute("aria-hidden", "true");
      dot.title = "something scored this move";
      row.append(dot);
    }
    row.addEventListener("click", () => {
      stopAutoplay();
      setIndex(i);
    });
    rows.push(row);
  }
  list.replaceChildren(...rows);
  R.moveRows = rows;
}

// Sync the highlight + scroll with R.index. The current move is the one about
// to be applied (R.index); at the terminal position (R.index === M) no row is
// current.
function updateMoveHighlight() {
  const rows = R.moveRows;
  if (!rows) return;
  for (let i = 0; i < rows.length; i++) {
    const on = i === R.index;
    rows[i].classList.toggle("current", on);
    if (on) rows[i].setAttribute("aria-current", "true");
    else rows[i].removeAttribute("aria-current");
  }
  const active = rows[R.index];
  if (active) active.scrollIntoView({ block: "nearest" });
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

// The board at R.index is states[R.index] — the position BEFORE move R.index,
// i.e. the position PRODUCED by move R.index-1. So both overlays key off that
// previous move: ① rings the tile it placed, ④ spotlights what it scored
// (states[R.index].last_events). Stepping Next thus rings the just-added tile
// and lights up what it scored, together. At index 0 there is no prior move.
function render() {
  const { w, h } = resizeToDisplay(board, bctx);
  bctx.clearRect(0, 0, w, h);
  if (R.data && R.tiledefs) {
    const view = R.data.states[R.index];
    drawTiles(bctx, board, view, R.tiledefs, R.cam);
    drawFrontier(bctx, board, view, R.cam);

    // Record meeple discs so pointermove can hit-test them.
    R.meepleHits = collectMeepleHits(view, R.tiledefs, board, R.cam);

    // (④) Score-event spotlight: highlight the tiles that scored to reach this
    // board, in the scoring player's colour.
    const events = view.last_events ?? [];
    for (const ev of events) {
      const keys = ev.tiles.map(([x, y]) => `${x},${y}`);
      drawTileHighlight(bctx, board, R.cam, keys, playerColor(ev.player));
    }

    // (①) Ring the tile placed by the move that produced this board.
    if (R.index >= 1) {
      const rec = R.data.moves[R.index - 1];
      const color = playerColor(rec.player);
      drawMoveRing(bctx, board, R.cam, rec.move, color);
      markPlacedMeeple(view, rec.move, color); // if the move also placed a meeple
    }

    // (F3) Hovered meeple's feature outline, over the highlights.
    if (R.hoverMeeple) {
      drawFeatureOutline(bctx, board, R.cam, R.hoverMeeple.feature_tiles);
    }

    // (④) Float a small +N over each scoring event's tiles, on top of everything.
    for (const ev of events) {
      drawScoreFloat(ev.tiles, ev.points, playerColor(ev.player));
    }

    drawWinProb();
  }
  requestAnimationFrame(render);
}

// (①) Emphasise a meeple the just-played move placed: a thin ring in the mover's
// colour around the meeple's anchor on its tile.
function markPlacedMeeple(view, mv, color) {
  const a = mv.action;
  if (!a || a.type !== "meeple") return;
  const tile = view.tiles.find((t) => t.x === mv.x && t.y === mv.y);
  if (!tile) return;
  const def = R.tiledefs[tile.type];
  const anchor = featureAnchor(def, a.feature, tile.rot);
  const c = cellPx(R.cam);
  const { sx, sy } = worldToScreen(board, R.cam, tile.x, tile.y);
  bctx.save();
  bctx.strokeStyle = color;
  bctx.lineWidth = 2;
  bctx.beginPath();
  bctx.arc(sx - c / 2 + anchor.x * c, sy - c / 2 + anchor.y * c, c * 0.16, 0, 2 * Math.PI);
  bctx.stroke();
  bctx.restore();
}

// (④) A small "+N" chip centred over an event's tiles, in the scoring colour.
function drawScoreFloat(tiles, points, color) {
  if (tiles.length === 0) return;
  let sx = 0;
  let sy = 0;
  for (const [x, y] of tiles) {
    const p = worldToScreen(board, R.cam, x, y);
    sx += p.sx;
    sy += p.sy;
  }
  sx /= tiles.length;
  sy /= tiles.length;
  const c = cellPx(R.cam);
  const text = `+${points}`;
  bctx.save();
  bctx.font = `700 ${Math.max(11, c * 0.24)}px ${getComputedStyle(document.body).getPropertyValue("--ui") || "sans-serif"}`;
  bctx.textAlign = "center";
  bctx.textBaseline = "middle";
  const tw = bctx.measureText(text).width;
  const padX = Math.max(4, c * 0.08);
  const chipH = Math.max(14, c * 0.3);
  const chipW = tw + padX * 2;
  bctx.fillStyle = "rgba(20, 24, 30, 0.82)";
  bctx.strokeStyle = color;
  bctx.lineWidth = 1.5;
  const left = sx - chipW / 2;
  const top = sy - chipH / 2;
  bctx.beginPath();
  bctx.roundRect(left, top, chipW, chipH, Math.min(6, chipH / 2));
  bctx.fill();
  bctx.stroke();
  bctx.fillStyle = color;
  bctx.fillText(text, sx, sy + 0.5);
  bctx.restore();
}

// --- board pan / zoom (view-only) --------------------------------------------

board.addEventListener("pointerdown", (e) => {
  R.drag = { px: e.clientX, py: e.clientY };
  board.setPointerCapture(e.pointerId);
  board.classList.add("dragging");
});

board.addEventListener("pointermove", (e) => {
  const rect = board.getBoundingClientRect();
  if (R.drag) {
    clearMeepleHover();
    const c = cellPx(R.cam);
    R.cam.x -= (e.clientX - R.drag.px) / c;
    R.cam.y += (e.clientY - R.drag.py) / c; // screen y down = world south
    R.drag.px = e.clientX;
    R.drag.py = e.clientY;
    return;
  }
  updateMeepleHover(e.clientX - rect.left, e.clientY - rect.top);
});

board.addEventListener("pointerleave", clearMeepleHover);

// Find the meeple under the cursor (if any) and drive the hover tooltip.
function updateMeepleHover(px, py) {
  const found = hitTestMeeple(R.meepleHits, px, py);
  if (!found) {
    clearMeepleHover();
    return;
  }
  R.hoverMeeple = found;
  showMeepleTip(meepleTip, stage, found, px, py);
}

function clearMeepleHover() {
  R.hoverMeeple = null;
  hideMeepleTip(meepleTip);
}

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
    if (e.target.closest(".move-row")) return; // let a focused move row self-activate
    e.preventDefault();
    toggleAutoplay();
  } else {
    return;
  }
});


// --- helpers -----------------------------------------------------------------

function showError(msg) {
  $("replay-error").textContent = msg || "";
}

function hidePanels() {
  for (const id of ["scrubber", "move-info", "search-panel", "winprob-panel"]) {
    $(id).hidden = true;
  }
  $("move-list").replaceChildren();
  R.moveRows = null;
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
