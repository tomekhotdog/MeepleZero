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
import { TOKENS, featureAnchor, drawTile } from "./tiles.js";
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
  scores: [], // (③) per-position [p0, p1] running score, length states.length (M+1)
  scoreMax: 1, // (③) y-axis top for the score chart (max score reached, min 1)
  cam: { x: 0, y: 0, zoom: 1 },
  drag: null,
  playing: false,
  timer: null,
  meepleHits: [], // per-frame [{cx, cy, r, meeple}] for meeple hover hit-testing
  hoverMeeple: null, // the meeple dict under the cursor, or null
  showAlts: false, // (②) "Show alternatives" toggle; persists across scrubbing
  ghostHits: [], // per-frame [{left, top, size, cand, priorPct, state}] for ghost hover
  hoverGhost: null, // the ghost hit under the cursor, or null
};

// --- DOM ---------------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const board = $("board");
const bctx = board.getContext("2d");
const winCanvas = $("winprob");
const scoreCanvas = $("score-chart");
const select = $("replay-select");
const stage = $("stage");
// The meeple-hover tooltip lives in the stage (same as the play view); the
// #meeple-tip CSS is global so it matches. replay.html is a separate page, so
// we build the element here.
const meepleTip = createMeepleTip(stage);
// (②) A second tooltip for hovered alternative-move ghosts, styled with the
// --ai accent (#ghost-tip in style.css). Lives in the stage like the meeple tip.
const ghostTip = document.createElement("div");
ghostTip.id = "ghost-tip";
ghostTip.setAttribute("role", "tooltip");
stage.append(ghostTip);

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
  buildScore();
  buildMoveList();
  const M = R.data.moves.length;
  const slider = $("slider");
  slider.max = String(M);
  slider.value = "0";
  $("empty-hint").style.display = "none";
  $("scrubber").hidden = false;
  $("winprob-panel").hidden = false;
  $("winprob-axis").textContent = `move 0 → ${M} · higher = blue (P0) ahead`;
  $("score-panel").hidden = false;
  $("score-axis").textContent = `move 0 → ${M} · points · blue P0 · red P1`;
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
  // The current move is the just-played one: moves[R.index - 1]. At R.index 0
  // no move has been played yet (start position).
  $("position").textContent = R.index === 0 ? `start / ${M}` : `move ${R.index} / ${M}`;
  clearMeepleHover(); // a moved board makes any hovered feature stale
  clearGhostHover(); // and any hovered ghost from the previous position
  renderMoveInfo();
  renderSearchPanel();
  renderAltPanel();
  renderScoreReadout();
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
      setIndex(i + 1); // make move i the just-played move (board shows its result)
    });
    rows.push(row);
  }
  list.replaceChildren(...rows);
  R.moveRows = rows;
}

// Sync the highlight + scroll with R.index. The current move is the just-played
// one, moves[R.index - 1], so row R.index - 1 is highlighted. At the start
// position (R.index === 0) no row is current; at the terminal position
// (R.index === M) the last row (M - 1) is current.
function updateMoveHighlight() {
  const rows = R.moveRows;
  if (!rows) return;
  const cur = R.index - 1;
  for (let i = 0; i < rows.length; i++) {
    const on = i === cur;
    rows[i].classList.toggle("current", on);
    if (on) rows[i].setAttribute("aria-current", "true");
    else rows[i].removeAttribute("aria-current");
  }
  const active = rows[cur];
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
  const info = $("move-info");
  // The current move is the just-played one: moves[R.index - 1]. At the start
  // position (R.index === 0) there is no move to describe.
  const c = R.index - 1;
  if (c < 0) {
    info.hidden = true;
    return;
  }
  info.hidden = false;
  const rec = moves()[c];
  const who = $("move-who");
  who.replaceChildren(
    swatch(rec.player),
    document.createTextNode(` ${rec.player === 0 ? "blue" : "red"} · ${R.data.header.agents[rec.player]}`)
  );
  $("move-detail").textContent = `${rec.tile} → ${moveLabel(rec.move)}`;

  // Score delta: diff the running scores across this move's two states.
  const before = R.data.states[c].scores;
  const after = R.data.states[c + 1].scores;
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

// P0-perspective win% for a move's annot, or null if it has no search data.
// annot.value is the MOVING player's value in [-1, 1] (tanh); flip it to P0's
// perspective and map to [0, 100] — the exact transform buildWinProb uses for
// the win-probability chart, so this number always agrees with that chart.
function winPctOf(rec) {
  if (!rec || !rec.annot) return null;
  const p0Value = rec.player === 0 ? rec.annot.value : -rec.annot.value;
  return Math.round(50 * (1 + p0Value));
}

function renderSearchPanel() {
  const panel = $("search-panel");
  const bars = $("search-bars");
  const note = $("search-note");
  const valueEl = $("search-value");
  const metaEl = $("search-meta");
  bars.replaceChildren();
  valueEl.textContent = "";
  metaEl.textContent = "";
  note.textContent = "";

  // The current move is the just-played one: moves[R.index - 1]. At the start
  // position (R.index === 0) there is no move to describe.
  const c = R.index - 1;
  if (c < 0) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const rec = moves()[c];
  const annot = rec.annot;
  if (!annot || annot.top.length === 0) {
    note.textContent = "No search data for this move.";
    return;
  }

  const totalVisits = annot.top.reduce((s, c) => s + c.visits, 0);
  const totalPrior = annot.top.reduce((s, c) => s + c.prior, 0) || 1;
  const heuristic = totalVisits === 0;

  // Value as a P0-perspective win%, with a delta vs the previous ANNOTATED move.
  // Human moves have no annot, so we search back past them for the last search.
  const winPct = winPctOf(rec);
  let prevPct = null;
  for (let i = c - 1; i >= 0; i--) {
    const p = winPctOf(moves()[i]);
    if (p !== null) {
      prevPct = p;
      break;
    }
  }
  valueEl.replaceChildren();
  valueEl.append(document.createTextNode("P0 win: "));
  const pctEl = document.createElement("span");
  pctEl.className = "ai"; // --ai accents the win% only, not the delta
  pctEl.textContent = `${winPct}%`;
  valueEl.append(pctEl);
  const deltaEl = document.createElement("span");
  if (prevPct === null) {
    deltaEl.className = "win-delta flat";
    deltaEl.textContent = "—"; // first annotated move: no baseline to diff against
  } else {
    const diff = winPct - prevPct;
    deltaEl.className = `win-delta ${diff > 0 ? "up" : diff < 0 ? "down" : "flat"}`;
    deltaEl.textContent =
      diff > 0 ? `▲ +${diff}%` : diff < 0 ? `▼ −${Math.abs(diff)}%` : "—";
  }
  valueEl.append(deltaEl);

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

  // Agree/overrule flag: did the tree search stick with the network's top prior,
  // or did visits crown a different move? Only meaningful with real MCTS.
  if (heuristic) {
    note.textContent = "Heuristic — no search (visit counts 0; priors shown).";
  } else {
    const priorLeader = annot.top.reduce((a, b) => (b.prior > a.prior ? b : a));
    const visitLeader = annot.top.reduce((a, b) => (b.visits > a.visits ? b : a));
    note.textContent = sameMove(priorLeader.move, visitLeader.move)
      ? "Search agreed with its top prior."
      : `Search overruled its top prior (chose ${moveLabel(visitLeader.move)} over ${moveLabel(priorLeader.move)}).`;
  }
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

// --- (②) alternative-move ghosts ---------------------------------------------

// Classify the current move's search annot for the ghosts overlay:
//   "none"      — no annot (human move): nothing to show, toggle disabled.
//   "heuristic" — priors present but zero visits (greedy): size ghosts by prior.
//   "mcts"      — real tree search (visits > 0): size ghosts by visits.
// Also returns the totals used to normalise, and a one-line explanatory note.
function altInfo(rec) {
  const annot = rec ? rec.annot : null;
  if (!annot || annot.top.length === 0) {
    return { state: "none", note: "No search data for this move.", totalVisits: 0, totalPrior: 1 };
  }
  const totalVisits = annot.top.reduce((s, c) => s + c.visits, 0);
  const totalPrior = annot.top.reduce((s, c) => s + c.prior, 0) || 1;
  if (totalVisits === 0) {
    return { state: "heuristic", note: "Heuristic priors (no search).", totalVisits, totalPrior };
  }
  return { state: "mcts", note: "Ghost opacity ∝ search visits.", totalVisits, totalPrior };
}

// Toggle panel: shown for any current move, disabled when there is no search
// data. The checkbox reflects the persistent R.showAlts so scrubbing keeps it.
function renderAltPanel() {
  const panel = $("alt-panel");
  const toggle = $("alt-toggle");
  const note = $("alt-note");
  // The current move is the just-played one: moves[R.index - 1].
  const c = R.index - 1;
  if (c < 0) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  toggle.checked = R.showAlts;
  const info = altInfo(moves()[c]);
  toggle.disabled = info.state === "none";
  // Explain the state: always for "none"/"heuristic"; for real search only once
  // the ghosts are actually on (otherwise the note describes nothing visible).
  note.textContent = info.state === "mcts" && !R.showAlts ? "" : info.note;
}

// Draw a translucent ghost of the drawn tile at each candidate placement the AI
// weighed for this move, opacity ∝ its share of visits (or prior, for greedy).
// The move actually played is left to drawMoveRing (solid + ringed), so it is
// skipped here. Records screen boxes in R.ghostHits for hover hit-testing.
function drawGhosts() {
  R.ghostHits = [];
  if (!R.showAlts) return;
  const c = R.index - 1;
  if (c < 0) return;
  const rec = moves()[c];
  const info = altInfo(rec);
  if (info.state === "none") return;
  const def = R.tiledefs[rec.tile];
  if (!def) return;
  const cell = cellPx(R.cam);
  for (const cand of rec.annot.top) {
    if (sameMove(cand.move, rec.move)) continue; // the played move stays solid + ringed
    const frac =
      info.state === "mcts" ? cand.visits / info.totalVisits : cand.prior / info.totalPrior;
    const alpha = Math.max(0.15, Math.min(0.85, frac));
    const { sx, sy } = worldToScreen(board, R.cam, cand.move.x, cand.move.y);
    const left = sx - cell / 2;
    const top = sy - cell / 2;
    const meeples = [];
    const a = cand.move.action;
    if (a && a.type === "meeple") {
      meeples.push({ player: rec.player, kind: a.kind, feature: a.feature });
    }
    bctx.save();
    bctx.translate(left, top);
    drawTile(bctx, def, cand.move.rot, cell, { alpha, meeples });
    bctx.restore();
    // A thin dashed --ai frame marks it as a hypothetical AI-weighed placement.
    bctx.save();
    bctx.globalAlpha = Math.max(0.4, alpha);
    bctx.strokeStyle = TOKENS.ai;
    bctx.setLineDash([4, 3]);
    bctx.lineWidth = 1.5;
    bctx.strokeRect(left + 1.5, top + 1.5, cell - 3, cell - 3);
    bctx.restore();
    R.ghostHits.push({
      left,
      top,
      size: cell,
      cand,
      priorPct: cand.prior / info.totalPrior,
      state: info.state,
    });
  }
}

// Ghost under (px, py), canvas-relative, or null. Later-drawn ghosts sit on top,
// so hit-test back to front.
function hitTestGhost(px, py) {
  for (let i = R.ghostHits.length - 1; i >= 0; i--) {
    const g = R.ghostHits[i];
    if (px >= g.left && px <= g.left + g.size && py >= g.top && py <= g.top + g.size) return g;
  }
  return null;
}

function updateGhostHover(px, py) {
  const hit = hitTestGhost(px, py);
  if (!hit) {
    clearGhostHover();
    return false;
  }
  clearMeepleHover(); // never stack the two tooltips
  R.hoverGhost = hit;
  showGhostTip(hit, px, py);
  return true;
}

function clearGhostHover() {
  R.hoverGhost = null;
  ghostTip.classList.remove("show");
}

// A small tag near the ghost: prior % (normalised to match the search panel),
// visit count, and the meeple action it would take.
function showGhostTip(g, px, py) {
  const mv = g.cand.move;
  const a = mv.action;
  const act = a === null ? "no meeple" : a.type === "meeple" ? `${a.kind} f${a.feature}` : "retrieve abbot";
  const head = document.createElement("div");
  head.className = "tip-kind";
  head.textContent = g.state === "mcts" ? "alternative" : "alternative · prior only";
  const prior = document.createElement("div");
  prior.textContent = `prior ${Math.round(g.priorPct * 100)}%`;
  const visits = document.createElement("div");
  visits.textContent = `visits ${g.cand.visits}`;
  const action = document.createElement("div");
  action.textContent = act;
  ghostTip.replaceChildren(head, prior, visits, action);
  ghostTip.classList.add("show");
  // Offset from the cursor, clamped inside the stage so nothing gets clipped.
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  const tw = ghostTip.offsetWidth;
  const th = ghostTip.offsetHeight;
  let left = px + 16;
  let top = py + 16;
  if (left + tw > sw) left = px - tw - 16;
  if (top + th > sh) top = py - th - 16;
  ghostTip.style.left = `${Math.max(4, left)}px`;
  ghostTip.style.top = `${Math.max(4, top)}px`;
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

  // Current-move marker: the just-played move, moves[R.index - 1]. At the start
  // position (R.index === 0) there is no current move, so draw no marker.
  if (R.index >= 1 && n > 0) {
    const markIdx = Math.min(R.index - 1, n - 1);
    const mx = xAt(markIdx);
    ctx.strokeStyle = "rgba(232, 226, 212, 0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mx, padT);
    ctx.lineTo(mx, padT + plotH);
    ctx.stroke();
  }
}

function winProbClick(e) {
  const n = R.winprob.length;
  if (n === 0) return;
  const rect = winCanvas.getBoundingClientRect();
  const padL = 22;
  const padR = 6;
  const plotW = rect.width - padL - padR;
  const frac = (e.clientX - rect.left - padL) / plotW;
  // Data point i is move i; make it the just-played move so the marker lands
  // where the user clicked (marker is drawn at R.index - 1).
  const i = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
  setIndex(i + 1);
}

// --- (③) running score + score-over-time chart -------------------------------

// The actual game score after each position, distinct from the win-prob chart
// (which is the AI's belief). states[i].scores is [p0, p1] AFTER move i-1, so
// this series is index-aligned to R.index directly — no -1 offset. The terminal
// state's running .scores is pre-end-game; its .final_scores adds the end-game
// scoring of incomplete features, so use that for the last point (and readout)
// to reflect the true game result.
function buildScore() {
  const states = R.data.states;
  R.scores = states.map((s) => (s.terminal && s.final_scores ? s.final_scores : s.scores));
  let max = 1;
  for (const [a, b] of R.scores) max = Math.max(max, a, b);
  R.scoreMax = max;
}

// Prominent P0 : P1 readout at the current step, states[R.index].scores.
function renderScoreReadout() {
  const s = R.data ? R.scores[R.index] : null;
  $("score-p0").textContent = s ? String(s[0]) : "0";
  $("score-p1").textContent = s ? String(s[1]) : "0";
}

// Mirror of drawWinProb: two lines (P0 blue, P1 red) of running points vs
// position i (0..M), a mono axis, and a vertical marker at the current step.
function drawScore() {
  if (!R.data) return;
  const ctx = scoreCanvas.getContext("2d");
  const { w, h } = resizeToDisplay(scoreCanvas, ctx);
  ctx.clearRect(0, 0, w, h);
  const padL = 22;
  const padR = 6;
  const padT = 8;
  const padB = 4;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  const n = R.scores.length;
  const top = R.scoreMax;
  const xAt = (i) => padL + (n <= 1 ? plotW / 2 : (i / (n - 1)) * plotW);
  const yAt = (v) => padT + (1 - v / top) * plotH;

  // y labels: 0 at the bottom, the max score at the top.
  ctx.fillStyle = "rgba(155, 163, 173, 0.8)";
  ctx.font = "9px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText(String(top), padL - 4, yAt(top));
  ctx.fillText("0", padL - 4, yAt(0));

  // Baseline (0 points).
  ctx.strokeStyle = "rgba(155, 163, 173, 0.25)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padL, yAt(0));
  ctx.lineTo(padL + plotW, yAt(0));
  ctx.stroke();

  // Two lines: P0 (blue), P1 (red).
  for (const p of [0, 1]) {
    ctx.strokeStyle = playerColor(p);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xAt(i);
      const y = yAt(R.scores[i][p]);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Current-step marker: index-aligned, so mark R.index directly.
  if (n > 0) {
    const mx = xAt(Math.min(R.index, n - 1));
    ctx.strokeStyle = "rgba(232, 226, 212, 0.7)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mx, padT);
    ctx.lineTo(mx, padT + plotH);
    ctx.stroke();
  }
}

// Click-to-jump: data point i is position i, so jump straight there.
function scoreClick(e) {
  const n = R.scores.length;
  if (n === 0) return;
  const rect = scoreCanvas.getBoundingClientRect();
  const padL = 22;
  const padR = 6;
  const plotW = rect.width - padL - padR;
  const frac = (e.clientX - rect.left - padL) / plotW;
  const i = Math.max(0, Math.min(n - 1, Math.round(frac * (n - 1))));
  stopAutoplay();
  setIndex(i);
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

    // (②) Alternative-move ghosts: the placements MCTS weighed for this move,
    // over the (empty) candidate cells. Under the score highlights / move ring.
    drawGhosts();

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
    drawScore();
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
  // If the move completed the feature the same turn, the meeple is already back
  // in supply on this board — don't ring an empty spot.
  if (!tile.meeples.some((m) => m.feature === a.feature)) return;
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
    clearGhostHover();
    const c = cellPx(R.cam);
    R.cam.x -= (e.clientX - R.drag.px) / c;
    R.cam.y += (e.clientY - R.drag.py) / c; // screen y down = world south
    R.drag.px = e.clientX;
    R.drag.py = e.clientY;
    return;
  }
  const px = e.clientX - rect.left;
  const py = e.clientY - rect.top;
  // Ghost tags take precedence over meeple tips (ghosts sit on empty cells, so
  // the two rarely collide, but never show both at once).
  if (updateGhostHover(px, py)) return;
  updateMeepleHover(px, py);
});

board.addEventListener("pointerleave", () => {
  clearMeepleHover();
  clearGhostHover();
});

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
scoreCanvas.addEventListener("click", scoreClick);
$("alt-toggle").addEventListener("change", (e) => {
  R.showAlts = e.target.checked;
  clearGhostHover();
  renderAltPanel(); // refresh the note now the toggle changed
});

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
  for (const id of ["scrubber", "move-info", "search-panel", "alt-panel", "winprob-panel", "score-panel"]) {
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
