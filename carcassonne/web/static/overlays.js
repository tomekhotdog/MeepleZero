// overlays.js — board overlays shared by the play and replay views.
//
// These are the pieces that are pixel-for-pixel identical between the two views:
// the last-/current-move ring, the score-event tile highlight, the meeple-hover
// feature outline, meeple hit-testing, and the hover tooltip element. Each takes
// its (ctx, canvas, cam) explicitly so it is bound to no particular page state.

import { TOKENS, featureAnchor } from "./tiles.js";
import { worldToScreen, cellPx } from "./board.js";

// A persistent move ring: a solid ring in the mover's colour, plus an optional
// labelled corner chip. Play uses the chip for "You" / "AI"; replay leaves it
// off (the colour already names the player).
export function drawMoveRing(ctx, canvas, cam, pos, color, label) {
  if (!pos) return;
  const c = cellPx(cam);
  const { sx, sy } = worldToScreen(canvas, cam, pos.x, pos.y);
  const left = sx - c / 2;
  const top = sy - c / 2;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(left + 1.5, top + 1.5, c - 3, c - 3);
  if (label) {
    ctx.font = `600 ${Math.max(8, c * 0.16)}px ${getComputedStyle(document.body).getPropertyValue("--ui") || "sans-serif"}`;
    const pad = Math.max(2, c * 0.04);
    const tw = ctx.measureText(label).width;
    const chipH = Math.max(11, c * 0.22);
    const chipW = tw + pad * 2;
    ctx.fillStyle = color;
    ctx.fillRect(left + 1.5, top + 1.5, chipW, chipH);
    ctx.fillStyle = TOKENS.ink;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(label, left + 1.5 + pad, top + 1.5 + chipH / 2 + 0.5);
  }
  ctx.restore();
}

// Score highlight: translucent fill + solid outline over a set of tiles, in the
// scoring player's colour (falls back to neutral --bone). `keys` is any iterable
// of "x,y" strings.
export function drawTileHighlight(ctx, canvas, cam, keys, color) {
  const c = cellPx(cam);
  const col = color ?? TOKENS.bone;
  for (const key of keys) {
    const [x, y] = key.split(",").map(Number);
    const { sx, sy } = worldToScreen(canvas, cam, x, y);
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = col;
    ctx.fillRect(sx - c / 2 + 2, sy - c / 2 + 2, c - 4, c - 4);
    ctx.restore();
    ctx.strokeStyle = col;
    ctx.lineWidth = 2.5;
    ctx.strokeRect(sx - c / 2 + 2, sy - c / 2 + 2, c - 4, c - 4);
  }
}

// A dashed neutral --bone outline around a hovered meeple's feature tiles.
export function drawFeatureOutline(ctx, canvas, cam, tiles) {
  const c = cellPx(cam);
  ctx.strokeStyle = TOKENS.bone;
  ctx.lineWidth = 2;
  ctx.setLineDash([2, 3]);
  for (const [x, y] of tiles) {
    const { sx, sy } = worldToScreen(canvas, cam, x, y);
    ctx.strokeRect(sx - c / 2 + 2.5, sy - c / 2 + 2.5, c - 5, c - 5);
  }
  ctx.setLineDash([]);
}

// Record every drawn meeple's screen disc so a pointer can hit-test them. The
// anchors are defined unrotated; featureAnchor with the tile's rotation
// reproduces the on-screen position drawTile drew the meeple at.
export function collectMeepleHits(view, tiledefs, canvas, cam) {
  const c = cellPx(cam);
  const hits = [];
  for (const t of view.tiles) {
    const def = tiledefs[t.type];
    const { sx, sy } = worldToScreen(canvas, cam, t.x, t.y);
    for (const m of t.meeples ?? []) {
      const a = featureAnchor(def, m.feature, t.rot);
      hits.push({ cx: sx - c / 2 + a.x * c, cy: sy - c / 2 + a.y * c, r: c * 0.1, meeple: m });
    }
  }
  return hits;
}

// Return the meeple dict under (px, py) — canvas-relative — or null.
export function hitTestMeeple(hits, px, py) {
  for (const hit of hits) {
    if (Math.hypot(px - hit.cx, py - hit.cy) <= hit.r + 3) return hit.meeple;
  }
  return null;
}

// The hover tooltip lives in the stage so it can be positioned over the board.
// The #meeple-tip CSS is global, so both views share the same look.
export function createMeepleTip(stage) {
  const tip = document.createElement("div");
  tip.id = "meeple-tip";
  tip.setAttribute("role", "tooltip");
  stage.append(tip);
  return tip;
}

export function showMeepleTip(tip, stage, m, px, py) {
  const kind = document.createElement("div");
  kind.className = "tip-kind";
  kind.textContent = m.feature_kind;
  const now = document.createElement("div");
  now.textContent = `score now: ${m.score_now}`;
  const rest = document.createElement("div");
  rest.textContent = m.complete ? "complete" : `if completed: ${m.score_potential}`;
  tip.replaceChildren(kind, now, rest);
  tip.classList.add("show");
  // Offset from the cursor, clamped inside the stage so nothing gets clipped.
  const sw = stage.clientWidth;
  const sh = stage.clientHeight;
  const tw = tip.offsetWidth;
  const th = tip.offsetHeight;
  let left = px + 16;
  let top = py + 16;
  if (left + tw > sw) left = px - tw - 16;
  if (top + th > sh) top = py - th - 16;
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.max(4, top)}px`;
}

export function hideMeepleTip(tip) {
  tip.classList.remove("show");
}
