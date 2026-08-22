// board.js — shared board renderer + camera math for the play and replay views.
//
// Draws a StateView's placed tiles and frontier dots onto a canvas under a
// pan/zoom camera, with DPI handling. The view-specific overlays (legal cells,
// hints, the search panel, …) stay in each page; this is only the common base
// both draw first, so the tiles look identical in play and replay.

import { drawTile } from "./tiles.js";

export const CELL = 64; // world cell size at zoom 1

// A camera is {x, y, zoom}: (x, y) is the world cell under the canvas centre.
export function cellPx(cam) {
  return CELL * cam.zoom;
}

// World: +x east, +y north (Pos.neighbor). Screen: y down, so north is up.
export function worldToScreen(canvas, cam, x, y) {
  const c = cellPx(cam);
  return {
    sx: canvas.clientWidth / 2 + (x - cam.x) * c,
    sy: canvas.clientHeight / 2 - (y - cam.y) * c,
  };
}

export function screenToCell(canvas, cam, px, py) {
  const c = cellPx(cam);
  return {
    x: Math.round((px - canvas.clientWidth / 2) / c + cam.x),
    y: Math.round(-(py - canvas.clientHeight / 2) / c + cam.y),
  };
}

export function screenToCellFloat(canvas, cam, px, py) {
  const c = cellPx(cam);
  return {
    x: (px - canvas.clientWidth / 2) / c + cam.x,
    y: -(py - canvas.clientHeight / 2) / c + cam.y,
  };
}

// Match the backing store to the display size at the current DPR and scale the
// context so all drawing is in CSS pixels. Returns those CSS-pixel dimensions.
export function resizeToDisplay(canvas, ctx) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h, dpr };
}

// Draw every placed tile (with its meeples). Returns the cell size in px so
// callers can align their overlays to the same grid.
export function drawTiles(ctx, canvas, view, tiledefs, cam) {
  const c = cellPx(cam);
  for (const t of view.tiles) {
    const { sx, sy } = worldToScreen(canvas, cam, t.x, t.y);
    ctx.save();
    ctx.translate(sx - c / 2, sy - c / 2);
    drawTile(ctx, tiledefs[t.type], t.rot, c, { meeples: t.meeples });
    ctx.restore();
  }
  return c;
}

// Subtle dots on the empty neighbours of the board — the open frontier.
export function drawFrontier(ctx, canvas, view, cam) {
  const c = cellPx(cam);
  const placed = new Set(view.tiles.map((t) => `${t.x},${t.y}`));
  const frontier = new Set();
  for (const t of view.tiles) {
    for (const [dx, dy] of [[0, 1], [1, 0], [0, -1], [-1, 0]]) {
      const key = `${t.x + dx},${t.y + dy}`;
      if (!placed.has(key)) frontier.add(key);
    }
  }
  ctx.fillStyle = "rgba(155, 163, 173, 0.3)";
  for (const key of frontier) {
    const [x, y] = key.split(",").map(Number);
    const { sx, sy } = worldToScreen(canvas, cam, x, y);
    ctx.beginPath();
    ctx.arc(sx, sy, Math.max(1.5, c * 0.03), 0, 2 * Math.PI);
    ctx.fill();
  }
}
