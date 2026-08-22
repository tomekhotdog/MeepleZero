# Lessons & Decisions

- [decision] 2026-08-22 — **Agent protocol lives in `carcassonne/game/agent.py`, not
  `agents/`.** The protocol's vocabulary (GameState, Move, Annot) is defined at or
  below the game layer, so putting it there keeps imports strictly one-way
  (`core → game → agents/nn → training/web`); `agents/` holds the registry and
  concrete agents and re-exports the protocol from `agents.base` for API
  stability. Chosen over moving `match.py` up into `agents/` because the match
  runner is cohesive with `game/replay.py`. Trade-off: the protocol sits apart
  from its implementations. Do not "tidy" it back into `agents/` — that
  reintroduces an upward `game → agents` import that will trip TID251 layering
  enforcement.

- [rule] 2026-08-22 — **This project's venv breaks under iCloud sync.** The repo
  lives in iCloud-synced `~/Documents`; a daemon re-applies `UF_HIDDEN` to the
  editable-install `.pth`, which Python ≥3.12.4 silently skips, killing the
  `carcassonne` console script while pytest still passes. Fix: recreate the venv
  (`rm -rf .venv && uv sync --extra dev`). Diagnose this first if the CLI raises
  ModuleNotFoundError.
