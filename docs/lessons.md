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

- [decision] 2026-08-22 — **Pin `numpy>=1.26,<2` and `torch>=2.2,<2.3` (the `ml` extra).**
  Intel Mac (x86_64) has no torch wheels past 2.2.x, and torch 2.2.x was built
  against NumPy 1.x so it segfaults/aborts under NumPy 2. NumPy-2 support only
  arrived in torch 2.3, which has no Intel-Mac wheels — so on this machine
  torch+NumPy-2 is simply impossible. Chosen resolution: hold the whole project
  on NumPy 1.26 and torch 2.2.x, which also has aarch64-Linux wheels for the
  Pi 5, so one pin set works on both. MPS is unavailable on Intel (Apple-Silicon
  only) — dev is CPU-only; real training runs on the Pi. Revisit only if dev
  moves to Apple Silicon or a CUDA box, where newer torch+NumPy-2 become available.

- [rule] 2026-08-22 — **This project's venv breaks under iCloud sync.** The repo
  lives in iCloud-synced `~/Documents`; a daemon re-applies `UF_HIDDEN` to the
  editable-install `.pth`, which Python ≥3.12.4 silently skips, killing the
  `carcassonne` console script while pytest still passes. Fix: recreate the venv
  (`rm -rf .venv && uv sync --extra dev`). Diagnose this first if the CLI raises
  ModuleNotFoundError.
