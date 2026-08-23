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

- [rule] 2026-08-22 — **Never run a git-mutating review/agent concurrently with
  git-mutating build work on the same repo.** (Corrects an earlier, wrong "iCloud
  corrupted .git" entry.) During Task 22 the working tree reverted twice and a
  fresh commit was orphaned (`main` snapped back one commit). The cause was NOT
  iCloud: a "read-only" code-review agent's fork violated its contract by
  editing + committing the same encoder fix, and the review agent then ran
  `git reset --hard` on the shared working directory while this build session was
  concurrently committing the identical fix. Two agents mutating one `.git` =
  phantom reverts and orphaned commits. Lessons: (a) dispatch review agents
  read-only for real (no git writes) and don't run them while you're committing;
  (b) if a working file unexpectedly reverts, check for other processes on the
  repo (`git reflog`, `ps`) before blaming the filesystem; (c) `git checkout -- .`
  and `git reset --hard` are blunt on a shared tree — they discarded good
  uncommitted work here; prefer per-file operations. Outcome: nothing permanently
  lost — HEAD ended correct, 174 tests green. The venv/`.pth` iCloud issue above
  is real and independent.

- [rule] 2026-08-23 — **Document-style pages under the app shell need their own
  scroll container.** The web shell sets `html, body { height: 100%; overflow:
  hidden }` on purpose — Play/Replay/Training are fixed-viewport layouts whose
  inner panels scroll (board fills the viewport; replay columns and the training
  charts column each have `overflow-y: auto`). A new *long-document* page (the
  Tiles reference tab) added under `#app` was therefore clipped below the fold
  with no way to scroll. Fix: give the page element its own `overflow-y: auto`
  (the `.atlas` block; it IS `#app`, already `flex:1` with `min-height:0`, so it
  becomes the scroll container). Lesson: when adding a scrolling document page to
  a `body{overflow:hidden}` app, the page container must opt into its own scroll —
  the body won't do it for you.
