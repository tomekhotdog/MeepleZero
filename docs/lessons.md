# Lessons & Decisions

- [decision] 2026-08-22 — **Agent protocol lives in `carcassonne/game/agent.py`, not
  `agents/`.** The protocol's vocabulary (GameState, Move, Annot) is defined at or
  below the game layer, so putting it there keeps imports strictly one-way
  (`core → game → agents/nn → training/web`); `agents/` holds the registry and
  concrete agents and re-exports the protocol from `agents.base` for API
  stability. Chosen over moving `match.py` up into `agents/` because the match
  runner is cohesive with `game/replay.py`. Trade-off: the protocol sits apart
  from its implementations. Do not "tidy" it back into `agents/` — that
  reintroduces an upward `game → agents` import that fails the layering test
  (`tests/test_layering.py`, an AST walk over every import; ruff TID251 was
  selected but never configured, so it enforced nothing — corrected 2026-10-02).

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

- [rule] 2026-10-03 — **When an orchestration plan executes other plans by reference,
  tick the source plans' checkboxes as each task completes.** The mega plan tracked
  M1–M19 while MP/DC/M18b plans kept their own Review boxes and manual browser
  checklists; `verify` found 19 unchecked items at the very end although the work
  (incl. headless-Chrome browser checks) had been done. Ticking them retroactively
  needed digging evidence out of task reports. Rule: when marking an M-task done in
  the orchestration plan, also tick (with commit/evidence) the matching items in the
  source plan in the same commit.

- [decision] 2026-10-02 — **The iteration-3 "≥5× single-process self-play" bar was
  missed (3.51×) and accepted as a recorded FAIL.** Profile-driven engine work
  (M18b: table-driven placements, meeple-claim via neighbour roots, carried
  placements) doubled throughput with byte-identical games, guarded by a golden
  engine digest (`tests/core/test_engine_golden.py` — never re-record it to make a
  change pass). After that the net forward pass is 56% of self-play time; the
  remaining lever is MCTS **leaf batching with virtual loss**, which changes the
  search algorithm (games no longer identical; needs a strength check), so it
  must go through `design`, not be patched in for speed. Pi throughput comes from
  workers (≈8.8× aggregate with 4). Rejected: redefining the bar as aggregate
  throughput (moves the goalposts).

- [decision] 2026-09-30 — **MCTS subtree reuse is ON for training/arena, OFF for
  interactive checkpoint agents** (`agents/checkpoints.py` builds
  `MctsConfig(..., reuse_tree=False)`). Reuse exists for self-play throughput.
  But the web's hint path calls `opponent.choose()` (the Agent protocol has no
  side-effect-free "analyse" method), so with reuse a hint's search left a
  stashed subtree that the AI's next reply inherited — asking for a hint changed
  the game (4/80 seeds diverged). Rejected: a reuse-free MctsAgent method for
  hints (pierces the Agent-protocol boundary decided above) and save/restore of
  the private `_stash` from the web layer. Bonus: the RL microscope stays exact —
  visits == sims. Do not "re-enable reuse for stronger web play" without first
  giving hints a genuinely side-effect-free path; the seeded
  `test_repeated_hints_do_not_perturb_checkpoint_game` will fail if you do.

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
