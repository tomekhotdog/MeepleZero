# ProjectCarcasonne — Iteration 1 Verification

Date: 2026-08-22. Records what is **proven now** vs **deferred to a runtime you
execute**, honestly, per the plan's Task 22.

## Automated gates (proven)

- **Test suite:** `uv run pytest` → **174 passed** (no skips beyond two
  environment-conditional `node`-on-PATH skips, no xfails).
- **Lint:** `uv run ruff check .` → clean.
- **Types:** `uv run mypy carcassonne tests` → clean (strict, 74 files).
- Environment: Intel Mac, CPU only, Python 3.12.6, numpy 1.26.4, torch 2.2.2
  (see `docs/lessons.md` for the pin rationale).

## Success criteria (from the design)

### 1. Engine correctness proven — ✅ MET
- Rules engine behind the five-function API (`new_game`, `legal_moves`, `apply`,
  `is_terminal`, `final_scores`), all pure/deterministic.
- Hypothesis property tests over random games assert: meeple conservation
  (7 meeples + 1 abbot per player, always), scores monotonically non-decreasing,
  `apply` rejects any move not in `legal_moves`, and replay determinism
  (`(seed, moves)` reproduces identical states).
- Deck data verified against the official base-game distribution (72 tiles, 10
  shields, per-letter counts). Full 72-tile games terminate cleanly.

### 2. Trained agent beats GreedyAgent >70% over 100 games — ⏸ DEFERRED TO TRAINING
- The *mechanism* is built and tested: `evaluate --p0 ckpt:latest --p1 greedy
  --games 100` runs; arena, gating (>55% promotion), and the yardstick-vs-greedy
  are wired and unit-tested.
- The *outcome* is a training result, not a code property. An **untrained**
  network's MCTS is weak (measured 3/10 vs RandomAgent at 64 sims — search only
  amplifies the value head, Task 16 finding). Reaching >70% requires a real
  training run (hours on this CPU Mac; intended for multi-day runs on the Pi).
  This is the design's "gate the claim, not the codebase" case.
- **How to complete it:** follow `docs/pi-setup.md`; when `wr_greedy` exceeds
  0.70 (dashboard or `evaluate`), this criterion is met.

### 3. Browser play vs AI → annotated replay → replay view — ✅ MET (pipeline), UI spot-checked
- Verified end-to-end via the real FastAPI app (in-process TestClient driving the
  actual endpoints): trained a tiny checkpoint → served it → played a full
  71-move game vs `ckpt:latest` → **35 AI replies each carried a real MCTS
  annotation** (value, sims, prior-vs-visit `top`) → the game saved as a Replay →
  `load_replay` + `replay_states` re-derived all 72 states through the engine
  (verification-on-load) → the dashboard listed the run's learner/gate metrics
  and checkpoints.
- The browser UIs (Play, Replay, Training) were verified with headless-browser
  runs during Tasks 12/13/17/21 (full game through the canvas UI, zero console
  errors; search panel, hint overlay, win-prob chart, dashboard charts render).

### 4. Multi-hour resumable Pi training run — ⏸ DEFERRED TO HARDWARE
- Resumability is **proven in code**: the smoke test kills and resumes a run,
  asserting the step counter continues and the promoted-best pointer + game
  counter survive. SIGINT/SIGTERM clean-stop finishes the current iteration,
  flushes a checkpoint, exits 0.
- The multi-hour run on the physical Pi 5 requires the hardware (absent here).
  Full procedure — install, systemd unit, `--resume`, LAN dashboard — is in
  `docs/pi-setup.md`. Running it is a hardware step, not a code gap.

## Bug found and fixed during this verification

The final end-to-end pass caught a real bug (this is why e2e matters):
`encode_state` **raised** `RulesError` when a game board spread past 31 tiles in
one axis (reachable in long/linear games), which crashed `MctsAgent` mid-game —
breaking web play and, more seriously, would have crashed unattended self-play
training on the Pi. Fixed in `538ad69`: the encoder now crops out-of-window tiles
to the 31×31 window instead of raising (matching how `legal_mask` already prunes
out-of-window moves), with a regression test.

## Known follow-ups (iteration 2)

Surfaced by the final whole-implementation review; none block iteration 1, all
are hardening/operational or precision-of-claim items:

- **Checkpoint retention:** `training/` never prunes `step_*.pt` — a multi-day Pi
  run will fill the disk. Add a keep-last-N / keep-promoted policy.
- **Fresh-run determinism:** network weight init is unseeded (no
  `torch.manual_seed`), so a *from-scratch* run is not bit-reproducible even with
  the same `--seed` (replay/inference and resume-from-checkpoint *are*
  deterministic). Seed net construction, or keep the claim scoped as stated here.
- **`torch.load` hardening:** pass `weights_only=True` in `nn/checkpoint.py`
  (path traversal is already blocked; cheap defense-in-depth).
- **Web session concurrency:** FastAPI sync handlers run in a threadpool, so
  `SessionStore`'s dict has real multi-thread exposure; add a lock or `async def`
  (low risk for a single-user local tool; the current comment is imprecise).
- **Multiprocess self-play:** the design's worker pool is not built (self-play is
  sequential) — the main throughput lever for the Pi.
- **Small dedup:** winner-from-scores is computed in 4 places; `_unique_replay_path`
  is duplicated in `game/match.py` and `web/sessions.py`.

## Summary

Two of four criteria are fully met now (engine correctness; browser
play→replay→analytics pipeline). The other two are **built, tested, and
documented** but their final sign-off is a *runtime* outcome — a real training
run (criterion 2) and a run on the physical Pi (criterion 4) — not remaining
code. The system is a coherent, shippable iteration-1: a correct engine, a
complete RL-microscope web app, and a resumable AlphaZero training loop ready to
run.
