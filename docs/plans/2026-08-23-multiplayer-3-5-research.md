# Research: supporting 3-5 players (assessment only — no design approved)

Status: research phase output. Spec currently lists 3+ players as explicitly out
of scope; adopting this requires a `design` iteration and a spec rewrite.

## TL;DR

The codebase splits cleanly into three tiers:

1. **Already N-player-safe** — scoring majority logic, ActionIndexer, PUCT
   selection math, RandomAgent, `is_terminal`, board/overlay JS.
2. **Mechanical widening** — every per-player 2-tuple in GameState, the
   `1 - p` turn toggle, replay schema gates, match/arena/CLI signatures,
   web UI seat hardcodings. Pervasive but shallow.
3. **Algorithmic redesign (the real work)** — the zero-sum scalar value
   convention shared by the value head, MCTS negamax backup, self-play
   targets, GreedyAgent, and the win-probability UI. This one convention is
   deliberately kept in lockstep across five modules and has no direct
   N-player generalisation. It gates everything else.

## Tier 1 — already N-player-safe (no changes)

- **Feature majority scoring**: `carcassonne/core/engine.py:113-132` and
  `carcassonne/core/game.py:72-95` count meeples in a `dict[Player, int]`,
  take `max(counts.values())`, and award all tied leaders. Correct for any
  number of claimants today.
- **Monastery/garden scoring**: single-owner by construction — count-agnostic.
- **ActionIndexer** (`carcassonne/nn/actions.py`): encodes cell × rotation ×
  per-tile action only; `ACTION_SPACE = 53816` is independent of player count.
- **PUCT selection** (`carcassonne/agents/mcts.py:240-244`): per-node move
  statistics; the coupling is in backup, not selection.
- **RandomAgent**, `is_terminal` (`carcassonne/core/game.py:49-50`),
  Dirichlet noise, D4 augmentation, `board.js`/`overlays.js`,
  training-view charts.

## Tier 2 — mechanical widening (shallow, but everywhere)

### Core engine
- `GameState` per-player fields are fixed 2-tuples: `scores`, `meeples`,
  `abbots`, `abbot_at` (`carcassonne/core/engine.py:36-40`); rebuilt
  positionally in `apply` (`engine.py:165-168`); seeded as 2-tuples in
  `new_game` (`carcassonne/core/game.py:39-42`).
- `type Player = int  # 0 | 1` (`carcassonne/core/types.py:6`) — comment-only
  contract.
- Turn advance is the single line `current_player=1 - p`
  (`engine.py:164`) → becomes `(p + 1) % n_players`.
- `GameConfig` (`carcassonne/core/config.py:6-9`) has no `num_players` field;
  supplies (7 meeples + 1 abbot per player) are per-seat constants that are
  correct for base Carcassonne at any count of 2-5 — only tuple shape changes.
- `final_scores` returns literal `(scores[0], scores[1])`
  (`game.py:58`).

### Game layer / replay / CLI
- `play_game(agents: tuple[Agent, Agent], ...)` (`carcassonne/game/match.py:32`);
  loop body already indexes `agents[player]` generically.
- `_winner` binary comparison duplicated at `match.py:82-85` and
  `replay.py:201-204`; `GameResult.winner: int | None`.
- Replay schema: no player-count field; hard gates reject N≠2 —
  `len(agents) == 2` (`replay.py:220-226`), `player in (0, 1)`
  (`replay.py:249-251`), `len(scores) == 2` (`replay.py:266-274`).
  Widening is a `REPLAY_VERSION` bump; `winner` should become
  ranking/tie-set data.
- CLI `--p0/--p1` flags, `wins = [0, 0]` (`carcassonne/cli/main.py:34-43,91-144`).
- Arena `run_match(a, b)` 2-agent signature, binary seat-swap `_flipped`
  (`carcassonne/game/arena.py:25-72`) — N players need seat permutations
  (round-robin rotation) instead of one swap.

### Web
- API: `CreateGameBody.human_player: Literal[0, 1]` (`carcassonne/web/app.py:38-40`);
  `Session` holds one `opponent: Agent` — architecture is 1-human-vs-1-agent,
  N seats means a list of agents keyed by seat (`carcassonne/web/sessions.py:50-51,126-158`).
- Frontend: two-color palette `p0/p1` only (`tiles.js:24-25,45`, `style.css`),
  `1 - S.humanPlayer` idiom (`app.js:210,295`), two static score cards
  (`index.html:38-47`), `for (const seat of [0, 1])` (`app.js:589-600`),
  replay chips/labels for indices 0/1 (`replay.js:279-298`).

### Tests
No test parametrises player count. Files encoding 2 seats:
`tests/core/test_properties.py` (meeple conservation `for p in (0,1)`),
`tests/core/test_engine.py`, `test_game.py`, `tests/agents/test_greedy.py`,
`tests/game/test_arena.py`, `test_replay.py`, plus the nn/training suites
implicitly via `NUM_PLANES == 62` and ±1 value targets.

## Tier 3 — algorithmic redesign (the gating decisions)

One convention, five modules, all deliberately in lockstep (self-play docstring
says so at `carcassonne/training/selfplay.py:6-9`):

| Module | Zero-sum idiom |
|---|---|
| Value head | single tanh scalar in [-1,1], "current player's perspective" (`nn/model.py:129-134`) |
| MCTS backup | negamax `value = -value` per ply (`agents/mcts.py:264-270`) |
| MCTS terminal | `winner = argmax(s0, s1)` → ±1/0 (`agents/mcts.py:91-99`) |
| Self-play target | `_value_for(mover, winner)` → ±1/0 scalar (`training/selfplay.py:86-96`) |
| Learner | scalar MSE on that target (`training/learner.py:134`) |

**Why negamax breaks for N>2** (RL context): with 2 alternating players,
"child node's value from its mover's perspective" is exactly the negative of
the parent's — one scalar carries both players' outcomes. With 3+ players,
opponent A's loss is not necessarily my gain (it may be opponent B's), so a
single signed scalar can't be flipped through the tree; intermediate players
get misattributed value.

**Standard fix — max^n MCTS with a value vector**: the value head outputs an
N-vector (one expected payoff per seat, e.g. stored in fixed seat order or
rotated to relative-seat order); backup propagates the whole vector unchanged;
each node's PUCT selection uses the component belonging to *that node's mover*.
Terminal payoffs come from the final ranking (e.g. linear in rank:
1st=+1 … last=-1, splitting ties), not a win bit — a win-only signal is far
sparser with 4-5 players and makes early training noisier.

Downstream consequences:
- **Encoder** (`nn/encode.py`): the fixed 62-plane me/opp layout
  (`opp = 1 - me` at :96-97, binary ownership planes at :133-137) must become
  relative-seat planes (me, next-to-act, next+1, …) padded to max players,
  plus a num-players scalar plane. `NUM_PLANES` changes → **all existing
  checkpoints are invalid; retrain from scratch**.
- **Network**: value head `Linear(64, 1)` → `Linear(64, N_max)` (masked/
  padded for fewer seats) or a per-count head.
- **Self-play/learner**: vector targets, vector MSE. Buffer schema changes.
- **GreedyAgent** (`agents/greedy.py:42,59-70`): `opp = 1 - me` and
  "my delta minus opp delta" need a redesign — simplest sound choice is
  "my delta minus best-of-rest delta".
- **Arena/gating** (`training/arena.py`): head-to-head win-rate >55% doesn't
  transfer. Options: keep gating 2-player only; or N-seat matches of
  1 candidate + (N-1) current-best with full seat rotation, promoting on
  expected rank / win share vs the 1/N baseline.
- **Win-probability chart** (`replay.js:380-396`): currently maps the
  zero-sum scalar to one P0 line; becomes N lines (or stacked shares) once
  values are vectors — mechanically easy after the semantics change.

## Game-rules note

Base Carcassonne is natively 2-5 players with unchanged rules — no new rule
logic needed. The 72-tile deck is fixed, so total plies stay ~71 regardless of
seat count; each player just takes fewer turns (≈14 at 5 players), which means
fewer positions per seat per game and higher variance in credit assignment —
expect slower learning per game generated, relevant to the Pi 5 budget.

## Open design questions (for the interview, if pursued)

1. Which counts: parametric 2-5, or a fixed second mode (e.g. 4)?
2. One network for all counts (relative-seat padded encoder + count plane) vs
   one network per count?
3. Value target: rank-linear vs win-only vs normalised score share?
4. Gating design for multiplayer (and does GreedyAgent stay the yardstick)?
5. Does the web Play view need N seats (multi-human? human + several AIs?),
   or is multiplayer self-play/replay-only at first?
6. Keep 2-player replays loadable (schema versioning / migration)?

## Variant under consideration (2026-08-23): engine 2-5 players, RL strictly 2-player

Scope split: `core` + game layer + replay + CLI (and optionally web play)
become N-player; encoder, network, MCTS, self-play, learner, arena stay
2-player-only.

**What it saves** — all of Tier 3: no encoder v2, no value-vector head, no
max^n backup, no vector targets, no multiplayer gating, and crucially **no
retrain — existing checkpoints stay valid**, because the encoder reads a
2-player GameState identically as long as 2-player semantics are unchanged.
Win-probability chart and value readout stay as they are.

**What it still costs** — all of Tier 2 in core/game/replay/CLI: the container
widening, `(p+1) % n`, replay v2 with an explicit player-count field, match
runner over N agents, ranking-shaped winner data. This is the majority of the
*file count* of full multiplayer, just not the majority of the difficulty.
Web N-seat play is a separate optional slice.

**What it newly creates** (work that exists in neither status quo nor full
multiplayer):
1. **An enforcement boundary the type system won't check.** MctsAgent +
   encoder must *reject* states with `num_players != 2` loudly. Without a
   guard they fail silently: the encoder buckets every non-mover into the
   single "opp" plane (`encode.py:133-137`) and negamax backup misattributes
   values — plausible-looking garbage, not a crash. Guards + tests are
   mandatory, not optional hardening.
2. **A GreedyAgent decision.** Its value function is 2-player
   (`greedy.py:42,59-70`). Either guard it 2-player-only (then RandomAgent is
   the *only* multiplayer opponent, making the mode hollow) or generalise the
   heuristic to "my delta minus best-of-rest delta" (small, self-contained,
   recommended).
3. **Mode-conditional UI.** Agent picker must filter checkpoint agents out of
   3-5 player games; hint overlay and win-prob chart are 2-player-only paths;
   replay view must render N-player replays without the search panel.
4. **Replay v2 designed for the future.** Schema should carry `num_players`
   and not hard-gate annotation shape, so full multiplayer later is not a v3.

**Product consequence, stated plainly**: in 3-5 player games there is no
trained AI — the RL microscope (hints, priors-vs-visits, win-prob) remains a
2-player feature. Multiplayer is human-vs-human/random/greedy until Tier 3 is
done.

**Strategic read**: the engine widening is a strict subset of full multiplayer
and none of it is throwaway; the guarded seam makes later Tier-3 work cleanly
scoped. Deferring multiplayer RL is also the right learning path — 2-player
AlphaZero is the well-trodden ground; multiplayer MCTS (max^n) is
research-flavoured. Main risk is rot: nothing strong exercises the 3-5 path,
so engine property tests must parametrise over `num_players` to keep it honest.

## Rough shape of the work (if approved)

1. Core widening: `num_players` in `GameConfig`, list-shaped per-player state,
   `(p+1) % n`, generic `final_scores`/ranking — plus test parametrisation.
2. Game layer: match/arena/replay v2/CLI generalisation.
3. RL redesign: encoder v2, vector value head, max^n backup, vector targets,
   multiplayer gating. Retrain from scratch.
4. Web: seat-list sessions, N-color palette, dynamic score cards, N-line
   win chart.

Phases 1-2 are mostly mechanical refactoring guarded by existing tests;
phase 3 is the substantive RL work; phase 4 is broad-but-shallow UI work.
