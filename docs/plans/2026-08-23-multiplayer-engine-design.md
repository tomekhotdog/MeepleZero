# Multiplayer Engine (2-5 Players, RL Stays 2-Player) Design

Approved scope: the simulator (engine, game layer, replay, CLI) and the web app
(Play + Replay views) support 2-5 players; the RL training stack (encoder,
network, MCTS, self-play, learner, arena) **strictly assumes 2 players** and
enforces that loudly. No implementation started yet (another session is active
on the repo).

## Research

Full research in `docs/plans/2026-08-23-multiplayer-3-5-research.md` (three
parallel codebase-analyzer agents: engine/game, RL stack, replay/web).
Load-bearing findings:

- **Already N-player-safe**: feature-majority scoring (`core/engine.py:113-132`,
  `core/game.py:72-95` — dict-based counting, tie-aware), ActionIndexer
  (`nn/actions.py` — player-count independent), PUCT selection math,
  RandomAgent, `is_terminal`, `board.js`/`overlays.js`.
- **2-player hardcoding is container shape + a few idioms**: fixed 2-tuples on
  GameState (`core/engine.py:36-40`), `current_player = 1 - p`
  (`engine.py:164`), no `num_players` in `GameConfig` (`core/config.py:6-9`),
  binary `_winner` (`game/match.py:82-85`, `game/replay.py:201-204`), replay
  schema hard gates (`replay.py:220-226,249-251,266-274`), 2-agent
  `play_game`/`run_match` signatures, `--p0/--p1` CLI, two-colour frontend
  palette + `1 - player` idiom + static two score cards.
- **RL stack fails silently, not loudly, on N>2**: encoder buckets all
  non-movers into one "opp" plane (`nn/encode.py:133-137`); MCTS negamax
  backup (`agents/mcts.py:264-270`) misattributes value. Guards are mandatory.
- **Checkpoints survive this change** — the encoder reads 2-player GameStates
  identically as long as 2-player semantics are untouched. No retraining.

## Problem Statement

The simulator should let 2-5 players play full base-game+abbot Carcassonne
(rules are natively 2-5 with unchanged supplies), so games with 3-5 seats can
be simulated, played in the browser, and reviewed as Replays. Multiplayer RL
(max^n MCTS, vector value heads, multiplayer gating) is deliberately deferred:
2-player AlphaZero is the well-trodden learning path; multiplayer MCTS is
research-flavoured and would invalidate every checkpoint.

## Solution

Widen the engine/game/replay/CLI/web layers to parametric `num_players`
(2-5), keep the RL stack byte-for-byte equivalent for 2-player games, and add
an enforced boundary: RL components raise on `num_players != 2`. GreedyAgent
is generalised (my delta minus best-of-rest delta — reduces exactly to current
behaviour at n=2) so 3-5 player games have a non-trivial opponent.

## User Stories

1. As a simulator user, I want `simulate` to run games with 2-5 agents, so I
   can study multiplayer dynamics.
2. As a browser player, I want to start a Play game with 1-4 AI opponents
   (random/greedy) and see per-seat scores/meeples, so I can play the real
   3-5 player game.
3. As a replay reviewer, I want the Replay view to render N-player games
   (N score chips, winners on the final move), so any simulated game is
   reviewable.
4. As the RL learner (the human), I want the RL microscope to stay exactly as
   it is for 2-player games, and want checkpoint agents to be cleanly
   unavailable in 3-5 player games rather than silently wrong.

## Implementation Decisions

### Core (`carcassonne/core`)

- `GameConfig(num_players: int = 2, meeples_per_player: int = 7)`; validate
  `2 <= num_players <= 5` in `new_game`. Default 2 keeps every existing
  call site and replay semantically unchanged.
- `GameState` per-player containers become variable-length tuples:
  `scores: tuple[int, ...]`, `meeples: tuple[int, ...]`,
  `abbots: tuple[bool, ...]`, `abbot_at: tuple[NodeId | None, ...]`.
  Tuples (not lists) preserve immutability/hashability. Player count is
  **derived**: `num_players(state) == len(state.scores)` — no redundant field.
- Turn advance: `current_player = (p + 1) % len(scores)` (`engine.py:164`).
  `apply` rebuilds per-player tuples generically (comprehensions, not
  positional literals).
- `final_scores` returns `tuple[int, ...]`.
- `Player = int  # 0..num_players-1` (`types.py`).
- Scoring logic untouched (already N-safe).

### Game layer (`carcassonne/game`)

- `play_game(agents: Sequence[Agent], ...)`; assert
  `len(agents) == num_players`. Loop body already generic.
- Winner representation: `GameResult.winners: tuple[int, ...]` — the tie set
  of top scorers (singleton = outright win, len>1 = shared). Replaces
  `winner: int | None`; a `winner_of(result) -> int | None` helper serves
  2-player callers (arena). Chosen over a full ranking field: ranking is
  derivable from `final_scores`, so storing it would be redundant data that
  can drift.
- Replay **v2** (`REPLAY_VERSION = 2`): header gains `num_players`; `agents`
  list length must equal it; per-move `score_after` and footer `final_scores`
  lists sized to it; footer `winners: list[int]`. Reader still accepts v1
  (maps `winner` → `winners`, `num_players=2`). Annotation shape is not
  hard-gated, so a future multiplayer-RL v-bump is avoided.
- `game/arena.py` stays strictly 2-agent (gating is an RL concern); it adapts
  to `winners` via the helper.

### Agents (`carcassonne/agents`)

- RandomAgent: no change.
- GreedyAgent: `_value` becomes
  `my_delta - max(opponent deltas)` (score and endgame-potential terms alike).
  At n=2 this is arithmetically identical to today — existing tests must pass
  unchanged, plus a new equivalence test pins it.
- MctsAgent: guard at `choose`/`choose_with_policy` entry — raise `RulesError`
  (or `ValueError`) when `len(state.scores) != 2`, message naming the
  limitation. Test pins the raise.

### NN / training (`carcassonne/nn`, `carcassonne/training`)

- `encode()` gets the same guard (raise on `num_players != 2`). Everything
  else — planes, value head, negamax backup, self-play targets, learner,
  arena gating, TrainConfig — is untouched. Existing checkpoints stay valid.

### CLI (`carcassonne/cli`)

- `simulate` gains `--agents random,greedy,greedy` (comma list, 2-5 entries);
  `--p0/--p1` kept as 2-player sugar (mutually exclusive with `--agents`).
  Summary printing loops over seats.
- `evaluate` unchanged (strictly 2-player head-to-head).

### Web (`carcassonne/web`)

- Sessions: `Session.agents: dict[int, Agent]` (seat → agent) with the human
  seat absent; `CreateGameBody` gains `num_players: int = 2`,
  `opponents: list[str]` (length `num_players - 1`), `human_player: int`
  (0..n-1). Server rejects checkpoint/MCTS opponents when `num_players > 2`
  with a clear error; the agent picker filters them client-side too.
- After each human move the session steps **all** consecutive AI seats until
  it is the human's turn again (generalising the current single `ai_step`).
- Frontend: palette extended to five player colours (`tiles.js` TOKENS,
  CSS `--p0..--p4`, swatch classes); `playerColor(p)` becomes a lookup;
  score cards rendered dynamically from state (replacing static `card-0/1`);
  `1 - S.humanPlayer` idioms replaced with "the seat that just moved";
  win/lose/draw messaging driven by `winners`.
- Hint overlay + value readout: available only in 2-player games (they need a
  checkpoint agent, which is 2-player-gated anyway) — UI hides them otherwise.
- Replay view: N score chips/deltas, winners text; win-probability chart and
  search panel render only for 2-player annotated replays, hidden otherwise.

### Deep-module note

The seam is `GameState` itself: everything downstream reads per-player data
through `len(state.scores)`-sized tuples, so no module needs a parallel
`num_players` parameter threaded through. The RL boundary is two one-line
guards at the only two entry points (encoder, MctsAgent), not a new layer.

## Testing Decisions

- Parametrise engine property tests over `num_players in 2..5`
  (`tests/core/test_properties.py`): meeple conservation per seat, replay
  determinism, legality invariants, turn rotation `(p+1) % n`.
- New engine tests: 3-way majority/tie scoring actually reached (logic was
  N-safe but untested beyond 2), endgame awards with 3+ claimants.
- Guard tests: encoder and MctsAgent raise on a 3-player state.
- Greedy: n=2 equivalence (same move choices on a fixed seed suite as the old
  formula) + a 3-player best-of-rest behaviour test.
- Replay: v2 round-trip at n=3/5; v1 file still loads (fixture); winners tie
  set correctness.
- Web: session test for consecutive-AI-seats stepping; API validation tests
  (opponent count, checkpoint rejection at n>2).
- Full existing suite must stay green throughout — it is the refactor guard.

## Out of Scope

- Multiplayer RL: max^n MCTS, vector value heads, encoder v2, multiplayer
  gating, retraining. (Design groundwork lives in the research doc.)
- Multi-human (true hotseat or networked) play — one human seat only.
- Farmers, other expansions, Gymnasium adapter, distributed training, auth.
- Arena/evaluate for N>2.
