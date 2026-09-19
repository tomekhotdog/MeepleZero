# Multiplayer Engine (2-5 Players) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use tomek-superpowers:build to implement this plan task-by-task.

> ⚠️ **DO NOT START while another session is active on this repo.**
> Before executing, re-read the touched files against HEAD — this plan was
> authored 2026-08-23 against commit `c3df249`-era sources; exact line anchors
> and code snippets may need rebasing. Verify each `old` snippet exists before
> editing.

**Goal:** Engine, game layer, replay format, CLI, and web app support 2-5 players; the RL stack (encoder, MCTS, self-play, learner, arena) stays strictly 2-player and raises loudly otherwise.

**Architecture:** Player count is data, not structure: `GameConfig.num_players` sizes the per-player tuples on `GameState`, and every consumer derives N from `len(state.scores)`. Game outcome becomes a **Winners** tie set. Replay bumps to v2 (v1 stays loadable). The RL boundary is two one-line guards (encoder, MctsAgent).

**Tech Stack:** Python 3.12, frozen dataclasses, pytest+hypothesis, FastAPI + vanilla JS canvas. Test command: `uv run pytest` (add `-q`); lint/type: `uv run ruff check .` and `uv run mypy carcassonne`.

**Design:** `docs/plans/2026-08-23-multiplayer-engine-design.md` (research: `docs/plans/2026-08-23-multiplayer-3-5-research.md`)

**Conventions to preserve:** frozen/slots dataclasses; tuples for immutable per-player state; `RulesError` for rules/format violations, `IllegalMove` for illegal play; import layering `core → game → agents/nn → training/web`; commit style `feat(scope): …`.

---

### Task 1: Core widening — `num_players` in config, N-sized GameState, turn rotation
**Depends on:** none

**Files:**
- Modify: `carcassonne/core/config.py`, `carcassonne/core/types.py`, `carcassonne/core/engine.py`, `carcassonne/core/game.py`
- Modify (annotation-only, keeps mypy green): `carcassonne/game/match.py`, `carcassonne/game/replay.py`, `carcassonne/training/selfplay.py`
- Test: `tests/core/test_properties.py`, `tests/core/test_game.py`

**Step 1: Write the failing tests**

Rewrite `tests/core/test_properties.py` (whole file — it currently hardcodes `for p in (0, 1)` and `fs[0]/fs[1]`):

```python
import random

from hypothesis import given, settings
from hypothesis import strategies as st

from carcassonne import core


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 10_000), n_players=st.integers(2, 5))
def test_random_game_invariants(seed: int, n_players: int) -> None:
    config = core.GameConfig(num_players=n_players)
    state = core.new_game(seed, config)
    rng = random.Random(seed)
    while not core.is_terminal(state):
        moves = core.legal_moves(state)
        assert moves, "legal_moves must be non-empty on non-terminal state"
        prev = state.scores
        state = core.apply(state, rng.choice(moves))
        assert state.current_player == state.turn % n_players  # strict rotation
        assert all(s >= p for s, p in zip(state.scores, prev, strict=True))
        for p in range(n_players):
            on_board = sum(m.player == p for d in state.features.data.values() for m in d.meeples)
            in_supply = state.meeples[p] + int(state.abbots[p])
            assert on_board + in_supply == 8
    fs = core.final_scores(state)
    assert len(fs) == n_players
    assert all(f >= s for f, s in zip(fs, state.scores, strict=True))


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 10_000), n_players=st.integers(2, 5))
def test_replay_determinism(seed: int, n_players: int) -> None:
    config = core.GameConfig(num_players=n_players)
    s1, s2 = core.new_game(seed, config), core.new_game(seed, config)
    rng = random.Random(seed)
    while not core.is_terminal(s1):
        mv = rng.choice(core.legal_moves(s1))
        s1, s2 = core.apply(s1, mv), core.apply(s2, mv)
        assert s1.scores == s2.scores and s1.board == s2.board and s1.deck == s2.deck
```

Append to `tests/core/test_game.py`:

```python
import pytest

from carcassonne.core import GameConfig, RulesError, new_game


def test_new_game_sizes_per_player_state() -> None:
    state = new_game(0, GameConfig(num_players=4))
    assert state.scores == (0, 0, 0, 0)
    assert state.meeples == (7, 7, 7, 7)
    assert state.abbots == (True, True, True, True)
    assert state.abbot_at == (None, None, None, None)


def test_new_game_default_is_two_players() -> None:
    assert new_game(0).scores == (0, 0)


@pytest.mark.parametrize("n", [0, 1, 6])
def test_new_game_rejects_bad_player_count(n: int) -> None:
    with pytest.raises(RulesError):
        new_game(0, GameConfig(num_players=n))
```

(Adapt imports to the file's existing style — it already imports from `carcassonne.core`.)

**Step 2: Run tests to verify they fail**
Run: `uv run pytest tests/core/test_properties.py tests/core/test_game.py -q`
Expected: FAIL — `GameConfig` has no `num_players` field (TypeError).

**Step 3: Implement**

`carcassonne/core/config.py`:

```python
@dataclass(frozen=True, slots=True)
class GameConfig:
    num_players: int = 2  # 2..5; validated by new_game
    meeples_per_player: int = 7
    # abbots are always 1 per player in the base+abbot ruleset
```

`carcassonne/core/types.py:6`: `type Player = int  # 0..num_players-1 (seat index)`

`carcassonne/core/engine.py` — GameState fields (lines 37-40) become variadic:

```python
    scores: tuple[int, ...]
    meeples: tuple[int, ...]         # plain meeples in supply (7 per seat at start)
    abbots: tuple[bool, ...]         # abbot in supply?
    abbot_at: tuple[NodeId | None, ...]
```

`apply` tail (lines 164-168):

```python
        current_player=(p + 1) % len(state.scores),
        scores=tuple(scores),
        meeples=tuple(meeples),
        abbots=tuple(abbots),
        abbot_at=tuple(abbot_at),
```

`carcassonne/core/game.py` — import `RulesError` from `carcassonne.core.types`; in `new_game` (before building the state):

```python
    n = config.num_players
    if not 2 <= n <= 5:
        raise RulesError(f"num_players must be 2..5, got {n}")
```

and the per-player fields (lines 39-42):

```python
        scores=(0,) * n,
        meeples=(m,) * n,
        abbots=(True,) * n,
        abbot_at=(None,) * n,
```

`final_scores` (lines 53-58): return type `tuple[int, ...]`, body ends `return tuple(scores)`.

Annotation-only widenings so `mypy --strict` stays green (no logic change):
- `carcassonne/game/match.py`: `GameResult.final_scores: tuple[int, ...]`; `_winner(scores: tuple[int, ...])`.
- `carcassonne/game/replay.py`: `MoveRecord.score_after`, `Replay.final_scores`, `ReplayWriter.record(score_after=...)`, `finish(final_scores: tuple[int, ...])`, `_winner(scores: tuple[int, ...])` — types only; the `len(v) == 2` gates stay (they fall in Task 2).
- `carcassonne/training/selfplay.py`: `_winner(scores: tuple[int, ...])` — annotation only; still 2-player logic, guarded in Task 4.

**Step 4: Verify**
Run: `uv run pytest -q && uv run mypy carcassonne && uv run ruff check .`
Expected: full suite PASS (existing 2-player tests unchanged — `(0,)*2 == (0, 0)`), mypy clean.

**Step 5: Commit**
```bash
git add carcassonne/core tests/core carcassonne/game/match.py carcassonne/game/replay.py carcassonne/training/selfplay.py
git commit -m "feat(core): parametric 2-5 player GameState, config and turn rotation"
```

---

### Task 2: Replay v2 — `num_players`, N-sized scores, `winners` tie set, v1 back-compat
**Depends on:** Task 1

**Files:**
- Modify: `carcassonne/game/replay.py`
- Test: `tests/game/test_replay.py`

**Step 1: Write the failing tests** (append to `tests/game/test_replay.py`, matching its existing helpers/imports):

```python
def test_winners_of_tie_sets() -> None:
    from carcassonne.game.replay import winners_of

    assert winners_of((5, 3)) == (0,)
    assert winners_of((3, 3)) == (0, 1)
    assert winners_of((1, 7, 7, 2)) == (1, 2)
    assert winners_of((0, 0, 0, 0, 0)) == (0, 1, 2, 3, 4)


def test_three_player_replay_round_trips(tmp_path: Path) -> None:
    """Write a full random 3-player game through ReplayWriter; load + verify."""
    import random as _random

    config = GameConfig(num_players=3)
    state = new_game(11, config)
    header = ReplayHeader(
        v=REPLAY_VERSION, seed=11, config=config,
        agents=("random", "random", "random"), checkpoint=None, started_at="t",
    )
    path = tmp_path / "g.jsonl"
    rng = _random.Random(11)
    with ReplayWriter(path, header) as w:
        n = 0
        while not is_terminal(state):
            player, tile = state.current_player, state.current_tile
            move = rng.choice(legal_moves(state))
            state = apply(state, move)
            w.record(n, player, tile, move, state.scores)
            n += 1
        w.finish(final_scores(state))
    replay = load_replay(path)
    assert replay.header.config.num_players == 3
    assert len(replay.final_scores) == 3
    assert replay.winners == winners_of(replay.final_scores)
    list(replay_states(replay))  # engine-verified; must not raise


def test_v1_two_player_replay_still_loads(tmp_path: Path) -> None:
    """Take a real v2 file and rewrite header/end lines to the v1 schema."""
    import json as _json

    # produce a valid 2-player v2 file first (reuse the loop above with defaults)
    config = GameConfig()
    ...  # same write loop as above with agents=("random", "random"), seed=5
    lines = path.read_text(encoding="utf-8").splitlines()
    header = _json.loads(lines[0])
    header["v"] = 1
    header["config"] = {"meeples_per_player": header["config"]["meeples_per_player"]}
    end = _json.loads(lines[-1])
    ws = end.pop("winners")
    end["winner"] = ws[0] if len(ws) == 1 else None
    v1 = tmp_path / "v1.jsonl"
    v1.write_text("\n".join([_json.dumps(header), *lines[1:-1], _json.dumps(end)]) + "\n")
    replay = load_replay(v1)
    assert replay.header.v == 1
    assert replay.header.config.num_players == 2
    assert replay.winners == tuple(ws)
    list(replay_states(replay))
```

(Fill the elided write-loop; keep it a shared local helper `_write_game(tmp_path, config, agents, seed)` used by both tests. Also mechanically update the file's existing assertions: any `replay.winner ==` → `replay.winners ==` with tie-set values, and `_header_from_json`-shape tests to include `num_players` in v2 configs.)

**Step 2: Verify fail**
Run: `uv run pytest tests/game/test_replay.py -q`
Expected: FAIL — `winners_of` doesn't exist; `ReplayHeader.agents` 3-tuple rejected by writer/loader.

**Step 3: Implement** in `carcassonne/game/replay.py`:

- `REPLAY_VERSION = 2`; add `_SUPPORTED_VERSIONS = (1, 2)`.
- `ReplayHeader.agents: tuple[str, ...]`.
- Replace both `_winner` duplicates (here and the check in `replay_states`) with:

```python
def winners_of(scores: tuple[int, ...]) -> tuple[int, ...]:
    """Tie set of top scorers: singleton = outright win, larger = shared."""
    best = max(scores)
    return tuple(p for p, s in enumerate(scores) if s == best)
```

- `Replay.winner: int | None` → `winners: tuple[int, ...]`.
- Writer `__init__`: `config` dict gains `"num_players": header.config.num_players`; validate `len(header.agents) == header.config.num_players` (RulesError otherwise); keep the version check against `REPLAY_VERSION` (the writer only writes v2).
- `finish`: write `"winners": list(winners_of(final_scores))` instead of `"winner"`.
- `_header_from_json`: accept `v in _SUPPORTED_VERSIONS`. Config shape by version — v1: exactly `{"meeples_per_player"}`, `num_players=2`; v2: exactly `{"meeples_per_player", "num_players"}`. Agents gate becomes `len(agents_raw) == config.num_players`.
- `load_replay` end-line: v2 reads `winners` (validated: non-empty list of ints, each `0 <= w < num_players`, strictly increasing); v1 reads `winner` and maps `None → (0, 1)` (a v1 draw is a two-way tie), `w → (w,)`.
- `_move_record_from_json` / `_scores_field`: thread `num_players` through (`player` gate `0 <= player < num_players`; scores gate `len(v) == num_players`).
- `replay_states`: final check `winners_of(finals) == replay.winners`.

**Step 4: Verify**
Run: `uv run pytest tests/game -q && uv run mypy carcassonne`
Expected: PASS (note: `tests/game/test_arena.py` and web/CLI tests still pass because they read `GameResult`, not `Replay.winner` — `Replay.winner` consumers in `web/app.py` break at import/runtime only when exercised; if `uv run pytest -q` shows failures in `tests/web` from the renamed field, apply the two-line `app.py` fallback from Task 7 Step 3 early, or do Tasks 2+7's `_list_replays` change together — surface whichever you did in the commit message.)

**Step 5: Commit**
```bash
git add carcassonne/game/replay.py tests/game/test_replay.py
git commit -m "feat(game): replay v2 - num_players header, winners tie set, v1 loader compat"
```

---

### Task 3: Match runner over N agents + arena adaptation
**Depends on:** Task 2

**Files:**
- Modify: `carcassonne/game/match.py`, `carcassonne/game/arena.py`, `carcassonne/cli/main.py` (winner→winners prints), `carcassonne/training/arena.py` (only if it touches `.winner` — check)
- Test: `tests/game/test_arena.py`, new `tests/game/test_match_multiplayer.py`

**Step 1: Failing test** — `tests/game/test_match_multiplayer.py`:

```python
"""play_game generalised to N seats; GameResult carries a winners tie set."""

from __future__ import annotations

from pathlib import Path

from carcassonne.agents import GreedyAgent, RandomAgent
from carcassonne.core import GameConfig
from carcassonne.game.match import play_game
from carcassonne.game.replay import load_replay, replay_states, winners_of


def test_three_player_game_runs_and_records(tmp_path: Path) -> None:
    result = play_game(
        (GreedyAgent(), RandomAgent(), RandomAgent()),
        seed=9,
        replay_dir=tmp_path,
        config=GameConfig(num_players=3),
    )
    assert len(result.final_scores) == 3
    assert result.winners == winners_of(result.final_scores)
    replay = load_replay(result.replay_path)
    assert replay.header.agents == ("greedy", "random", "random")
    list(replay_states(replay))


def test_agent_count_must_match_config() -> None:
    import pytest

    from carcassonne.core import RulesError

    with pytest.raises(RulesError):
        play_game((RandomAgent(), RandomAgent()), seed=1, config=GameConfig(num_players=3))
```

**Step 2: Verify fail**
Run: `uv run pytest tests/game/test_match_multiplayer.py -q`
Expected: FAIL — `GameResult` has no `winners`; agent-count mismatch not checked.

**Step 3: Implement**

`carcassonne/game/match.py`:
- `from collections.abc import Sequence`; import `winners_of` from `carcassonne.game.replay`; import `RulesError` from `carcassonne.core`.
- `GameResult`: replace `winner: int | None` with `winners: tuple[int, ...]`.
- `play_game(agents: Sequence[Agent], ...)`; first line of body:

```python
    if len(agents) != config.num_players:
        raise RulesError(f"{len(agents)} agents for a {config.num_players}-player game")
```

- Header: `agents=tuple(a.name for a in agents)`.
- Delete local `_winner`; result: `winners=winners_of(finals)`.
- `_unique_replay_path`: `names = "-vs-".join(a.name for a in agents)` in the raw string; parameter type `Sequence[Agent]`.

`carcassonne/game/arena.py` (stays strictly 2-agent):
- Add at module level:

```python
def winner_2p(winners: tuple[int, ...]) -> int | None:
    """Collapse a 2-player winners tie set to the old 0/1/None convention."""
    return winners[0] if len(winners) == 1 else None
```

- Tally: `w = winner_2p(result.winners)`; `draws += 1` when `w is None` else `wins[w] += 1`.
- `_flipped`: `winners=tuple(sorted(1 - w for w in r.winners))`.

`carcassonne/cli/main.py` `_simulate`/`_evaluate` line prints: `winner={result.winner}` → `winners={list(result.winners)}`; the `wins[result.winner]` tallies switch to `winner_2p` (import from `carcassonne.game.arena`) — `_simulate`'s tally is reworked properly in Task 6; here just keep it compiling for 2p.

Check `carcassonne/training/arena.py` and `tests/cli/test_evaluate.py` for `.winner` usages and apply the same `winner_2p` mapping. Update `tests/game/test_arena.py` lines 25-26: `winner = None if raw.winner is None else 1 - raw.winner` becomes winners-based (`tuple(sorted(1 - w for w in raw.winners))`).

**Step 4: Verify**
Run: `uv run pytest -q && uv run mypy carcassonne`
Expected: full suite PASS.

**Step 5: Commit**
```bash
git add carcassonne/game carcassonne/cli/main.py carcassonne/training/arena.py tests/game tests/cli
git commit -m "feat(game): N-agent match runner; GameResult.winners tie set"
```

---### Task 4: RL boundary guards — encoder and MctsAgent raise on `num_players != 2`
**Depends on:** Task 1 (parallel with Tasks 2-3)

**Files:**
- Modify: `carcassonne/nn/encode.py`, `carcassonne/agents/mcts.py`
- Test: `tests/nn/test_encode.py`, `tests/agents/test_mcts.py`

**Step 1: Failing tests** — append:

`tests/nn/test_encode.py`:

```python
def test_encode_rejects_non_two_player_state() -> None:
    from carcassonne.core import GameConfig, RulesError, new_game
    from carcassonne.nn.encode import encode_state

    with pytest.raises(RulesError, match="2-player"):
        encode_state(new_game(0, GameConfig(num_players=3)))
```

`tests/agents/test_mcts.py` (reuse the file's tiny-net fixture pattern):

```python
def test_mcts_rejects_non_two_player_state() -> None:
    from carcassonne.core import GameConfig, RulesError, new_game

    agent = _tiny_agent(sims=1)  # match the fixture/helper name in this file
    with pytest.raises(RulesError, match="2-player"):
        agent.choose(new_game(0, GameConfig(num_players=3)), TurnContext(rng=random.Random(0)))
```

**Step 2: Verify fail**
Run: `uv run pytest tests/nn/test_encode.py tests/agents/test_mcts.py -q`
Expected: the two new tests FAIL (encoder silently buckets the 3rd player into the opp plane today; no raise).

**Step 3: Implement**

`carcassonne/nn/encode.py` — first lines of `encode_state`, and import `RulesError` from `carcassonne.core.types`:

```python
    if len(state.scores) != 2:
        raise RulesError(
            f"the encoder is 2-player only (me/opp planes); got a {len(state.scores)}-player state"
        )
```

`carcassonne/agents/mcts.py` — first lines of both `choose` and `choose_with_policy` (RulesError is already imported):

```python
        if len(state.scores) != 2:
            raise RulesError(
                f"MctsAgent is 2-player only (negamax backup); got a {len(state.scores)}-player state"
            )
```

Also extend the module docstring's convention list with one line noting the guard.

**Step 4: Verify**
Run: `uv run pytest tests/nn tests/agents tests/training -q`
Expected: PASS (training suite proves the guard doesn't trip 2-player self-play).

**Step 5: Commit**
```bash
git add carcassonne/nn/encode.py carcassonne/agents/mcts.py tests/nn tests/agents
git commit -m "feat(nn,agents): loud 2-player guards on encoder and MctsAgent"
```

---

### Task 5: GreedyAgent best-of-rest generalisation
**Depends on:** Task 1 (parallel with Tasks 2-4)

**Files:**
- Modify: `carcassonne/agents/greedy.py`
- Test: `tests/agents/test_greedy.py`

**Step 1: Failing tests** — append to `tests/agents/test_greedy.py`:

```python
def test_value_equals_old_two_player_formula() -> None:
    """Best-of-rest reduces exactly to (my - opp) at n=2: prove over a real game."""
    from carcassonne.agents.greedy import _W_POTENTIAL, _W_SUPPLY, _value
    from carcassonne.core import apply as core_apply
    from carcassonne.core import new_game
    from dataclasses import replace as dc_replace

    state = new_game(4)
    rng = random.Random(4)
    for _ in range(20):  # 20 plies deep, checking every legal move at each
        if not legal_moves(state):
            break
        me, opp = state.current_player, 1 - state.current_player
        pot_me = endgame_potential(state, me)
        pot_opp = endgame_potential(state, opp)
        probe = dc_replace(state, deck=())
        for move in legal_moves(state):
            nxt = core_apply(probe, move)
            supply = nxt.meeples[me] + (1 if nxt.abbots[me] else 0)
            old = (
                (nxt.scores[me] - probe.scores[me])
                - (nxt.scores[opp] - probe.scores[opp])
                + _W_POTENTIAL * (endgame_potential(nxt, me) - pot_me)
                - _W_POTENTIAL * (endgame_potential(nxt, opp) - pot_opp)
                + _W_SUPPLY * supply
            )
            assert _value(probe, move, me, pot_me) == pytest.approx(old)
        state = core_apply(state, rng.choice(legal_moves(state)))


def test_greedy_plays_a_legal_three_player_game() -> None:
    from carcassonne.core import GameConfig
    from carcassonne.game.match import play_game

    r = play_game(
        (GreedyAgent(), RandomAgent(), RandomAgent()), seed=2, config=GameConfig(num_players=3)
    )
    assert len(r.final_scores) == 3  # play_game verifies legality throughout
```

(Adjust the `_value` signature in the first test to whatever Step 3 lands on; the point is exact numeric equivalence with the old formula at n=2. The 3-player test depends on Task 3's `play_game`; if running Task 5 before Task 3, drive the loop manually with `apply`.)

**Step 2: Verify fail**
Run: `uv run pytest tests/agents/test_greedy.py -q`
Expected: FAIL — `_value`'s current signature takes `opp`, and a 3-player game crashes on `opp = 1 - me` producing a bogus seat.

**Step 3: Implement** — `carcassonne/agents/greedy.py`:

Update the module docstring formula to:

```
    value(move) = (my_score_delta + 0.7 * Δ(my endgame_potential))
                - max over opponents o of (o_score_delta + 0.7 * Δ(o endgame_potential))
                + 0.15 * my pieces in supply after the move
```

`choose` (drop `opp`/`pot_opp`, keep everything else):

```python
        me: Player = state.current_player
        pot_me = endgame_potential(state, me)
        ...
        values = [_value(probe, move, me, pot_me) for move in moves]
```

`_value`:

```python
def _value(state: GameState, move: Move, me: Player, pot_me: int) -> float:
    nxt = apply(state, move)
    supply = nxt.meeples[me] + (1 if nxt.abbots[me] else 0)
    my_gain = (nxt.scores[me] - state.scores[me]) + _W_POTENTIAL * (
        endgame_potential(nxt, me) - pot_me
    )
    best_rest = max(
        (nxt.scores[o] - state.scores[o])
        + _W_POTENTIAL * (endgame_potential(nxt, o) - endgame_potential(state, o))
        for o in range(len(state.scores))
        if o != me
    )
    return my_gain - best_rest + _W_SUPPLY * supply
```

Note: this recomputes each opponent's pre-move potential per move (the old code hoisted `pot_opp` out of the loop). If `test_greedy_crushes_random` gets noticeably slower, hoist a `pot_others: dict[Player, int]` in `choose` and pass it through — same numbers, same speed as before.

**Step 4: Verify**
Run: `uv run pytest tests/agents/test_greedy.py -q`
Expected: PASS, including the untouched exact-value test `test_greedy_annotates_when_asked` (proof of n=2 equivalence) and `test_greedy_crushes_random`.

**Step 5: Commit**
```bash
git add carcassonne/agents/greedy.py tests/agents/test_greedy.py
git commit -m "feat(agents): greedy values my-gain minus best-of-rest (2-5 players)"
```

---

### Task 6: CLI — `simulate --agents` for 2-5 players
**Depends on:** Task 3

**Files:**
- Modify: `carcassonne/cli/main.py`
- Test: `tests/cli/test_simulate.py`

**Step 1: Failing test** — append to `tests/cli/test_simulate.py`:

```python
def test_simulate_four_players_with_agents_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        ["simulate", "--agents", "greedy,random,random,random",
         "--seed", "5", "--out", str(tmp_path)]
    )
    assert rc == 0
    (path,) = tmp_path.glob("*.jsonl")
    replay = load_replay(path)
    assert replay.header.agents == ("greedy", "random", "random", "random")
    assert replay.header.config.num_players == 4
    list(replay_states(replay))
    out = capsys.readouterr().out
    assert "winners=" in out and "summary:" in out


def test_simulate_rejects_both_agent_styles(tmp_path: Path) -> None:
    rc = main(
        ["simulate", "--agents", "random,random", "--p0", "random", "--p1", "random",
         "--seed", "1", "--no-replay", "--out", str(tmp_path)]
    )
    assert rc != 0
```

**Step 2: Verify fail**
Run: `uv run pytest tests/cli/test_simulate.py -q`
Expected: FAIL — argparse: unrecognized `--agents`.

**Step 3: Implement** — `carcassonne/cli/main.py`:

Parser: make `--p0/--p1` non-required, add:

```python
    sim.add_argument("--p0", help="agent spec for player 0 (2-player shorthand)")
    sim.add_argument("--p1", help="agent spec for player 1 (2-player shorthand)")
    sim.add_argument(
        "--agents",
        help="comma-separated agent specs for all seats, 2-5 (e.g. 'greedy,random,random')",
    )
```

`_simulate` spec resolution + generic tally:

```python
    if args.agents and (args.p0 or args.p1):
        print("error: use either --agents or --p0/--p1, not both", file=sys.stderr)
        return 2
    if args.agents:
        specs = [s.strip() for s in args.agents.split(",")]
    elif args.p0 and args.p1:
        specs = [args.p0, args.p1]
    else:
        print("error: pass --agents, or both --p0 and --p1", file=sys.stderr)
        return 2
    if not 2 <= len(specs) <= 5:
        print(f"error: need 2-5 agents, got {len(specs)}", file=sys.stderr)
        return 2
    try:
        agents = tuple(make_agent(s) for s in specs)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    config = GameConfig(num_players=len(agents))
```

Loop: pass `config=config` to `play_game`; per-game line prints `winners={list(result.winners)}`; tally `wins = [0] * len(agents)`, `draws` counts `len(result.winners) > 1`, sole winners increment `wins[result.winners[0]]`; summary:

```python
    print(
        "summary: "
        + " ".join(f"p{i}[{a.name}] wins={wins[i]}" for i, a in enumerate(agents))
        + f" shared={draws}"
    )
```

(Existing tests assert `"summary:" in out` and `"game 0: seed=7"` — both preserved. `_evaluate` is untouched beyond Task 3's `winner_2p` fix.) Import `GameConfig` from `carcassonne.core`.

**Step 4: Verify**
Run: `uv run pytest tests/cli -q`
Expected: PASS (old `--p0/--p1` tests included).

**Step 5: Commit**
```bash
git add carcassonne/cli/main.py tests/cli/test_simulate.py
git commit -m "feat(cli): simulate --agents for 2-5 player games"
```

---

### Task 7: Web backend — N-seat sessions and API
**Depends on:** Task 2 (parallel with 3-6; needs replay v2 writer)

**Files:**
- Modify: `carcassonne/web/sessions.py`, `carcassonne/web/app.py`, `carcassonne/web/views.py`
- Test: `tests/web/test_api.py`

**Step 1: Failing tests** — append to `tests/web/test_api.py`:

```python
def test_three_player_game_end_to_end(client: TestClient, replays_dir: Path) -> None:
    r = client.post(
        "/api/games",
        json={"opponents": ["random", "random"], "num_players": 3, "human_player": 1, "seed": 7},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["seats"] == ["random", "human", "random"]
    # AI seat 0 has already moved; it is now the human's turn (seat 1)
    assert body["state"]["current_player"] == 1
    assert [m["seat"] for m in body["ai_moves"]] == [0]
    game_id = body["game_id"]
    state = body["state"]
    while not state["terminal"]:
        moved = client.post(f"/api/games/{game_id}/move", json={"idx": 0})
        assert moved.status_code == 200
        state = moved.json()["state"]
    assert len(state["final_scores"]) == 3
    (path,) = replays_dir.glob("*.jsonl")
    replay = load_replay(path)
    assert replay.header.config.num_players == 3
    list(replay_states(replay))


def test_checkpoint_opponents_rejected_beyond_two_players(client: TestClient) -> None:
    r = client.post(
        "/api/games",
        json={"opponents": ["random", "ckpt:step_000100"], "num_players": 3},
    )
    assert r.status_code == 422


def test_legacy_two_player_body_still_works(client: TestClient) -> None:
    r = client.post("/api/games", json={"opponent": "random", "human_player": 0, "seed": 3})
    assert r.status_code == 200
    assert r.json()["seats"] == ["human", "random"]
```

**Step 2: Verify fail**
Run: `uv run pytest tests/web/test_api.py -q`
Expected: new tests FAIL (422 on unknown `opponents` field / missing keys).

**Step 3: Implement**

`carcassonne/web/sessions.py`:
- `Session` fields: replace `opponent: Agent` with `agents: dict[int, Agent]` (seat → agent, human seat absent) and add `seat_names: tuple[str, ...]`.
- Replace `ai_step` with:

```python
    def run_ai_turns(self) -> list[tuple[int, Move, Annot | None, tuple[ScoreEvent, ...]]]:
        """Step every consecutive AI seat until it's the human's turn or the game ends."""
        out: list[tuple[int, Move, Annot | None, tuple[ScoreEvent, ...]]] = []
        while not self.terminal and self.state.current_player != self.human_player:
            seat = self.state.current_player
            move, annot = self.agents[seat].choose(self.state, TurnContext(self.rng, annotate=True))
            out.append((seat, move, annot, self._apply(move, annot)))
        return out
```

- `hint_annot`: `if len(self.agents) != 1: return None` first line (hints are a 2-player feature), else unchanged via `next(iter(self.agents.values()))`.
- `SessionStore.create(self, opponent_specs: list[str], human_player: int, seed: int) -> Session`:

```python
        num_players = len(opponent_specs) + 1
        if num_players > 2:
            for spec in opponent_specs:
                if spec.startswith("ckpt:"):
                    raise UnknownOpponent(
                        f"checkpoint agents are 2-player only; cannot seat {spec!r}"
                        f" in a {num_players}-player game"
                    )
        try:
            opponents = [make_agent(s) for s in opponent_specs]
        except ValueError as e:
            raise UnknownOpponent(str(e)) from e
        config = GameConfig(num_players=num_players)
        state = new_game(seed, config)
        ai_seats = [s for s in range(num_players) if s != human_player]
        agents = dict(zip(ai_seats, opponents, strict=True))
        names_list = ["human"] * num_players
        for seat, agent in agents.items():
            names_list[seat] = agent.name
        names = tuple(names_list)
```

  Header uses `agents=names` and `config=config`; `Session(...)` gets `agents=agents, seat_names=names`. `_unique_replay_path` takes `names: tuple[str, ...]` and joins with `"-vs-"`.

`carcassonne/web/app.py`:

```python
class CreateGameBody(BaseModel):
    opponent: str | None = None          # legacy 2-player shorthand
    opponents: list[str] | None = None   # AI seats in seat order (human seat skipped)
    num_players: int = Field(default=2, ge=2, le=5)
    human_player: int = Field(default=0, ge=0, le=4)
    seed: int | None = None
```

(`from pydantic import BaseModel, Field`.) `create_game`:

```python
    def create_game(body: CreateGameBody) -> dict[str, Any]:
        if (body.opponent is None) == (body.opponents is None):
            raise UnknownOpponent("pass exactly one of 'opponent' (2-player) or 'opponents'")
        specs = body.opponents if body.opponents is not None else [body.opponent]
        n = len(specs) + 1
        if body.opponents is not None and n != body.num_players:
            raise UnknownOpponent(f"{len(specs)} opponents implies {n} players, not {body.num_players}")
        if not body.human_player < n:
            raise UnknownOpponent(f"human_player {body.human_player} out of range for {n} players")
        seed = body.seed if body.seed is not None else random.randrange(2**31)
        session = store.create(specs, body.human_player, seed)
        ai_moves = [views.ai_move_view(seat, m, a) for seat, m, a, _ in session.run_ai_turns()]
        return {
            "game_id": session.id,
            "seats": list(session.seat_names),
            "ai_moves": ai_moves,
            "state": views.state_view(session.state),
        }
```

`post_move`: replace the single `ai_step` block with `run_ai_turns()`, extending `events` from each step and returning `"ai_moves": [...]` (list) instead of `"ai_move"`. `get_game` response gains `"seats": list(session.seat_names)`.

Replay endpoints (winners rename from Task 2): in `get_replay` return `"winners": list(replay.winners)`; in `_list_replays`:

```python
        winners = end.get("winners")
        if winners is None:  # v1 file
            w = end.get("winner")
            winners = [0, 1] if w is None else [w]
```

and emit `"winners": winners`. `views.py`: `ai_move_view(seat: int, move: Move, annot: Annot | None)` adds `"seat": seat` to the dict; `replay_header_view` config dict gains `"num_players": header.config.num_players`.

**Step 4: Verify**
Run: `uv run pytest tests/web -q && uv run mypy carcassonne`
Expected: PASS. Existing tests referencing `ai_move` in responses must be updated to `ai_moves` (grep `tests/web` for `ai_move`).

**Step 5: Commit**
```bash
git add carcassonne/web tests/web
git commit -m "feat(web): N-seat sessions, multi-opponent game creation, winners in replay API"
```

---

### Task 8: Frontend — Play view for 2-5 players
**Depends on:** Task 7

**Files:**
- Modify: `carcassonne/web/static/tiles.js`, `style.css`, `index.html`, `app.js`

No JS test harness exists; the backend contract is covered by Task 7's tests. Verification is a manual checklist (Step 3) — run `uv run carcassonne serve` and check in the browser.

**Step 1: Palette + shared colour lookup**

`tiles.js` TOKENS: add after `p1`:

```js
  p2: "#8B6BC9",
  p3: "#C965A8",
  p4: "#E0B54B",
```

(Chosen to stay clear of field green/city gold/AI teal; tune by eye against the board.) Replace the local ternary (line 45) with an exported lookup and use it everywhere `p === 0 ?` ternaries pick colours:

```js
export const playerColor = (p) => [TOKENS.p0, TOKENS.p1, TOKENS.p2, TOKENS.p3, TOKENS.p4][p] ?? TOKENS.ink;
```

`style.css`: add `--p2: #8b6bc9; --p3: #c965a8; --p4: #e0b54b;` beside `--p0/--p1` (line 17) and `.swatch.p2/.p3/.p4` classes beside the existing two (line 845).

**Step 2: Dynamic score cards + dialog**

`index.html`: replace the two static `score-card` divs (lines 38-47) with `<div id="cards"></div>`. In the dialog, replace the single-opponent row with:

```html
      <div class="row">
        <label for="opt-players">Players</label>
        <select id="opt-players">
          <option value="2" selected>2</option><option value="3">3</option>
          <option value="4">4</option><option value="5">5</option>
        </select>
      </div>
      <div id="opt-opponents"></div>
```

`app.js`:
- `newGame(opponents, seat, seed)` posts `{opponents, num_players: opponents.length + 1, human_player: seat, seed}`; store `S.seats = body.seats` from the response; apply `body.ai_moves` (list) where the old code consumed the single opening `ai_move`.
- Dialog start handler: build `opponents` from the visible selects; the `#opt-players` change handler renders `n-1` opponent `<select>`s (options: Greedy/Random, plus checkpoints only when `n === 2` — reuse the existing checkpoint-population code) and repopulates `#opt-seat` with seats `0..n-1` labelled by turn order.
- `updateSidebar` (line 591): `for (let seat = 0; seat < scores.length; seat++)`, creating cards into `#cards` on first render: edge colour `playerColor(seat)` (import from tiles.js), `.who` text `seat === S.humanPlayer ? "You" : S.seats[seat]`.
- Replace the `1 - S.humanPlayer` idioms: banner final-scores line (211-212) lists every seat (`You 34 · random 28 · greedy 41`, winners bolded via `state.final_scores` argmax set); move rings (296-297) become "ring the human's last move + the most recent AI move", tracking `S.lastAiSeat` when consuming `ai_moves` and colouring with `playerColor(S.lastAiSeat)`.
- Score-log rows (647-665): tint by `playerColor(e.player)` and label with seat name instead of the mine/AI binary.
- Hint row (`#hint-row`, `#value-readout`): hide via `hidden = scores.length > 2` on game start (server returns no hint annot for n>2 anyway).
- `drawMeepleGlyph(...)` calls that pass a seat already generalise through `playerColor`.

**Step 3: Verify (manual checklist — record results in the commit/PR)**
Run: `uv run pytest tests/web -q` (still green), then `uv run carcassonne serve` and check:
- [ ] 2-player game vs greedy: identical to before (cards, hint toggle visible, colours).
- [ ] New game with Players=4, three opponents: 4 colour-coded cards, correct names; consecutive AI seats play automatically after your move; log rows tinted per seat.
- [ ] Human seat 2 of 3: AIs open the game up to your turn.
- [ ] Finished 4-player game: banner lists all four finals; replay saved.

**Step 4: Commit**
```bash
git add carcassonne/web/static
git commit -m "feat(web): play view for 2-5 players - palette, dynamic cards, multi-AI turns"
```

---

### Task 9: Frontend — Replay view for N players
**Depends on:** Task 7 (palette from Task 8)

**Files:**
- Modify: `carcassonne/web/static/replay.js`, `replay.html`

**Step 1: Implement**

`replay.js`:
- Delete the local `playerColor` (line 78); import it from `./tiles.js`.
- Derive `const NP = R.data.header.config.num_players ?? R.data.final_scores.length;` once per loaded replay; replace every `[0, 1]` iteration (lines 324, 771, 989) with `Array.from({length: NP}, (_, p) => p)` loops.
- Replay list row (line 98): winners-based — `m.winners.length === NP ? "draw" : m.winners.length === 1 ? \`P${m.winners[0]} won\` : \`tie: ${m.winners.map((w) => "P" + w).join(" & ")}\``.
- Score panel: keep `#score-readout` for 2 players; for `NP > 2` render one `P{i} n` chip per seat (same swatch+number pattern as line 989's block); the score-over-time chart already loops seats once the `[0, 1]` is generalised; axis text lists seat colours generically.
- Win-probability + search panels are 2-player features: `if (NP !== 2) { $("winprob-panel").hidden = true; }` and skip `buildWinProb`/lead-change markers; the P0-perspective transform (lines 352, 621) is only reached when `NP === 2`. Search panel (prior-vs-visits) renders only when a move has `annot` — unchanged, since 3-5 player replays carry none.
- Per-move score-delta chips (line 324 block) and meeple/tile highlights already colour via `playerColor(...)` once generalised.

`replay.html`: no structural change needed beyond tolerating a hidden winprob panel (already `hidden` by default, line 93).

**Step 2: Verify (manual + API)**
- `uv run carcassonne simulate --agents greedy,random,random,random,random --seed 3 --out replays-tmp` then `uv run carcassonne serve --replays replays-tmp`:
- [ ] The 5-player replay lists with a correct winners label and opens.
- [ ] Stepping shows 5 seat colours, per-seat score chips, no win-prob panel, no crash in the console.
- [ ] A pre-existing 2-player annotated replay still shows search panel + win-prob chart.

**Step 3: Commit**
```bash
git add carcassonne/web/static
git commit -m "feat(web): replay view renders 2-5 player games; RL panels stay 2-player"
```

---

### Task 10: End-to-end verification (spec success bar)
**Depends on:** Tasks 1-9

**Files:**
- Test: `tests/web/test_api.py` (one addition), no source changes expected

**Step 1: Failing-first e2e test** — spec bar item 2 as code (5-player CLI replay served through the API):

```python
def test_five_player_cli_replay_serves_through_api(tmp_path: Path) -> None:
    from carcassonne.cli.main import main as cli_main

    replays = tmp_path / "replays"
    rc = cli_main(
        ["simulate", "--agents", "greedy,random,random,random,random",
         "--seed", "13", "--out", str(replays)]
    )
    assert rc == 0
    client = TestClient(create_app(replays))
    listed = client.get("/api/replays").json()["replays"]
    assert len(listed) == 1 and len(listed[0]["final_scores"]) == 5
    detail = client.get(f"/api/replays/{listed[0]['name']}").json()
    assert detail["header"]["config"]["num_players"] == 5
    assert len(detail["states"]) == len(detail["moves"]) + 1
```

**Step 2: Full gate** (all must pass, none skipped):
```bash
uv run pytest -q
uv run ruff check .
uv run mypy carcassonne
```
Expected: entire suite green (~174 pre-existing + new), zero lint/type errors.

**Step 3: Runtime bar** (manual, from spec success bar):
- [ ] Browser: one human vs 3 agents end-to-end, per-seat scoring correct, replay saved and reviewable (spec bar 3).
- [ ] 2-player checkpoint agent (if a trained checkpoint exists locally) loads and plays unchanged (spec bar 4).

**Step 4: Commit**
```bash
git add tests/web/test_api.py
git commit -m "test(e2e): 5-player CLI replay served through the web API"
```

---

## Parallelism

- **Task 1** first (everything depends on it).
- Then in parallel: **Task 2** (→ 3 → 6), **Task 4**, **Task 5**.
- **Task 7** after Task 2 (parallel with 3-6 once replay v2 exists; its checkpoint-rejection test needs nothing from 3-6).
- **Task 8** after 7; **Task 9** after 7 (8 and 9 parallel, both touch `static/` — serialise commits).
- **Task 10** last.

## Review
- [ ] Code review requested (`request-review` skill, five-axis)
- [ ] All feedback addressed
- [ ] Final verification passed (`verify` skill: full suite + manual browser checklist)
- [ ] `docs/features.md` planned items flipped to implemented; spec success bar checked off
