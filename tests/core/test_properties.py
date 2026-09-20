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
