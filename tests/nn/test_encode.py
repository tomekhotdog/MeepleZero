from __future__ import annotations

import random

import numpy as np
import pytest

from carcassonne.core import GameConfig, RulesError, apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.core.types import PlaceMeeple
from carcassonne.nn.actions import Crop
from carcassonne.nn.encode import (
    _CUR0,
    _EDGE0,
    NUM_PLANES,
    PLANES,
    encode_state,
    valid_mask,
)
from carcassonne.training.learner import rotate_state_90

# ---------------------------------------------------------------------------
# helpers


def _mid_state(seed: int = 7, n_moves: int = 12) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed, GameConfig())
    for _ in range(n_moves):
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def _play_to_end(seed: int) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed)
    while not is_terminal(state):
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def _first_state_with_unequal_scores(seed: int) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed)
    while not is_terminal(state):
        if state.scores[0] != state.scores[1]:
            return state
        state = apply(state, rng.choice(legal_moves(state)))
    raise AssertionError("no state with unequal scores found")


def _meeple_biased_game(seed: int, moves: int) -> GameState:
    """Play `moves` random moves, preferring ones that place a meeple."""
    rng = random.Random(seed)
    state = new_game(seed)
    for _ in range(moves):
        if is_terminal(state):
            break
        legal = legal_moves(state)
        preferred = [m for m in legal if isinstance(m.action, PlaceMeeple)]
        state = apply(state, rng.choice(preferred or list(legal)))
    return state


# ---------------------------------------------------------------------------
# Shapes, manifest, padding


def test_shapes_track_crop() -> None:
    state = new_game(0, GameConfig())
    crop = Crop.of(state)
    planes = encode_state(state)
    assert planes.shape == (NUM_PLANES, crop.h, crop.w) == (52, 3, 3)
    assert planes.dtype == np.float32
    assert len(PLANES) == NUM_PLANES == 52
    assert len(set(PLANES)) == 52  # no duplicate names
    mid = _mid_state()
    c2 = Crop.of(mid)
    assert encode_state(mid).shape == (52, c2.h, c2.w)


def test_padded_encoding_zero_outside_crop() -> None:
    state = _mid_state()
    crop = Crop.of(state)
    ph, pw = crop.h + 3, crop.w + 2
    planes = encode_state(state, pad_hw=(ph, pw))
    assert planes.shape == (NUM_PLANES, ph, pw)
    assert np.array_equal(planes[:, : crop.h, : crop.w], encode_state(state))
    assert not planes[:, crop.h :, :].any()
    assert not planes[:, :, crop.w :].any()
    vm = valid_mask(crop, pad_hw=(ph, pw))
    assert vm.shape == (ph, pw)
    assert vm[: crop.h, : crop.w].all() and vm.sum() == crop.h * crop.w


def test_broadcast_planes_cover_exactly_the_crop() -> None:
    state = _mid_state()
    planes = encode_state(state)
    turn_plane = planes[PLANES.index("s_turn")]
    assert (turn_plane == np.float32(state.turn / 72.0)).all()  # constant across the whole crop


# ---------------------------------------------------------------------------
# Content: perspective, current-tile planes, FeaturePlanes


def test_perspective_swaps_on_apply() -> None:
    state = _first_state_with_unequal_scores(seed=3)
    me = state.current_player
    planes = encode_state(state)
    assert planes[PLANES.index("s_my_score")][0, 0] == np.float32(state.scores[me] / 50.0)
    assert planes[PLANES.index("s_opp_score")][0, 0] == np.float32(state.scores[1 - me] / 50.0)

    nxt = apply(state, legal_moves(state)[0])
    assert nxt.current_player == 1 - me  # perspective flips
    planes2 = encode_state(nxt)
    assert planes2[PLANES.index("s_my_score")][0, 0] == np.float32(
        nxt.scores[nxt.current_player] / 50.0
    )
    assert planes2[PLANES.index("s_opp_score")][0, 0] == np.float32(
        nxt.scores[1 - nxt.current_player] / 50.0
    )


def test_current_tile_planes_broadcast_structural_edges() -> None:
    state = new_game(1)
    for _ in range(6):
        planes = encode_state(state)
        # Per world side exactly one of {city, road, field} is set, all-ones.
        cur_edges = planes[_CUR0 : _CUR0 + 12]
        for si in range(4):
            side = cur_edges[si * 3 : si * 3 + 3]
            assert (side.sum(axis=0) == 1.0).all(), f"side {si} not one-hot"
        state = apply(state, legal_moves(state)[0])

    terminal = _play_to_end(2)
    assert is_terminal(terminal)
    planes = encode_state(terminal)
    assert not planes[_CUR0 : _CUR0 + 15].any()  # no current tile when terminal


def test_featureplanes_report_connected_feature() -> None:
    """Property: a cell has fp_city_size > 0 iff it is occupied by a tile with a
    city edge (every city feature touches >= 1 edge and has size >= 1)."""
    mid = _mid_state()
    planes = encode_state(mid)
    occ = planes[PLANES.index("occupied")].astype(bool)
    fp_size = planes[PLANES.index("fp_city_size")]
    city_edge = planes[_EDGE0 : _EDGE0 + 12 : 3].sum(axis=0) > 0  # any city edge
    assert ((fp_size > 0) == (occ & city_edge)).all()


def test_featureplanes_meeples_are_perspective_correct() -> None:
    state = _meeple_biased_game(seed=5, moves=22)
    me = state.current_player
    planes = encode_state(state)
    fp_mine = planes[PLANES.index("fp_city_mine")] + planes[PLANES.index("fp_road_mine")]
    fp_theirs = planes[PLANES.index("fp_city_theirs")] + planes[PLANES.index("fp_road_theirs")]
    my_on_board = sum(
        1
        for d in state.features.data.values()
        for m in d.meeples
        if m.player == me and d.kind.value in ("city", "road")
    )
    opp_on_board = sum(
        1
        for d in state.features.data.values()
        for m in d.meeples
        if m.player != me and d.kind.value in ("city", "road")
    )
    assert (fp_mine.sum() > 0) == (my_on_board > 0)
    assert (fp_theirs.sum() > 0) == (opp_on_board > 0)
    assert my_on_board + opp_on_board > 0  # the planes are genuinely exercised


# ---------------------------------------------------------------------------
# D4 equivariance: a genuine 90-degree board rotation == np.rot90 of the
# planes, with the 12 board-edge planes side-permuted (N<-W, E<-N, S<-E, W<-S).
# Current-tile planes and scalars are broadcast constants, so np.rot90 leaves
# them unchanged -- correctly so: the tile in hand does not rotate with the
# board, and scalars are rotation-invariant.


def test_d4_rotation_equivariance() -> None:
    state = _meeple_biased_game(seed=5, moves=22)
    planes = encode_state(state)
    # Sanity: the state we test is non-trivial (real content to transform).
    assert int(planes[PLANES.index("occupied")].sum()) == len(state.board)
    meeple_total = int(
        planes[PLANES.index("meeple_me")].sum() + planes[PLANES.index("meeple_opp")].sum()
    )
    assert meeple_total > 0  # meeple planes are genuinely exercised

    rplanes = encode_state(rotate_state_90(state))
    # World (x,y)->(y,-x) with crop re-anchoring maps grid cell (gy, gx) to
    # (gy', gx') = ((w-1) - gx, gy), which is exactly np.rot90 with k=+1 on the
    # (gy, gx) axes: rot90(A, 1)[i, j] == A[j, (w-1) - i].
    expected = np.rot90(planes, k=1, axes=(1, 2)).copy()
    # Permute the 12 edge planes: new world side N was old W, E was N, S was E, W was S.
    perm = [3, 0, 1, 2]  # index by new side -> old side (N,E,S,W = 0,1,2,3)
    edges = expected[_EDGE0 : _EDGE0 + 12].copy()
    for new_s in range(4):
        expected[_EDGE0 + new_s * 3 : _EDGE0 + new_s * 3 + 3] = edges[
            perm[new_s] * 3 : perm[new_s] * 3 + 3
        ]
    assert np.allclose(rplanes, expected)


# ---------------------------------------------------------------------------
# Guards


def test_encode_rejects_non_two_player_state() -> None:
    with pytest.raises(RulesError, match="2-player"):
        encode_state(new_game(0, GameConfig(num_players=3)))


def test_encode_rejects_padding_smaller_than_crop() -> None:
    state = _mid_state()
    crop = Crop.of(state)
    with pytest.raises(RulesError, match="padding"):
        encode_state(state, pad_hw=(crop.h - 1, crop.w))
