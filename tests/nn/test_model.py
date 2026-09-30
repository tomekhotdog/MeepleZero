from __future__ import annotations

import itertools
import random

import numpy as np
import torch

from carcassonne.core import GameConfig, apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.nn.actions import (
    ACTIONS,
    ACTIONS_PER_CELL,
    ROTS,
    Crop,
    encode_move,
    flat_index,
    move_key,
)
from carcassonne.nn.encode import NUM_PLANES, encode_state, valid_mask
from carcassonne.nn.model import (
    CarcassonneNet,
    _policy_head_flatten,
    evaluate_state,
    param_count,
)


def _mid_state(seed: int = 5, n_moves: int = 10) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed, GameConfig())
    for _ in range(n_moves):
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def _tiny_net() -> CarcassonneNet:
    torch.manual_seed(0)
    return CarcassonneNet(channels=16, n_blocks=2).eval()


def test_param_budget() -> None:
    assert param_count(CarcassonneNet()) <= 250_000  # design bar; expect ~192k


def test_policy_flatten_matches_action_indexer_exactly() -> None:
    """THE alignment proof for the most bug-prone reshape in the project.

    Fill the conv output with unique values so every (channel, gy, gx) slot is
    distinguishable, then check that the value at every flat index is exactly
    the one from cell (gy, gx), channel rot*14+a — where the flat index comes
    from the ActionIndexer's own `flat_index`, not a re-derived formula. Any
    permutation error (swapped rot/a split, transposed gy/gx) fails loudly.
    """
    h, w = 3, 5  # non-square so transposed axes cannot alias
    t = torch.arange(ACTIONS_PER_CELL * h * w, dtype=torch.float32).reshape(
        1, ACTIONS_PER_CELL, h, w
    )
    flat = _policy_head_flatten(t)
    assert flat.shape == (1, h * w * ACTIONS_PER_CELL)
    crop = Crop(0, 0, h, w)  # world == grid coordinates
    for gy, gx, rot, a in itertools.product(range(h), range(w), range(ROTS), range(ACTIONS)):
        idx = flat_index(crop, (gx, gy, rot, a))
        assert flat[0, idx] == t[0, rot * ACTIONS + a, gy, gx]


def test_policy_head_matches_action_indexer() -> None:
    """End-to-end: masked softmax puts nonzero mass on exactly the legal set."""
    from carcassonne.nn.actions import legal_mask

    net = _tiny_net()
    state = _mid_state()
    crop = Crop.of(state)
    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(state)).unsqueeze(0)
    logits, _ = net(x, mask=mask)
    assert logits.shape == (1, crop.n_actions)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    legal = {encode_move(state, m) for m in legal_moves(state)}
    # Exact partition: with a tiny random net logits are small, so no legal
    # move's probability can underflow to zero — equality is safe here.
    assert {int(i) for i in torch.nonzero(probs).flatten()} == legal


def test_forward_accepts_any_size() -> None:
    net = _tiny_net()
    for h, w in ((3, 3), (5, 9), (12, 7)):
        x = torch.randn(2, NUM_PLANES, h, w)
        logits, value = net(x)
        assert logits.shape == (2, h * w * 56)
        assert value.shape == (2,)
        assert (value.abs() <= 1).all()


def test_padded_forward_equals_unpadded() -> None:
    """THE invariant: padding + cell_mask changes nothing for the real cells."""
    net = _tiny_net()
    state = _mid_state()
    crop = Crop.of(state)
    ph, pw = crop.h + 4, crop.w + 3

    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    logits, value = net(x)

    xp = torch.from_numpy(encode_state(state, pad_hw=(ph, pw))).unsqueeze(0)
    cm = torch.from_numpy(valid_mask(crop, pad_hw=(ph, pw))).unsqueeze(0)
    logits_p, value_p = net(xp, cell_mask=cm)

    # Compare per real cell: reshape to (gy, gx, 56) grids.
    grid = logits.reshape(crop.h, crop.w, 56)
    grid_p = logits_p.reshape(ph, pw, 56)[: crop.h, : crop.w]
    assert torch.allclose(grid, grid_p, atol=1e-5)
    assert torch.allclose(value, value_p, atol=1e-5)


def test_mask_sets_illegal_to_neg_inf() -> None:
    from carcassonne.nn.actions import legal_mask

    net = _tiny_net()
    state = new_game(0, GameConfig())
    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(state)).unsqueeze(0)
    logits, _ = net(x, mask=mask)
    probs = torch.softmax(logits, dim=1).squeeze(0)
    assert float(probs[~mask.squeeze(0)].sum()) == 0.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def _play(seed: int, n_moves: int) -> GameState:
    """Random playout of up to ``n_moves``, stopping short of a terminal state."""
    rng = random.Random(seed)
    state = new_game(seed, GameConfig())
    for _ in range(n_moves):
        nxt = apply(state, rng.choice(legal_moves(state)))
        if is_terminal(nxt):
            break
        state = nxt
    return state


def _reference_priors(net: CarcassonneNet, state: GameState) -> tuple[np.ndarray, float]:
    """The pre-refactor path, spelled out: full-crop masked softmax, then gather
    each legal move's probability via flat_index. evaluate_state must match it
    bit-for-bit -- the aligned return is a layout change, not a numerics change."""
    from carcassonne.nn.actions import legal_mask

    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    mask = torch.from_numpy(legal_mask(state)).unsqueeze(0)
    with torch.no_grad():
        logits, value = net(x, mask)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy().astype(np.float32)
    crop = Crop.of(state)
    aligned = np.array([probs[flat_index(crop, move_key(m))] for m in legal_moves(state)])
    return aligned, float(value.item())


def test_evaluate_state_priors_align_with_callers_moves_exactly() -> None:
    """Priors come back indexed like the caller's ``moves`` (MCTS's node arrays)
    and equal the old full-crop-then-gather path exactly, on empty, mid and late
    boards -- so the policy/Crop layout contract lives only inside nn."""
    net = _tiny_net()
    for state in (new_game(0, GameConfig()), _play(5, 10), _play(7, 60)):
        moves = legal_moves(state)
        priors, value = evaluate_state(net, state, torch.device("cpu"), moves)
        ref_priors, ref_value = _reference_priors(net, state)
        assert priors.shape == (len(moves),)
        assert priors.dtype == np.float32
        assert np.array_equal(priors, ref_priors)
        assert value == ref_value
        assert abs(float(priors.sum()) - 1.0) < 1e-4
        assert -1.0 <= value <= 1.0


def test_evaluate_state_follows_callers_move_order() -> None:
    """Alignment is to the *given* sequence, not to legal_moves' own order."""
    net = _tiny_net()
    state = _mid_state()
    moves = legal_moves(state)
    fwd, _ = evaluate_state(net, state, torch.device("cpu"), moves)
    rev, _ = evaluate_state(net, state, torch.device("cpu"), moves[::-1])
    assert np.array_equal(fwd, rev[::-1])
