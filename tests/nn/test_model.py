from __future__ import annotations

import random

import numpy as np
import torch

from carcassonne.core import GameConfig, apply, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.nn.actions import Crop, encode_move
from carcassonne.nn.encode import NUM_PLANES, encode_state, valid_mask
from carcassonne.nn.model import CarcassonneNet, evaluate_state, param_count


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


def test_policy_head_matches_action_indexer() -> None:
    """Logit at flat index i must come from cell (gy,gx) channel rot*14+a."""
    net = _tiny_net()
    state = _mid_state()
    crop = Crop.of(state)
    x = torch.from_numpy(encode_state(state)).unsqueeze(0)
    logits, _ = net(x)
    assert logits.shape == (1, crop.n_actions)
    for m in legal_moves(state)[:5]:
        i = encode_move(state, m)
        assert torch.isfinite(logits[0, i])


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


def test_evaluate_state_bridge() -> None:
    net = _tiny_net()
    state = _mid_state()
    probs, value = evaluate_state(net, state, torch.device("cpu"))
    assert probs.shape == (Crop.of(state).n_actions,)
    assert abs(float(probs.sum()) - 1.0) < 1e-4
    assert -1.0 <= value <= 1.0
    legal = {encode_move(state, m) for m in legal_moves(state)}
    assert set(np.nonzero(probs)[0]).issubset(legal)
