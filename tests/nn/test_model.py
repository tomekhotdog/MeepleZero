from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch

from carcassonne.core import apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.nn.actions import ACTION_SPACE, WINDOW, encode_move, legal_mask
from carcassonne.nn.checkpoint import latest, load, pick_device, save
from carcassonne.nn.encode import NUM_PLANES, encode_state
from carcassonne.nn.model import CarcassonneNet, evaluate_state, param_count

SEED = 0


def _net() -> CarcassonneNet:
    torch.manual_seed(SEED)
    return CarcassonneNet()


def _mid_game(seed: int, moves: int) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed)
    for _ in range(moves):
        if is_terminal(state):
            break
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def _batch(states: list[GameState]) -> torch.Tensor:
    return torch.from_numpy(np.stack([encode_state(s) for s in states]))


# ---------------------------------------------------------------------------
# 1. forward shape + param count


def test_forward_shape_and_param_count() -> None:
    net = _net().eval()
    x = torch.randn(4, NUM_PLANES, WINDOW, WINDOW)
    with torch.no_grad():
        policy_logits, value = net(x)
    assert policy_logits.shape == (4, ACTION_SPACE)
    assert value.shape == (4,)
    assert torch.all(value >= -1.0) and torch.all(value <= 1.0)

    n = param_count(net)
    print(f"CarcassonneNet param count: {n:,}")
    assert n < 3_000_000  # Pi-sized sanity


# ---------------------------------------------------------------------------
# 2. masking: illegal logits are -inf, softmax gives exactly 0 and sums to 1


def test_masking_zeros_illegal_probs() -> None:
    net = _net().eval()
    state = _mid_game(seed=1, moves=12)
    mask_np = legal_mask(state)
    x = _batch([state, state])
    mask = torch.from_numpy(np.stack([mask_np, mask_np]))

    with torch.no_grad():
        policy_logits, _ = net(x, mask)

    illegal = ~torch.from_numpy(mask_np)
    assert torch.all(torch.isinf(policy_logits[0][illegal]))
    assert torch.all(policy_logits[0][illegal] < 0)

    probs = torch.softmax(policy_logits, dim=1)
    assert torch.all(probs[0][illegal] == 0.0)
    assert torch.isclose(probs[0].sum(), torch.tensor(1.0), atol=1e-6)


# ---------------------------------------------------------------------------
# 3. THE ALIGNMENT TEST: policy head order == ActionIndexer layout


def test_policy_head_matches_action_indexer() -> None:
    net = _net()
    state = _mid_game(seed=2, moves=15)

    legal_ids = {encode_move(state, m) for m in legal_moves(state)}
    mask_np = legal_mask(state)
    # legal_mask and encode_move must already agree (sanity for the test itself).
    assert set(np.flatnonzero(mask_np).tolist()) == legal_ids

    probs, value = evaluate_state(net, state, torch.device("cpu"))
    assert probs.shape == (ACTION_SPACE,)
    assert -1.0 <= value <= 1.0

    nonzero = set(np.flatnonzero(probs > 0.0).tolist())
    # The exact set of actions the net assigns probability to is the legal set:
    # this can only hold if head channel/spatial order matches the action id.
    assert nonzero == legal_ids

    # Positionally: probs[encode_move(state, m)] > 0 for every legal move, and
    # every illegal id is exactly 0.
    for m in legal_moves(state):
        assert probs[encode_move(state, m)] > 0.0
    for i in range(ACTION_SPACE):
        if i not in legal_ids:
            assert probs[i] == 0.0


# ---------------------------------------------------------------------------
# 4. save/load round-trip


def test_save_load_round_trip(tmp_path: Path) -> None:
    net = _net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    device = torch.device("cpu")

    x = torch.randn(2, NUM_PLANES, WINDOW, WINDOW)
    net.eval()
    with torch.no_grad():
        ref_policy, ref_value = net(x)

    path = save(tmp_path, step=7, net=net, optimizer=optimizer, config={"lr": 1e-3})
    assert path.name == "step_000007.pt"

    # Reconstruct from net_arch alone (net not passed).
    payload = load(path, device)
    fresh = payload["net"]
    assert fresh is not net
    fresh.eval()
    with torch.no_grad():
        out_policy, out_value = fresh(x)
    assert torch.equal(out_policy, ref_policy)
    assert torch.equal(out_value, ref_value)
    assert payload["step"] == 7
    assert payload["config"] == {"lr": 1e-3}

    # Optimizer state restored into a matching optimizer.
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-3)
    load(path, device, net=fresh, optimizer=fresh_opt)
    assert fresh_opt.state_dict()["param_groups"][0]["lr"] == 1e-3

    # latest() picks the highest step.
    save(tmp_path, step=3, net=net)
    save(tmp_path, step=42, net=net)
    top = latest(tmp_path)
    assert top is not None and top.name == "step_000042.pt"


def test_latest_empty_dir(tmp_path: Path) -> None:
    empty = tmp_path / "nada"
    empty.mkdir()
    assert latest(empty) is None


# ---------------------------------------------------------------------------
# 5. determinism


def test_eval_is_deterministic() -> None:
    net = _net().eval()
    x = torch.randn(3, NUM_PLANES, WINDOW, WINDOW)
    with torch.no_grad():
        p1, v1 = net(x)
        p2, v2 = net(x)
    assert torch.equal(p1, p2)
    assert torch.equal(v1, v2)


# ---------------------------------------------------------------------------
# 6. runs on pick_device()


def test_runs_on_pick_device() -> None:
    device = pick_device()
    net = _net().to(device).eval()
    state = _mid_game(seed=4, moves=8)
    probs, value = evaluate_state(net, state, device)
    assert probs.shape == (ACTION_SPACE,)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
