"""Checkpoint I/O: save/load round trip, latest(), pick_device.

Recovered from the pre-rewrite tests/nn/test_model.py (a55b4f8), adapted to
the size-agnostic net: arbitrary (h, w) inputs instead of the fixed WINDOW,
per-state action counts instead of ACTION_SPACE, tiny nets for speed.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from carcassonne.core import apply, is_terminal, legal_moves, new_game
from carcassonne.core.engine import GameState
from carcassonne.nn.checkpoint import CHECKPOINT_FORMAT, latest, load, pick_device, save
from carcassonne.nn.encode import NUM_PLANES
from carcassonne.nn.model import CarcassonneNet, evaluate_state

SEED = 0


def _net() -> CarcassonneNet:
    torch.manual_seed(SEED)
    return CarcassonneNet(channels=16, n_blocks=1)


def _mid_game(seed: int, moves: int) -> GameState:
    rng = random.Random(seed)
    state = new_game(seed)
    for _ in range(moves):
        if is_terminal(state):
            break
        state = apply(state, rng.choice(legal_moves(state)))
    return state


def test_save_load_round_trip(tmp_path: Path) -> None:
    net = _net()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    device = torch.device("cpu")

    x = torch.randn(2, NUM_PLANES, 5, 7)  # any size: the net is size-agnostic
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


def test_eval_is_deterministic() -> None:
    net = _net().eval()
    x = torch.randn(3, NUM_PLANES, 5, 7)
    with torch.no_grad():
        p1, v1 = net(x)
        p2, v2 = net(x)
    assert torch.equal(p1, p2)
    assert torch.equal(v1, v2)


def test_runs_on_pick_device() -> None:
    device = pick_device()
    net = _net().to(device).eval()
    state = _mid_game(seed=4, moves=8)
    moves = legal_moves(state)
    probs, value = evaluate_state(net, state, device, moves)
    assert probs.shape == (len(moves),)
    assert abs(probs.sum() - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0


def _legacy_checkpoint(directory: Path, fmt: int | None = None) -> Path:
    """A pre-iteration-3-shaped payload: 31x31-era keys the dynamic-crop net
    doesn't have, and no (or a foreign) format marker."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "step": 3,
        "model_state": {"policy_fc.weight": torch.zeros(2, 2)},
        "optimizer_state": None,
        "config": None,
        "net_arch": {"channels": 16, "n_blocks": 1, "num_planes": NUM_PLANES},
    }
    if fmt is not None:
        payload["format"] = fmt
    path = directory / "step_000003.pt"
    torch.save(payload, path)
    return path


def test_save_writes_format_marker_and_round_trips(tmp_path: Path) -> None:
    path = save(tmp_path, step=1, net=_net())
    assert torch.load(path)["format"] == CHECKPOINT_FORMAT
    assert load(path, torch.device("cpu"))["step"] == 1


@pytest.mark.parametrize("fmt", [None, 2, CHECKPOINT_FORMAT + 1])
def test_load_rejects_foreign_format_with_one_clear_error(tmp_path: Path, fmt: int | None) -> None:
    """An old run must fail with a single actionable message -- not torch's wall
    of missing/unexpected state_dict keys -- whether or not a net is passed."""
    path = _legacy_checkpoint(tmp_path, fmt)
    for net in (None, _net()):
        with pytest.raises(ValueError, match="start a new run directory") as exc:
            load(path, torch.device("cpu"), net=net)
        msg = str(exc.value)
        assert str(path) in msg
        assert f"format {fmt}" in msg
        assert "Missing key" not in msg


@pytest.mark.parametrize("payload", [[1, 2, 3], torch.zeros(2), "step_000001", None])
def test_load_rejects_non_dict_payload_with_clear_error(tmp_path: Path, payload: object) -> None:
    """A .pt that isn't a checkpoint dict (a bare tensor, a list...) must get the
    same kind of clear ValueError as a foreign format, not an AttributeError
    from poking at it as a dict."""
    path = tmp_path / "step_000001.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="not a checkpoint payload") as exc:
        load(path, torch.device("cpu"))
    assert str(path) in str(exc.value)


@pytest.mark.parametrize("missing", ["num_planes", "channels", "n_blocks"])
def test_load_without_net_requires_complete_stored_arch(tmp_path: Path, missing: str) -> None:
    """Rebuilding from a checkpoint must use exactly the stored shape. Filling a
    missing key from today's model defaults would build a different net and fail
    later with a state_dict shape wall -- or, worse, load silently if the
    default happens to match. Name the missing key instead."""
    path = save(tmp_path, step=1, net=_net())
    payload = torch.load(path)
    del payload["net_arch"][missing]
    torch.save(payload, path)
    with pytest.raises(ValueError, match=rf"net_arch.*missing.*{missing}") as exc:
        load(path, torch.device("cpu"))
    assert str(path) in str(exc.value)
    # A live net makes the stored arch irrelevant: still loads.
    assert load(path, torch.device("cpu"), net=_net())["step"] == 1


def test_load_without_net_rejects_missing_net_arch(tmp_path: Path) -> None:
    path = save(tmp_path, step=1, net=_net())
    payload = torch.load(path)
    del payload["net_arch"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="net_arch"):
        load(path, torch.device("cpu"))
