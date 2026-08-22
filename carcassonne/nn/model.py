"""Residual policy/value CNN for the AlphaZero-style agent.

The network reads the ``(NUM_PLANES, WINDOW, WINDOW)`` encoding produced by
:mod:`carcassonne.nn.encode` and emits, per state:

* a **policy** over the whole action space -- one logit per action id, in exactly
  the :mod:`carcassonne.nn.actions` layout ``((wy*B+wx)*4+rot)*14+a`` so that
  ``policy_logits[..., i]`` is the logit for action id ``i``;
* a **value** in ``[-1, 1]`` (tanh), from the *current player's* perspective.

Head/ActionIndexer alignment (the single most bug-prone thing here) is proven in
``tests/nn/test_model.py`` -- see :func:`_policy_head_flatten` for the reshape.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from carcassonne.core.engine import GameState
from carcassonne.nn.actions import ACTION_SPACE, WINDOW
from carcassonne.nn.encode import NUM_PLANES, encode_state

_ROTS = 4
_ACTIONS = 14
_ACTIONS_PER_CELL = _ROTS * _ACTIONS  # 56 policy channels, encoding rot*14 + a
_VALUE_CHANNELS = 4

assert _ACTIONS_PER_CELL * WINDOW * WINDOW == ACTION_SPACE


def _conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


class _ResBlock(nn.Module):
    """[conv3x3+BN+ReLU, conv3x3+BN] + skip, then ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _conv3x3(channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


def _policy_head_flatten(logits: torch.Tensor) -> torch.Tensor:
    """``(N, 56, H, W)`` conv output -> ``(N, ACTION_SPACE)`` in ActionIndexer order.

    The 56 channels encode ``rot*14 + a`` and the spatial axes are ``(wy, wx)``
    (matching ``encode``'s ``planes[c, wy, wx]``). The target flat index is
    ``((wy*B+wx)*4+rot)*14+a``, i.e. axis order (slow->fast) ``wy, wx, rot, a``.

    So: split the channel dim ``56 -> (rot=4, a=14)``, permute to
    ``(N, wy, wx, rot, a)``, then flatten. Row-major flatten of that order is
    exactly the action-id formula.
    """
    n = logits.shape[0]
    # (N, 56, H, W) -> (N, rot, a, wy, wx)
    logits = logits.reshape(n, _ROTS, _ACTIONS, WINDOW, WINDOW)
    # -> (N, wy, wx, rot, a)
    logits = logits.permute(0, 3, 4, 1, 2)
    return logits.reshape(n, ACTION_SPACE)


class CarcassonneNet(nn.Module):
    """Small residual CNN: shared trunk, policy head, value head.

    ``forward(x, mask=None) -> (policy_logits, value)``:

    * ``policy_logits`` -- ``(N, ACTION_SPACE)``. If ``mask`` (bool, same shape)
      is given, logits at illegal actions (``mask`` False) are set to ``-inf``,
      so a softmax over the row yields exactly 0 probability there.
    * ``value`` -- ``(N,)`` in ``[-1, 1]`` (tanh), current-player perspective.
    """

    def __init__(
        self,
        num_planes: int = NUM_PLANES,
        channels: int = 64,
        n_blocks: int = 5,
    ) -> None:
        super().__init__()
        self.num_planes = num_planes
        self.channels = channels
        self.n_blocks = n_blocks

        self.stem = nn.Sequential(
            _conv3x3(num_planes, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(n_blocks))

        # Policy head: conv1x1 -> BN -> ReLU, then reshape to ACTION_SPACE.
        self.policy_conv = nn.Conv2d(channels, _ACTIONS_PER_CELL, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(_ACTIONS_PER_CELL)

        # Value head: conv1x1 -> BN -> ReLU -> flatten -> Linear -> ReLU -> Linear -> tanh.
        self.value_conv = nn.Conv2d(channels, _VALUE_CHANNELS, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(_VALUE_CHANNELS)
        self.value_fc1 = nn.Linear(_VALUE_CHANNELS * WINDOW * WINDOW, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        policy_logits = _policy_head_flatten(p)
        if mask is not None:
            policy_logits = policy_logits.masked_fill(~mask, float("-inf"))

        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)
        v = torch.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)

        return policy_logits, value


def param_count(net: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def evaluate_state(
    net: CarcassonneNet, state: GameState, device: torch.device
) -> tuple[NDArray[np.float32], float]:
    """Encode one ``state``, run the net masked to legal moves, return (probs, value).

    ``probs`` is a ``(ACTION_SPACE,)`` float32 array that sums to 1 over legal
    actions and is exactly 0 on illegal ones (the masked ``-inf`` logits softmax
    to zero). ``value`` is the scalar tanh estimate for the current player. This
    is the bridge MCTS/agents call; the model itself stays pure tensors.
    """
    from carcassonne.nn.actions import legal_mask  # local: avoid import cycle at module load

    planes = encode_state(state)
    x = torch.from_numpy(planes).unsqueeze(0).to(device)
    mask_np = legal_mask(state)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)

    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            policy_logits, value = net(x, mask)
            probs = torch.softmax(policy_logits, dim=1).squeeze(0)
    finally:
        net.train(was_training)

    return probs.cpu().numpy().astype(np.float32), float(value.item())
