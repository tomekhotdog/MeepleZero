"""Fully-convolutional residual policy/value CNN -- size-agnostic (iteration 3).

Reads the ``(NUM_PLANES, H, W)`` dynamic-crop encoding and emits, per state:

* **policy** -- one logit per flat action index, layout ``((gy*W+gx)*4+rot)*14+a``
  (see :mod:`carcassonne.nn.actions`); shape ``(N, H*W*56)`` for whatever H, W
  the input has.
* **value** in ``[-1, 1]`` (tanh), current-player perspective, via a masked
  global-average pool -- no Linear tied to a board size anywhere.

Normalisation is :class:`MaskedGroupNorm`: GroupNorm statistics over *valid*
cells only, output re-zeroed on padding. Invariant (tested): a batch-padded
forward pass equals the unpadded forward pass exactly on the real cells --
padding cells stay 0 through every layer, so a 3x3 conv at a real border cell
sees the same zeros as implicit zero-padding.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from carcassonne.core.engine import GameState
from carcassonne.core.types import Move
from carcassonne.nn.actions import ACTIONS, ACTIONS_PER_CELL, ROTS, Crop, move_indices
from carcassonne.nn.encode import NUM_PLANES, encode_state

_VALUE_CHANNELS = 8
_VALUE_HIDDEN = 32
DEFAULT_CHANNELS = 48
DEFAULT_N_BLOCKS = 4


def _conv3x3(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)


def _gn_groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else math.gcd(8, channels)


class MaskedGroupNorm(nn.Module):
    """GroupNorm whose statistics ignore padding cells.

    ``forward(x, cell_mask)``: with ``cell_mask=None`` this is exactly
    ``F.group_norm`` (single-state inference, no padding). With a ``(N, 1, H, W)``
    0/1 mask, mean/var are computed over valid cells only and the output is
    multiplied by the mask -- keeping padded cells exactly 0 (the affine bias
    would otherwise leak into real cells through the next conv).
    """

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.groups = _gn_groups(channels)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor, cell_mask: torch.Tensor | None = None) -> torch.Tensor:
        if cell_mask is None:
            return F.group_norm(x, self.groups, self.weight, self.bias, self.eps)
        n, c, h, w = x.shape
        g = self.groups
        xg = x.reshape(n, g, c // g, h, w)
        mg = cell_mask.reshape(n, 1, 1, h, w)
        count = mg.sum(dim=(2, 3, 4), keepdim=True) * (c // g)
        mean = (xg * mg).sum(dim=(2, 3, 4), keepdim=True) / count
        var = ((xg - mean) ** 2 * mg).sum(dim=(2, 3, 4), keepdim=True) / count
        xg = (xg - mean) / torch.sqrt(var + self.eps)
        out = xg.reshape(n, c, h, w)
        out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out * cell_mask


class _ResBlock(nn.Module):
    """[conv3x3+MGN+ReLU, conv3x3+MGN] + skip, then ReLU. Mask-preserving."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _conv3x3(channels, channels)
        self.gn1 = MaskedGroupNorm(channels)
        self.conv2 = _conv3x3(channels, channels)
        self.gn2 = MaskedGroupNorm(channels)

    def forward(self, x: torch.Tensor, cell_mask: torch.Tensor | None) -> torch.Tensor:
        out = torch.relu(self.gn1(self.conv1(x), cell_mask))
        out = self.gn2(self.conv2(out), cell_mask)
        return torch.relu(out + x)


def _policy_head_flatten(logits: torch.Tensor) -> torch.Tensor:
    """``(N, 56, H, W)`` conv output -> ``(N, H*W*56)`` in flat-index order
    ``((gy*W + gx)*4 + rot)*14 + a`` (see :func:`carcassonne.nn.actions.flat_index`).

    Channels encode ``rot*14 + a``; target flat order (slow->fast) is
    ``gy, gx, rot, a`` -- split channels to (rot, a), permute, flatten.
    """
    n, _, h, w = logits.shape
    logits = logits.reshape(n, ROTS, ACTIONS, h, w)
    logits = logits.permute(0, 3, 4, 1, 2)
    return logits.reshape(n, h * w * ACTIONS_PER_CELL)


# The keys of an arch dict (see CarcassonneNet.from_arch / .arch).
ARCH_KEYS = ("num_planes", "channels", "n_blocks")


class CarcassonneNet(nn.Module):
    """Size-agnostic residual CNN: shared trunk, per-cell policy head, pooled
    value head.

    ``forward(x, mask=None, cell_mask=None) -> (policy_logits, value)``:

    * ``x`` -- ``(N, NUM_PLANES, H, W)`` for any H, W.
    * ``mask`` -- optional ``(N, H*W*56)`` bool; illegal logits -> ``-inf``.
    * ``cell_mask`` -- optional ``(N, H, W)`` or ``(N, 1, H, W)`` 0/1 float;
      REQUIRED whenever the batch is padded (learner), None for unpadded input.
    """

    def __init__(
        self,
        num_planes: int = NUM_PLANES,
        channels: int = DEFAULT_CHANNELS,
        n_blocks: int = DEFAULT_N_BLOCKS,
    ) -> None:
        super().__init__()
        self.num_planes = num_planes
        self.channels = channels
        self.n_blocks = n_blocks

        self.stem_conv = _conv3x3(num_planes, channels)
        self.stem_gn = MaskedGroupNorm(channels)
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(n_blocks))

        # Policy head: plain 1x1 conv (bias on, no norm) -- logits per cell.
        self.policy_conv = nn.Conv2d(channels, ACTIONS_PER_CELL, kernel_size=1)

        # Value head: 1x1 conv -> ReLU -> masked global-average pool -> MLP -> tanh.
        self.value_conv = nn.Conv2d(channels, _VALUE_CHANNELS, kernel_size=1)
        self.value_fc1 = nn.Linear(_VALUE_CHANNELS, _VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(_VALUE_HIDDEN, 1)

    @classmethod
    def from_arch(cls, arch: Mapping[str, Any]) -> CarcassonneNet:
        """Build a net from an arch dict (``num_planes``/``channels``/``n_blocks``,
        as stored in a checkpoint or a run config); missing keys take the defaults."""
        return cls(
            num_planes=int(arch.get("num_planes", NUM_PLANES)),
            channels=int(arch.get("channels", DEFAULT_CHANNELS)),
            n_blocks=int(arch.get("n_blocks", DEFAULT_N_BLOCKS)),
        )

    @property
    def arch(self) -> dict[str, int]:
        """The inverse of :meth:`from_arch`: every key needed to rebuild this
        net's shape (what checkpoints store and pool workers are sent)."""
        return {
            "num_planes": self.num_planes,
            "channels": self.channels,
            "n_blocks": self.n_blocks,
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        cell_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cm = None
        if cell_mask is not None:
            cm = cell_mask if cell_mask.dim() == 4 else cell_mask.unsqueeze(1)

        h = torch.relu(self.stem_gn(self.stem_conv(x), cm))
        for block in self.blocks:
            h = block(h, cm)

        policy_logits = _policy_head_flatten(self.policy_conv(h))
        if mask is not None:
            # Illegal (and padded) actions -> -inf; the learner's cross-entropy
            # must stay on the target's support (0 * -inf = NaN otherwise).
            policy_logits = policy_logits.masked_fill(~mask, float("-inf"))

        v = torch.relu(self.value_conv(h))
        if cm is None:
            pooled = v.mean(dim=(2, 3))
        else:
            pooled = (v * cm).sum(dim=(2, 3)) / cm.sum(dim=(2, 3))
        value = torch.tanh(self.value_fc2(torch.relu(self.value_fc1(pooled)))).squeeze(-1)

        return policy_logits, value


def param_count(net: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def evaluate_state(
    net: CarcassonneNet, state: GameState, device: torch.device, moves: Sequence[Move]
) -> tuple[NDArray[np.float32], float]:
    """Evaluate one ``state``: (priors aligned with ``moves``, value).

    ``moves`` are the caller's legal moves for ``state`` (passed in so a leaf
    enumerates them once). ``priors[i]`` is the net's masked-softmax probability
    of ``moves[i]``; they sum to 1 when ``moves`` is the full legal set. The Crop
    and flat-index layout stay inside nn -- callers only ever see move order.
    Unpadded single state -> ``cell_mask=None`` (fast ``F.group_norm`` path).
    """
    crop = Crop.of(state)
    idx = move_indices(crop, moves)
    mask = np.zeros(crop.n_actions, dtype=np.bool_)
    mask[idx] = True

    x = torch.from_numpy(encode_state(state, crop)).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)

    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            policy_logits, value = net(x, mask_t)
            probs = torch.softmax(policy_logits, dim=1).squeeze(0)
    finally:
        net.train(was_training)

    return probs.cpu().numpy().astype(np.float32)[idx], float(value.item())
