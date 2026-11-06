"""Family charts for the six-slot Crystalite lower-triangular latent.

States contain free slots only. ``ell`` rebuilds network inputs; ``params``
returns (a, b, c, alpha, beta, gamma), with angles in degrees.
"""
from __future__ import annotations

import math
from functools import cache

import torch

from orbitol.contracts import FAMILIES, FREE_SLOTS

_HEX_OFFSET = math.log(math.sqrt(3.0) / 2.0)


@cache
@torch.inference_mode(False)
def _free_masks(device):
    masks = [[i in FREE_SLOTS[f] for i in range(6)] for f in FAMILIES]
    return torch.tensor(masks, dtype=torch.bool, device=device)


def free_mask(family: torch.Tensor) -> torch.Tensor:
    return _free_masks(family.device)[family.long()]


def ell(z: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
    p = torch.where(free_mask(family), z, 0.0)
    tied = (family == 3) | (family == 4) | (family == 5)
    hp = z[:, 0].clamp(-5.0 - _HEX_OFFSET, math.log(20.0))
    p0 = torch.where(family == 4, hp, z[:, 0])
    return torch.stack([
        p0,
        torch.where(family == 4, -0.5 * hp.exp(), p[:, 1]),
        torch.where(family == 4, p0 + _HEX_OFFSET, torch.where(tied, p0, p[:, 2])),
        p[:, 3], p[:, 4], torch.where(family == 5, p0, p[:, 5]),
    ], dim=-1)


def readout(F6: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
    f0 = torch.where((family == 3) | (family == 4), (F6[:, 0] + F6[:, 2]) / 2.0, F6[:, 0])
    f0 = torch.where(family == 5, (F6[:, 0] + F6[:, 2] + F6[:, 5]) / 3.0, f0)
    out = torch.cat([f0[:, None], F6[:, 1:]], dim=-1)
    return torch.where(free_mask(family), out, 0.0)


def loss_weights(family: torch.Tensor) -> torch.Tensor:
    w = free_mask(family).float()
    w[:, 0] = torch.where(family == 5, 3.0, torch.where((family == 3) | (family == 4), 2.0, 1.0))
    return w


@cache
@torch.inference_mode(False)
def _clamp_bounds(device, dtype):
    return torch.tensor([5.0, 10.0, 5.0, 10.0, 10.0, 5.0], device=device, dtype=dtype)


def end_clamp(z: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
    bounds = _clamp_bounds(z.device, z.dtype)
    return torch.where(free_mask(family), torch.maximum(torch.minimum(z, bounds), -bounds), 0.0)


def to_matrix(z: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
    p = ell(z.double(), family)
    zero = torch.zeros_like(p[:, 0])
    return torch.stack([p[:, 0].exp(), zero, zero,
                        p[:, 1], p[:, 2].exp(), zero,
                        p[:, 3], p[:, 4], p[:, 5].exp()], dim=-1).reshape(-1, 3, 3)


def params(z: torch.Tensor, family: torch.Tensor) -> torch.Tensor:
    L = to_matrix(z, family)
    lengths = L.square().sum(-1).sqrt()
    a, b, c = lengths.unbind(-1)
    b = torch.where(family >= 3, a, b)
    c = torch.where(family == 5, a, c)
    alpha = torch.rad2deg(torch.acos(((L[:, 1] * L[:, 2]).sum(-1) / (b * c)).clamp(-1, 1)))
    beta = torch.rad2deg(torch.acos(((L[:, 0] * L[:, 2]).sum(-1) / (a * c)).clamp(-1, 1)))
    gamma = torch.rad2deg(torch.acos(((L[:, 0] * L[:, 1]).sum(-1) / (a * b)).clamp(-1, 1)))
    alpha = torch.where(family >= 1, 90.0, alpha)
    beta = torch.where(family >= 2, 90.0, beta)
    gamma = torch.where(family == 4, 120.0, torch.where(family >= 1, 90.0, gamma))
    return torch.stack([a, b, c, alpha, beta, gamma], dim=-1)
