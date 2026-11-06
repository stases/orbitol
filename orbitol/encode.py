"""Fixed-shape type lookup for caches validated before training."""
from __future__ import annotations

import numpy as np
import torch


def validate_a0_cache(cache, max_z):
    """Check the real tokens consumed by OrbitDataset, once on the CPU."""
    a0 = np.asarray(cache["A0"], dtype=np.int64)
    real = np.arange(a0.shape[1]) < np.asarray(cache["n_orbits"])[:, None]
    if np.any(real & ((a0 < 1) | (a0 > max_z))):
        raise ValueError(f"Cached atomic numbers outside descriptor range [1, {max_z}].")


class FastEncoder:
    """Compose the core's frozen table without per-batch host reads.

    Real tokens must be in [1, max_z]; training validates both caches here.
    No per-step assertion is needed because augmentation never changes A0.
    The private zero row makes padding exact even for arbitrary padded A0.
    """

    def __init__(self, encoding, device, *, caches=()):
        self.max_z = int(encoding.max_z)
        for cache in caches:
            validate_a0_cache(cache, self.max_z)
        self.z_to_enc = encoding.z_to_enc[:self.max_z + 1].detach().to(device, copy=True)
        self.z_to_enc[0].zero_()
        self.zero = torch.zeros((), dtype=torch.long, device=device)

    def __call__(self, a0, pad_mask):
        return self.z_to_enc[torch.where(pad_mask.bool(), self.zero, a0.long())]
