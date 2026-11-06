"""Distance-only GEM with one bias shared by all layers."""
from __future__ import annotations

import torch
from torch import nn

from orbitol.gem_geometry import PrimitiveMembers


GEM_FEATURES = ("distance_norm", "distance_abs")


def rbf(x, n, r):
    """Gaussian radial features with centre spacing r / (n - 1)."""
    centres = torch.linspace(0, r, n, device=x.device, dtype=x.dtype)
    return torch.exp(-((x[..., None] - centres) / (r / (n - 1))).square())


def aggregate(b_member, owner, valid, K):
    """Stable log-sum-exp from (B,H,K,N) member rows to orbit keys.

    Scatter reductions avoid a (B,H,K,K,N) temporary. Empty keys are zero
    with zero gradients, including a batch with no valid members.
    """
    B, H, queries, N = b_member.shape
    if K == 0 or N == 0:
        return b_member.new_zeros(B, H, queries, K)
    keep = (valid & (owner >= 0) & (owner < K))[:, None, None, :]
    index = owner.clamp(0, K - 1)[:, None, None, :].expand_as(b_member)
    # A detached maximum is just a numerical offset: it cancels in logsumexp.
    with torch.no_grad():
        maxima = b_member.new_full((B, H, queries, K), -torch.inf)
        maxima.scatter_reduce_(-1, index, b_member.masked_fill(~keep, -torch.inf),
                               reduce="amax", include_self=True)
        maxima = maxima.masked_fill(~torch.isfinite(maxima), 0.0)
    shifted = torch.where(keep, b_member - maxima.gather(-1, index), 0.0)
    terms = shifted.exp().masked_fill(~keep, 0.0)
    sums = torch.zeros_like(maxima).scatter_add(-1, index, terms)
    return (maxima + sums.clamp_min(torch.finfo(sums.dtype).tiny).log()).masked_fill(sums == 0, 0.0)


class OrbitGEM(nn.Module):
    """Primitive-member bias with the lattice token at the LAST index."""

    def __init__(self, n_heads, mode, *, features=GEM_FEATURES, n_rbf=32,
                 rbf_norm_max=2.0, rbf_abs_max=8.0, hidden=128):
        super().__init__()
        features = tuple(features)
        if mode not in ("multiplicity", "distance") or any(f not in GEM_FEATURES for f in features):
            raise ValueError("GEM requires multiplicity or distance mode and distance-only features; "
                             "direction-dependent features are not allowed.")
        self.n_heads, self.mode, self.features = n_heads, mode, features
        self.n_rbf, self.rbf_norm_max, self.rbf_abs_max = n_rbf, rbf_norm_max, rbf_abs_max
        if mode == "distance":
            self.mlp = nn.Sequential(nn.Linear(len(features) * n_rbf, hidden), nn.SiLU(),
                                     nn.Linear(hidden, n_heads))
            nn.init.zeros_(self.mlp[-1].weight)
            nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, members: PrimitiveMembers, d=None, scale=None, *, seq_len):
        B, K = members.count.shape
        if seq_len != K + 1:
            raise ValueError("GEM requires seq_len = K + 1 (lattice token last)")
        with torch.autocast(members.x.device.type, enabled=False):
            log_count = members.count.float().clamp_min(1).log()
            if self.mode == "multiplicity":
                orbit = log_count[:, None, None, :].expand(B, self.n_heads, K, K)
            else:
                if d is None or scale is None:
                    raise ValueError("Distance GEM requires distances and cell scale")
                # Preserve double precision for standalone geometry checks. The
                # model uses float32 parameters and geometry even under autocast.
                d = d.to(dtype=self.mlp[0].weight.dtype)
                scale = scale.to(d).reshape(B, 1, 1)
                features = [rbf(d / scale, self.n_rbf, self.rbf_norm_max) if f == "distance_norm"
                            else rbf(d, self.n_rbf, self.rbf_abs_max) for f in self.features]
                phi = torch.cat(features, -1) if features else d.new_empty(*d.shape, 0)
                member_bias = self.mlp(phi).permute(0, 3, 1, 2)
                N = members.valid.shape[1]
                pairs = members.valid[:, :, None] & members.valid[:, None, :]
                diagonal = torch.eye(N, device=d.device, dtype=torch.bool)
                member_bias = member_bias.masked_fill((~pairs | diagonal)[:, None], 0.0)
                reps = members.rep.clamp_min(0)[:, None, :, None].expand(B, self.n_heads, K, N)
                query_bias = member_bias.gather(2, reps)
                orbit = aggregate(query_bias, members.owner, members.valid, K).float()
            real = members.count > 0
            orbit = orbit.masked_fill(~(real[:, :, None] & real[:, None, :])[:, None], 0.0)
            bias = log_count.new_zeros(B, self.n_heads, seq_len, seq_len)
            bias[:, :, :K, :K] = orbit
            bias[:, :, K, :K] = log_count[:, None, :]
            return bias
