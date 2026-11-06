"""Batched primitive members and distance-only GEM geometry.

Geometry stays in the input float32/float64 dtype; call outside autocast.
Lattices have row vectors. Sorted size reduction produces an LLL basis at
delta=0.75, with at most 64 eager sweeps or 8 fixed compiled sweeps,
independent state for each cell and no loop over the batch. It is not the
classical indexed LLL algorithm.

Before distance evaluation, non-finite cells become the 1 Angstrom identity.
Row lengths are clamped to [exp(-5), exp(5)] Angstrom, preserving directions;
cells with absolute determinant of the row-normalised basis <= 1e-6 also
become the identity. These bounds follow Crystalite's log-length clamps
without imposing triangular-cell shear clamps on a primitive basis.
Non-finite coordinates become zero and fractional coordinates are wrapped.
Standalone LLL preserves finite singular cells with M=I; non-finite cells
return the identity instead. Numerical breakdown or exhaustion of the cap
also falls back to M=I. The 27-image search is an approximation, not a
general closest-vector guarantee for arbitrary lattices.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from orbitol.lattice import ell


LLL_SWEEPS = 8


@dataclass
class PrimitiveMembers:
    """Packed members, token ownership, representatives and counts."""

    x: torch.Tensor
    owner: torch.Tensor
    valid: torch.Tensor
    rep: torch.Tensor
    count: torch.Tensor


def primitive_lattice(z, G, tb):
    """Return P_c.T @ L_conv without to_matrix's float64 promotion."""
    p = ell(z, tb.group_family[G].long())
    p0, p1, p2, p3, p4, p5 = p.unbind(-1)
    zero = torch.zeros_like(p0)
    conventional = torch.stack([p0.exp(), zero, zero, p1, p2.exp(), zero,
                                p3, p4, p5.exp()], -1).view(-1, 3, 3)
    return torch.bmm(tb.group_Pc[G].to(z.dtype).transpose(-1, -2), conventional)


def expand_members(u_c, row, G, pad_mask, tb, *, n_max):
    """Expand and pack in token order, then frozen table member order."""
    if not isinstance(n_max, int) or n_max < 0:
        raise ValueError("n_max must be a nonnegative integer")
    B, K = row.shape
    keep = tb.row_member_mask[row] & tb.row_prim_mask[row] & ~pad_mask.unsqueeze(-1)
    count = keep.sum(-1)
    total = count.sum(-1)
    # The compiled model checks capacity asynchronously outside Inductor, whose
    # 2.8 decomposition drops _assert_async inside a graph.
    if not torch.compiler.is_compiling() and bool((total > n_max).any()):
        raise ValueError("primitive member count exceeds n_max")
    rep = torch.where(count > 0, count.cumsum(-1) - count, -1)
    valid = torch.arange(n_max, device=u_c.device) < total.unsqueeze(-1)
    if K == 0 or n_max == 0:
        return PrimitiveMembers(
            u_c.new_zeros(B, n_max, 3),
            row.new_full((B, n_max), -1, dtype=torch.int64), valid, rep, count,
        )

    # Select indices before gathering W/v: even a 192-member conventional
    # orbit need only materialise the requested primitive members' geometry.
    S = keep.shape[-1]
    indices = torch.arange(K * S, device=u_c.device).expand(B, -1)
    candidates = torch.where(keep.flatten(1), indices, K * S)
    selected = candidates.topk(min(n_max, K * S), largest=False, sorted=True).values
    if n_max > K * S:
        selected = torch.cat([selected, selected.new_full((B, n_max - K * S), K * S)], -1)
    selected = torch.where(valid, selected, 0)
    owner = selected // S
    member = selected % S
    packed_row = row.gather(1, owner)
    u = u_c.gather(1, owner.unsqueeze(-1).expand(-1, -1, 3)) + 0.5
    W = tb.row_W[packed_row, member].to(u_c.dtype)
    v = tb.row_v[packed_row, member].to(u_c.dtype)
    conventional = ((W * u.unsqueeze(-2)).sum(-1) + v).remainder(1.0)
    inverse, _ = torch.linalg.inv_ex(tb.group_Pc[G].to(u_c.dtype))
    x = torch.bmm(conventional, inverse.transpose(-1, -2)).remainder(1.0)
    # A negative value very near zero can round to one after remainder.
    x = x.remainder(1.0).masked_fill(~valid.unsqueeze(-1), 0.0)
    return PrimitiveMembers(x, owner.masked_fill(~valid, -1), valid, rep, count)


def _det3(matrix):
    """Scalar triple product for a batch of three-row matrices."""
    a, b, c = matrix.unbind(-2)
    return (a * torch.linalg.cross(b, c)).sum(-1)


@lru_cache(maxsize=16)
def _lower_indices(device):
    return torch.tensor([3, 6, 7], device=device)


def lll_reduce(L, *, exceedance_counter=None):
    """Sorted size reduction: L_red = M @ L, with integer-valued M.

    The compatibility name denotes the output's LLL conditions (delta=0.75).
    Each sweep size-reduces all three rows and stably sorts their lengths.
    A size-reduced basis with sorted lengths satisfies those LLL conditions
    (|mu| <= 1/2 and sorted lengths give both Lovasz inequalities). Stop
    independently for each cell when sorting is a no-op. Eager execution
    checks convergence on the host, up to 64 sweeps. Compiled GEM runs exactly
    LLL_SWEEPS sweeps without host checks; completed cells stay frozen.
    The 8-sweep cap differs from 64 only for cells needing more than 8 sweeps:
    still-active cells use the same M=I fallback as the old 64-sweep limit.
    If supplied, add the number of still-active cells at the cap to the scalar
    int64 device tensor exceedance_counter (cumulative), without a host read.
    Reduction decisions are discrete and detached. The returned cell retains
    gradients through M @ L. Non-finite inputs use an identity safe cell.
    """
    identity = torch.eye(3, device=L.device, dtype=L.dtype).expand_as(L)
    finite = torch.isfinite(L).all(dim=(-1, -2))
    safe = torch.where(finite.view(-1, 1, 1), L, identity)
    with torch.no_grad():
        # Uniform rescaling changes no LLL decisions and prevents squared
        # norms from overflowing for large but finite standalone inputs.
        scale = safe.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(torch.finfo(L.dtype).tiny)
        basis = safe / scale
        determinant = _det3(basis)
        # A computed determinant of exactly dependent rows can have rounding
        # residue. Compare to the row-norm product to detect numerical rank
        # loss without rejecting a merely anisotropic diagonal cell.
        volume_scale = torch.linalg.vector_norm(basis, dim=-1).prod(-1)
        good = finite & (determinant.abs() > 4 * torch.finfo(L.dtype).eps * volume_scale)
        basis = torch.where(good.view(-1, 1, 1), basis, identity)
        permutation = basis.square().sum(-1).argsort(dim=-1, stable=True)
        M = identity.gather(1, permutation.unsqueeze(-1).expand_as(identity))
        basis = torch.bmm(M, basis)
        rows = torch.arange(3, device=L.device)
        lower = _lower_indices(L.device).expand(L.shape[0], -1)
        flat_identity = identity.reshape(-1, 9)
        active = good
        # Accepted entries are exactly representable in float32; the bound
        # also keeps the final integer determinant check within int64.
        integer_limit = 2**18
        def sweep(basis, M, good, active):
            a, b, c = basis.unbind(1)
            n0 = (a * a).sum(-1).clamp_min(torch.finfo(L.dtype).tiny)
            mu10 = (a * b).sum(-1) / n0
            q1 = b - mu10.unsqueeze(-1) * a
            n1 = (q1 * q1).sum(-1).clamp_min(torch.finfo(L.dtype).tiny)
            mu21 = (c * q1).sum(-1) / n1
            mu20 = (c * a).sum(-1) / n0
            r10, r21 = mu10.round(), mu21.round()
            s10 = mu10 - r10
            r20 = (mu20 - r21 * s10).round()
            # b' = b-r10*a; c' = c-r21*b'-r20*a. One triangular,
            # determinant-one transform performs all three row operations.
            coefficients = torch.stack([-r10, r21 * r10 - r20, -r21], -1)
            transform = flat_identity.scatter(1, lower, coefficients).view(-1, 3, 3)
            transform = torch.where(active.view(-1, 1, 1), transform, identity)
            M = torch.bmm(transform, M)
            basis = torch.bmm(transform, basis)
            good = good & (M.abs().amax(dim=(-1, -2)) <= integer_limit)
            permutation = basis.square().sum(-1).argsort(dim=-1, stable=True)
            # Freeze completed cells, including at half-integer rounding
            # ties, so their basis does not depend on other batch members.
            active = active & good & (permutation != rows).any(-1)
            permutation = torch.where(active.unsqueeze(-1), permutation, rows).unsqueeze(-1).expand_as(M)
            basis = basis.gather(1, permutation)
            M = M.gather(1, permutation)
            return basis, M, good, active

        if torch.compiler.is_compiling():
            for _ in range(LLL_SWEEPS):
                basis, M, good, active = sweep(basis, M, good, active)
        else:
            for _ in range(64):
                basis, M, good, active = sweep(basis, M, good, active)
                if not bool(active.any()):
                    break
        if exceedance_counter is not None:
            exceedance_counter.add_(active.sum())
        integer_M = M.to(torch.int64)
        det_M = _det3(integer_M)
        good = good & ~active & (det_M.abs() == 1)
        M = torch.where(good.view(-1, 1, 1), M, identity)
        # Protect the public standalone call even if an extreme finite input
        # would overflow the final multiplication in the original units.
        good_output = torch.isfinite(torch.bmm(M, safe)).all(dim=(-1, -2))
        M = torch.where(good_output.view(-1, 1, 1), M, identity)
    return torch.bmm(M, safe), M


def sanitize_cell(L):
    """The cell every GEM distance uses (see the module docstring)."""
    identity = torch.eye(3, dtype=L.dtype, device=L.device).expand_as(L)
    safe = torch.where(torch.isfinite(L).all(dim=(-1, -2), keepdim=True), L, identity)
    scale = safe.abs().amax(-1, keepdim=True).clamp_min(torch.finfo(L.dtype).tiny)
    scaled = safe / scale
    norm = torch.linalg.vector_norm(scaled, dim=-1, keepdim=True).clamp_min(torch.finfo(L.dtype).tiny)
    direction = scaled / norm
    lengths = (scale.clamp_max(math.exp(5)) * norm).clamp(math.exp(-5), math.exp(5))
    bounded = direction * lengths
    good = _det3(direction).abs() > 1e-6
    return torch.where(good.view(-1, 1, 1), bounded, identity)


@lru_cache(maxsize=16)
def _image_offsets(device, dtype):
    return torch.tensor([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
                        device=device, dtype=dtype)


def pair_distances(x, valid, L, *, lll_exceedance_counter=None):
    """Finite minimum-image distances over 27 offsets in an LLL basis."""
    reduced, M = lll_reduce(sanitize_cell(L), exceedance_counter=lll_exceedance_counter)
    inverse, _ = torch.linalg.inv_ex(M)
    coords = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).remainder(1.0)
    delta = coords.unsqueeze(1) - coords.unsqueeze(2)
    delta = torch.bmm(delta.flatten(1, 2), inverse).view_as(delta)
    delta = delta - delta.round()
    offsets = _image_offsets(x.device, x.dtype)
    images = delta.unsqueeze(-2) + offsets
    cartesian = torch.bmm(images.flatten(1, 3), reduced).view_as(images)
    distances = torch.linalg.vector_norm(cartesian, dim=-1).amin(-1)
    pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
    diagonal = torch.eye(x.shape[1], device=x.device, dtype=torch.bool)
    return torch.nan_to_num(distances, nan=0.0, posinf=0.0, neginf=0.0).masked_fill(~pairs | diagonal, 0.0)


def cell_scale(L):
    """Distance normalisation: |det|^(1/3) of the sanitised cell, in Angstrom.

    Unlike the mean edge of an LLL basis, it is continuous in the cell and does not
    depend on the basis, and it comes from the same sanitised cell as the distances,
    so degenerate cells give a finite, positive value.
    """
    return torch.linalg.det(sanitize_cell(L)).abs().pow(1.0 / 3.0)
