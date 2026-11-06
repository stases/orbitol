"""Orbit EDM corruption, preconditioning and objectives M and P.

Symmetry maps use explicit products outside autocast. Their association and the
reductions at P1 deliberately agree with the pinned Crystalite implementation.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch

import orbitol.core  # noqa: F401 -- bind the read-only core before importing src
from orbitol import lattice
from src.crystalite.edm_utils import sample_sigma, sigma_to_cnoise, karras_sigma_steps
from orbitol.contracts import OBJECTIVES, canonical_objective


def _objective(objective):
    """(target, skip) of a coordinate objective; accepts the historical aliases M and P."""
    return OBJECTIVES[canonical_objective(objective)]


def map3(matrix, vector):
    """Batched column-vector map, without autocast's matrix-multiply dispatch."""
    return (matrix[..., 0] * vector[..., 0, None]
            + matrix[..., 1] * vector[..., 1, None]
            + matrix[..., 2] * vector[..., 2, None])


def _coord_mask(batch):
    return batch["dof_mask"] & (~batch["pad_mask"])[..., None]


def orbit_noise(batch, generator=None, *, K=None):
    """Draw at capacity K before slicing, preserving the fixed-width RNG stream."""
    u, lat = batch["U"], batch["lat"]
    width = u.shape[1]
    if K is not None and K < width:
        raise ValueError("Noise capacity K is smaller than the batch width")
    full_u = u if K is None or K == width else u.new_empty((u.shape[0], K, u.shape[2]))
    templates = (u.new_empty((*full_u.shape[:-1], 16)), full_u, lat)
    draws = [torch.randn_like(t) if generator is None else
             torch.randn(t.shape, device=t.device, dtype=t.dtype, generator=generator) for t in templates]
    if full_u.shape[1] != width:
        draws[:2] = [draw[:, :width].contiguous() for draw in draws[:2]]
    return dict(zip(("type", "coord", "lat"), draws))


def corrupt(batch, type_clean, sigma, noise, tb, *, objective):
    target, _ = _objective(objective)
    mask = _coord_mask(batch)
    fm = lattice.free_mask(batch["family"])
    uc = torch.where(mask, batch["U"] - 0.5, 0.0)
    zc = torch.where(fm, batch["lat"], 0.0)
    nu = map3(tb.row_N[batch["row"]].to(uc), noise["coord"])
    un = uc + sigma[:, None, None] * nu
    if target == "posterior":
        un = un - torch.round(un)
    return {"type_clean": type_clean, "u_clean": uc, "z_clean": zc,
            "type_noisy": type_clean + sigma[:, None, None] * noise["type"],
            "u_noisy": torch.where(mask, un, 0.0),
            "z_noisy": torch.where(fm, zc + sigma[:, None] * noise["lat"], 0.0)}


def _coeffs(sig, sd):
    return (sd**2 / (sig**2 + sd**2), sig * sd / torch.sqrt(sig**2 + sd**2),
            1.0 / torch.sqrt(sig**2 + sd**2))


def denoise_orbit(model, state, sigma, batch, tb, *, objective, sigma_data,
                  autocast_dtype=None, skip_type_scaling=False):
    _, skip = _objective(objective)
    # This also disables an enclosing training autocast context for all chart work.
    device_type = state["u_noisy"].device.type
    with torch.autocast(device_type, enabled=False):
        ty, u, z = (state[k].float() for k in ("type_noisy", "u_noisy", "z_noisy"))
        sigma = sigma.float()
        st, ot, it = _coeffs(sigma[:, None, None], sigma_data[0])
        su, ou, _ = _coeffs(sigma[:, None, None], sigma_data[1])
        sz, oz, iz = _coeffs(sigma[:, None], sigma_data[2])
        type_in = ty if skip_type_scaling else it * ty
        x = map3(tb.row_A[batch["row"]].to(u), u + 0.5) + tb.row_b[batch["row"]].to(u)
        x_in = x - torch.floor(x)
        lat_in = iz * lattice.ell(z, batch["family"])
        c_noise = sigma_to_cnoise(sigma)
    ctx = torch.autocast(device_type, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    with ctx:
        model_inputs = (dict(u=u, z=z, family=batch["family"])
                        if getattr(model, "gem_mode", "none") != "none" else {})
        if getattr(model, "coord_input", "explicit") == "internal":
            model_inputs["u"] = u
        raw = model(type_in, x_in, lat_in, batch["pad_mask"], c_noise, row=batch["row"], G=batch["G"],
                    **model_inputs)
    with torch.autocast(device_type, enabled=False):
        coord_map = tb.row_Hinv_sqrt if getattr(model, "coord_readout", "projection") == "direct" else tb.row_Pi
        fu = map3(coord_map[batch["row"]].to(u), raw["coord_vel"].float())
        if skip == "edm":
            base = su * u
        elif skip == "unit":
            base = u
        else:
            # Lift-equivariant skip: D(u + n) = D(u) + n, and equal to c_skip * u to first order near u = 0.
            base = u - (1 - su) * torch.sin(2 * math.pi * u) / (2 * math.pi)
        du = base + ou * fu
        dz = sz * z + oz * lattice.readout(raw["lattice_vel"].float(), batch["family"])
        # Padding has a P1 chart, and its unused prediction must remain identical
        # to Crystalite's. Only coordinates beyond the row's actual d are frozen here;
        # corruption, the loss and sampler separately mask padding.
        chart_mask = torch.arange(3, device=u.device) < tb.row_dof[batch["row"]][..., None]
        return {"D_type": st * ty + ot * raw["type_logits"],
                "D_u": torch.where(chart_mask, du, 0.0),
                "D_z": torch.where(lattice.free_mask(batch["family"]), dz, 0.0), "raw": raw}


def wrap_exact(delta, rows, tb):
    """Closest-vector wrap, using Babai plus strict Voronoi improvements.

    A residual is in the Voronoi cell iff no relevant vector
    decreases its squared H norm. Thus termination certifies the answer.
    """
    shape = delta.shape
    x = delta.reshape(-1, 3)
    row = rows.reshape(-1)
    out = x - torch.round(x)
    for r in torch.unique(row[~tb.row_wrap_fast[row]]).tolist():
        sel = row == r
        d = int(tb.row_dof[r])
        if not d:
            out[sel] = 0.0
            continue
        v = x[sel, :d].double()
        H = tb.row_H[r, :d, :d].double()
        basis = tb.row_lambda_basis[r, :d, :d].double()
        # Nearest-plane Babai in the whitened, LLL-reduced basis.
        whiten = torch.linalg.cholesky(H).T
        Q, R = torch.linalg.qr(whiten @ basis)
        y = v @ (Q.T @ whiten).T
        coeff = torch.zeros_like(y)
        for j in range(d - 1, -1, -1):
            coeff[:, j] = torch.round((y[:, j] - (coeff[:, j + 1:] * R[j, j + 1:]).sum(-1)) / R[j, j])
        e = v - coeff @ basis.T
        vr = tb.row_vrv[r, :int(tb.row_vrv_count[r]), :d].double()
        if len(vr):
            hv = vr @ H
            vv = (hv * vr).sum(-1)
            while True:
                gain, idx = (2.0 * (e @ hv.T) - vv).max(-1)
                improve = gain > 16 * torch.finfo(e.dtype).eps * (1 + (e.square().sum(-1)))
                if not improve.any():
                    break
                e = e - torch.where(improve[:, None], vr[idx], 0.0)
        padded = torch.zeros((len(v), 3), device=v.device, dtype=delta.dtype)
        padded[:, :d] = e.to(delta.dtype)
        out[sel] = padded
    mask = torch.arange(3, device=x.device) < tb.row_dof[row, None]
    return torch.where(mask, out, 0.0).reshape(shape)


def _integer_grid(radii, device):
    axes = [torch.arange(-int(r), int(r) + 1, device=device, dtype=torch.float64) for r in radii]
    return torch.cartesian_prod(*axes).reshape(-1, len(axes))


def _image_terms(delta, H, sigma, images=None):
    """Vectorised adaptive image window, including the nearest image at tiny sigma."""
    delta = delta - torch.round(delta)
    if images is None:
        inv = torch.linalg.inv(H)
        r0 = ((delta @ H) * delta).sum(-1)
        # r_min <= r0, so this bounding box contains every point of the required window.
        bound = r0 + 2 * sigma.square() * math.log(1e12)
        radii = torch.ceil(0.5 + (bound.max() * inv.diag()).sqrt()).long().tolist()
        images = _integer_grid(radii, delta.device)
    e = delta[:, None, :] - images
    r2 = ((e @ H) * e).sum(-1)
    rmin = r2.min(-1).values
    logw = -(r2 - rmin[:, None]) / (2 * sigma[:, None].square())
    logw = logw.masked_fill(r2 > rmin[:, None] + 2 * sigma[:, None].square() * math.log(1e12), -torch.inf)
    return e, torch.softmax(logw, -1), rmin


def periodic_target(u_noisy, u_clean, H, dof, sigma):
    """Conditional lifted mean on R^d/Z^d, evaluated in float64.

    Groups equal metrics, vectorises over tokens and chunks only to bound
    memory. Neither primal nor dual windows have a fixed image-count cutoff.
    Sigma may have token shape or the leading batch shape.
    """
    shape, dtype = u_noisy.shape, u_noisy.dtype
    token_shape = shape[:-1]
    sig = sigma
    while sig.ndim < len(token_shape):
        sig = sig.unsqueeze(-1)
    sig = sig.expand(token_shape).reshape(-1).double()
    if torch.any(sig <= 0):
        raise ValueError("periodic_target requires positive sigma")
    u, uc = u_noisy.reshape(-1, 3).double(), u_clean.reshape(-1, 3).double()
    H = H.expand(*token_shape, 3, 3).reshape(-1, 3, 3).double()
    dof = dof.expand(token_shape).reshape(-1)
    out = torch.zeros_like(u)
    descriptors = torch.cat([dof[:, None].double(), H.flatten(1)], -1)
    unique, group = torch.unique(descriptors, dim=0, return_inverse=True)
    for j, desc in enumerate(unique):
        d = int(desc[0])
        if not d:
            continue
        h = desc[1:].reshape(3, 3)[:d, :d]
        hi = torch.linalg.inv(h)
        ev = torch.linalg.eigvalsh(h)
        switch = 0.4 * ev[-1].sqrt()
        # Exact shortest nonzero dual vector: a coordinate vector provides the
        # upper bound, and the smallest eigenvalue bounds the search box.
        kr = math.ceil(math.sqrt(float(hi.diag().min() * ev[-1])))
        ks = _integer_grid([kr] * d, u.device)
        kk = ((ks @ hi) * ks).sum(-1)
        kmin = kk[kk > 0].min()
        # Conservative dimension/conditioning prefactor for the theta tail and
        # its derivative. Using it also gives headroom over the 1e-12 tolerance.
        C = 1e4 * (1 + float(ev[-1] / ev[0])) ** d
        logtol = math.log(1e12 * C)
        sigma_hi = torch.sqrt(logtol / (2 * math.pi**2 * kmin))
        indices = torch.nonzero(group == j).flatten()
        for ix in indices.split(256):
            out[ix, :d] = u[ix, :d]
            low = ix[sig[ix] < switch]
            if len(low):
                e, p, _ = _image_terms(u[low, :d] - uc[low, :d], h, sig[low])
                out[low, :d] = u[low, :d] - (p[..., None] * e).sum(1)
            dual = ix[(sig[ix] >= switch) & (sig[ix] < sigma_hi)]
            if len(dual):
                limit = logtol / (2 * math.pi**2 * sig[dual].min().square())
                radii = torch.ceil((limit * h.diag()).sqrt()).long().tolist()
                k = _integer_grid(radii, u.device)
                norm = ((k @ hi) * k).sum(-1)
                keep = (norm > 0) & (norm <= limit)
                k, norm = k[keep], norm[keep]
                phase = 2 * math.pi * ((u[dual, :d] - uc[dual, :d]) @ k.T)
                q = torch.exp(-2 * math.pi**2 * sig[dual, None].square() * norm)
                density = 1 + (q * torch.cos(phase)).sum(-1)
                grad = (-2 * math.pi * q * torch.sin(phase)) @ k
                out[dual, :d] = u[dual, :d] + sig[dual, None].square() * (grad @ hi) / density[:, None]
    return out.to(dtype).reshape(shape)


class PeriodicTarget:
    """Prepared training target for fixed tables and positive sampled sigma.

    For the wrapped delta, r0 <= sum(abs(H))/4. The primal
    branch has sigma < switch, so the resulting fixed box contains its whole
    adaptive window. In the dual branch sigma >= switch bounds every dual
    radius. The loop retains the original ellipsoids and chunk cutoffs.
    The loop groups CPU rows; the batched path gathers fixed-shape device rows.

    ``batched`` pads the prepared tables and evaluates all tokens together,
    with no metric/chunk loop or device-to-host reads. Image storage is bounded
    by the switch, not the sampled sigma: O(N * (P + K)) float64 values (with
    at most three coordinate components). The shipped tables have P <= 611 and
    K <= 178 after the conservative primal pruning below; the largest individual
    temporary at N = 128 * 20 is N * P * 3 * 8 bytes, about 36 MiB.
    """

    def __init__(self, row_H, row_dof, impl="loop"):
        if impl not in ("loop", "batched"):
            raise NotImplementedError(f"periodic_target_impl={impl!r} is not implemented yet")
        self.impl = impl
        self.device = row_H.device
        H, dof = row_H.detach().cpu().double(), row_dof.cpu()
        descriptors = torch.cat([dof[:, None].double(), H.flatten(1)], -1)
        unique, self.row_group = torch.unique(descriptors, dim=0, return_inverse=True)
        self.metrics = {}
        for j, desc in enumerate(unique):
            d = int(desc[0])
            if not d:
                continue
            h = desc[1:].reshape(3, 3)[:d, :d]
            hi, ev = torch.linalg.inv(h), torch.linalg.eigvalsh(h)
            switch = float(0.4 * ev[-1].sqrt())
            kr = math.ceil(math.sqrt(float(hi.diag().min() * ev[-1])))
            ks = _integer_grid([kr] * d, "cpu")
            kk = ((ks @ hi) * ks).sum(-1)
            C = 1e4 * (1 + float(ev[-1] / ev[0])) ** d
            logtol = math.log(1e12 * C)
            sigma_hi = float(torch.sqrt(logtol / (2 * math.pi**2 * kk[kk > 0].min())))
            bound = h.abs().sum() / 4 + 2 * switch**2 * math.log(1e12)
            radii = torch.ceil(0.5 + (bound * hi.diag()).sqrt()).long().tolist()
            images = _integer_grid(radii, "cpu")
            limit = logtol / (2 * math.pi**2 * switch**2)
            radii = torch.ceil((limit * h.diag()).sqrt()).long().tolist()
            k = _integer_grid(radii, "cpu")
            norm = ((k @ hi) * k).sum(-1)
            keep = (norm > 0) & (norm <= limit)
            self.metrics[j] = (d, switch, sigma_hi, logtol,
                               *(t.to(self.device if impl == "loop" else "cpu")
                                 for t in (h, hi, images, k[keep], norm[keep])))
        if impl == "batched":
            self._prepare_batched(len(unique))

    def _prepare_batched(self, count):
        """Bound image norms by the triangle inequality in H.

        For every retained image n, ||n||_H <= ||delta||_H + ||delta-n||_H
        <= sqrt(r0_max) + sqrt(r0_max + 2 * switch**2 * log(1e12)). Pruning
        outside this ball removes only terms masked out by the loop already.
        """
        images = {}
        for j, (_, switch, _, _, h, _, grid, _, _) in self.metrics.items():
            r0 = h.abs().sum() / 4
            bound = (r0.sqrt() + (r0 + 2 * switch**2 * math.log(1e12)).sqrt()).square()
            images[j] = grid[((grid @ h) * grid).sum(-1) <= bound * (1 + 1e-14)]
        p = max((len(v) for v in images.values()), default=1)
        k = max((len(v[7]) for v in self.metrics.values()), default=1)
        tables = dict(h=torch.zeros(count, 3, 3, dtype=torch.float64),
                      hi=torch.zeros(count, 3, 3, dtype=torch.float64),
                      params=torch.ones(count, 3, dtype=torch.float64),
                      mask=torch.zeros(count, 3, dtype=torch.bool),
                      images=torch.zeros(count, p, 3, dtype=torch.float64),
                      valid=torch.zeros(count, p, dtype=torch.bool),
                      k=torch.zeros(count, k, 3, dtype=torch.float64),
                      norm=torch.zeros(count, k, dtype=torch.float64))
        # A single zero image keeps the softmax finite for zero-dof groups.
        tables["valid"][:, 0] = True
        for j, (d, switch, sigma_hi, logtol, h, hi, _, ks, norm) in self.metrics.items():
            tables["h"][j, :d, :d], tables["hi"][j, :d, :d] = h, hi
            tables["params"][j] = torch.tensor([switch, sigma_hi, logtol], dtype=torch.float64)
            tables["mask"][j, :d] = True
            tables["images"][j, :len(images[j]), :d] = images[j]
            tables["valid"][j, :len(images[j])] = True
            tables["k"][j, :len(ks), :d], tables["norm"][j, :len(ks)] = ks, norm
        self.batched = {name: value.to(self.device) for name, value in tables.items()}
        self.row_group_dev = self.row_group.to(self.device)

    def groups(self, rows):
        if self.impl == "batched":
            return self.row_group_dev[rows.reshape(-1).to(self.device, non_blocking=True)]
        groups = self.row_group[rows.reshape(-1)]
        indices = [(j, torch.nonzero(groups == j).flatten()) for j in self.metrics]
        indices = [(j, ix) for j, ix in indices if len(ix)]
        if not indices:
            return []
        packed = torch.cat([ix for _, ix in indices]).to(self.device, non_blocking=True)
        return list(zip((j for j, _ in indices), packed.split([len(ix) for _, ix in indices])))

    def __call__(self, u_noisy, u_clean, sigma, groups):
        if self.impl == "batched":
            return self._batched(u_noisy, u_clean, sigma, groups)
        shape = u_noisy.shape
        u, uc = u_noisy.reshape(-1, 3).double(), u_clean.reshape(-1, 3).double()
        sig = sigma[:, None].expand(shape[:-1]).reshape(-1).double()
        out = torch.zeros_like(u)
        for j, indices in groups:
            d, switch, sigma_hi, logtol, h, hi, images, k, norm = self.metrics[j]
            for ix in indices.split(256):
                s = sig[ix]
                delta = u[ix, :d] - uc[ix, :d]
                e, p, _ = _image_terms(delta, h, s.clamp_max(switch), images)
                primal = u[ix, :d] - (p[..., None] * e).sum(1)
                dual = (s >= switch) & (s < sigma_hi)
                limit = logtol / (2 * math.pi**2 * s.masked_fill(~dual, torch.inf).min().square())
                phase = 2 * math.pi * (delta @ k.T)
                q = torch.exp(-2 * math.pi**2 * s.clamp_min(switch)[:, None].square() * norm)
                q = q.masked_fill(norm > limit, 0.0)
                density = 1 + (q * torch.cos(phase)).sum(-1)
                grad = (-2 * math.pi * q * torch.sin(phase)) @ k
                theta = u[ix, :d] + s[:, None].square() * (grad @ hi) / density[:, None]
                out[ix, :d] = torch.where((s < switch)[:, None], primal,
                                          torch.where(dual[:, None], theta, u[ix, :d]))
        return out.to(u_noisy.dtype).reshape(shape)

    def _batched(self, u_noisy, u_clean, sigma, groups):
        """Float64 reductions, returning the input dtype as in the loop.

        Per-token dual cutoffs make the result independent of row
        ordering and chunk boundaries. Compared with a chunk's minimum sigma,
        this discards only terms below 1/(1e12*C), beneath the 1e-12 target
        tolerance. Padded reductions can also change float64 rounding.
        """
        shape = u_noisy.shape
        u, uc = u_noisy.reshape(-1, 3).double(), u_clean.reshape(-1, 3).double()
        s = sigma[:, None].expand(shape[:-1]).reshape(-1).double()
        t = self.batched
        switch, sigma_hi, logtol = t["params"][groups].unbind(-1)
        mask = t["mask"][groups]
        delta = torch.where(mask, u - uc, 0.0)
        wrapped = delta - torch.round(delta)
        e = wrapped[:, None, :] - t["images"][groups]
        r2 = (torch.bmm(e, t["h"][groups]) * e).sum(-1)
        valid = t["valid"][groups]
        rmin = r2.masked_fill(~valid, torch.inf).min(-1).values[:, None]
        s2 = s.clamp_max(switch).square()[:, None]
        logw = -(r2 - rmin) / (2 * s2)
        logw = logw.masked_fill(~valid | (r2 > rmin + 2 * s2 * math.log(1e12)), -torch.inf)
        p = torch.softmax(logw, -1)
        primal = u - (p[..., None] * e).sum(1)
        del e, r2, logw, p

        dual = (s >= switch) & (s < sigma_hi)
        limit = logtol / (2 * math.pi**2 * s.square())
        k, norm = t["k"][groups], t["norm"][groups]
        phase = 2 * math.pi * torch.bmm(k, delta[..., None]).squeeze(-1)
        q = torch.exp(-2 * math.pi**2 * s.clamp_min(switch)[:, None].square() * norm)
        q = q.masked_fill((norm == 0) | (norm > limit[:, None]), 0.0)
        density = 1 + (q * torch.cos(phase)).sum(-1)
        grad = torch.bmm((-2 * math.pi * q * torch.sin(phase))[:, None, :], k)
        theta = u + s[:, None].square() * torch.bmm(grad, t["hi"][groups]).squeeze(1) / density[:, None]
        out = torch.where((s < switch)[:, None], primal, torch.where(dual[:, None], theta, u))
        return torch.where(mask, out, 0.0).to(u_noisy.dtype).reshape(shape)


def compute_orbit_loss(den, state, sigma, batch, tb, *, objective, sigma_data, loss_weights,
                       coord_target=None):
    target_kind, _ = _objective(objective)
    with torch.autocast(sigma.device.type, enabled=False):
        real = ~batch["pad_mask"]
        active = real & (tb.row_dof[batch["row"]] > 0)
        sb, sl = sigma.float()[:, None, None], sigma.float()[:, None]

        def weight(s, sd):
            return (s**2 + sd**2) / (s * sd)**2

        def reduce(value, mask):
            m = mask[..., None].float()
            return (value * m).sum() / m.sum().clamp_min(1.0)

        lt = reduce(weight(sb, sigma_data[0]) * (den["D_type"] - state["type_clean"])**2, real)
        if target_kind == "wrapped":
            diff = wrap_exact(den["D_u"] - state["u_clean"], batch["row"], tb)
        else:
            target = coord_target
            if target is None:
                target = periodic_target(state["u_noisy"], state["u_clean"], tb.row_H[batch["row"]],
                                         tb.row_dof[batch["row"]], sigma)
            diff = den["D_u"] - target
        e = map3(tb.row_A[batch["row"]].to(diff), diff)
        # M_G is I except in hexagonal axes. Keep e unchanged at P1.
        hexagonal = (tb.group_family[batch["G"]] == 4)[:, None]
        em = torch.stack([torch.where(hexagonal, (4 / 3) * e[..., 0] - (2 / 3) * e[..., 1], e[..., 0]),
                          torch.where(hexagonal, (4 / 3) * e[..., 1] - (2 / 3) * e[..., 0], e[..., 1]),
                          e[..., 2]], -1)
        err = em * e
        lf = reduce(weight(sb, sigma_data[1]) * err, active)
        errz = (den["D_z"] - state["z_clean"])**2
        ly = ((weight(sl, sigma_data[2]) * errz * lattice.loss_weights(batch["family"])).mean()
              if errz.numel() else sigma.new_tensor(0.0))
        w = [loss_weights[k] for k in ("A", "F", "Y")] if isinstance(loss_weights, dict) else loss_weights
        return {"total": w[0] * lt + w[1] * lf + w[2] * ly, "A": lt, "F": lf, "Y": ly}
