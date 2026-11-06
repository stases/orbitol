"""Crystalite's EDM Heun loop on orbit charts; the denoiser is the configured model, GEM included."""
from __future__ import annotations

import math

import torch

from orbitol import lattice
from orbitol.edm import _objective, denoise_orbit, karras_sigma_steps, map3, wrap_exact


@torch.no_grad()
def orbit_edm_sampler(model, skel, tb, *, objective, settings, generator=None,
                      denoise_fn=None):
    """Sample free states; rebuild the returned lattice with ``lattice.ell``.

    ``denoise_fn(state, sigma, skel, tb, *, objective, sigma_data)`` replaces
    the full denoiser and returns D_type, D_u and D_z. This permits analytic
    coordinate oracles without changing the integration or random stream.
    Settings may be a mapping or an object with the documented attributes.
    ``finite`` records each sample's active-state finiteness before sanitisation.
    """
    coord_target, coord_skip = _objective(objective)
    # The EDM skip is the one coordinate parameterisation that is not lift-equivariant (D(u + n) = D(u) + c_skip n),
    # so its denoiser must see the canonical chart state it was trained on. The drift and the update then use that
    # same state.
    canonical_chart = coord_target == "posterior" and coord_skip == "edm"
    def setting(name, default=None):
        return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)

    pad = skel["pad_mask"]
    device = pad.device
    bsz, nmax = pad.shape
    real = ~pad
    dof_mask = skel["dof_mask"] & real[..., None]
    fm = lattice.free_mask(skel["family"])
    N = tb.row_N[skel["row"]].float()
    num_steps = setting("num_steps")
    sigma_min, sigma_max, rho = (setting(k) for k in ("sigma_min", "sigma_max", "rho"))
    sd = setting("sigma_data")

    type_x = torch.randn((bsz, nmax, 16), device=device, generator=generator)
    u_x = torch.randn((bsz, nmax, 3), device=device, generator=generator)
    z_x = torch.randn((bsz, 6), device=device, generator=generator)
    type_x = torch.where(real[..., None], type_x, torch.zeros_like(type_x))
    u_x = torch.where(dof_mask, map3(N, u_x), torch.zeros_like(u_x))
    z_x = torch.where(fm, z_x, torch.zeros_like(z_x))
    steps = karras_sigma_steps(num_steps=num_steps, sigma_min=sigma_min,
                               sigma_max=sigma_max, rho=rho, device=device)
    scales = {}
    for target in ("types", "coords", "lattice"):
        aa = float(setting("aa_rho_" + target, 0.0))
        if not math.isfinite(aa) or aa < 0:
            raise ValueError(f"aa_rho_{target} must be finite and nonnegative")
        if aa <= 0:
            continue
        astep = karras_sigma_steps(num_steps=num_steps, sigma_min=sigma_min,
                                   sigma_max=sigma_max, rho=aa, device=device)
        tm, am = steps[:-1], astep[:-1]
        if tm.numel() >= 2:
            scale = ((am[1:] - am[:-1]).abs() / (tm[1:] - tm[:-1]).abs().clamp_min(1e-12)).clamp_min(1.0)
            scale = torch.cat([scale, scale[-1:]], 0)
        else:
            scale = torch.ones_like(tm)
        scales[target] = scale

    def denoise(ty, u, z, sigma):
        state = {"type_noisy": ty, "u_noisy": u, "z_noisy": z}
        if denoise_fn is not None:
            return denoise_fn(state, sigma, skel, tb, objective=objective, sigma_data=sd)
        return denoise_orbit(model, state, sigma, skel, tb, objective=objective, sigma_data=sd,
                             autocast_dtype=setting("autocast_dtype"),
                             skip_type_scaling=setting("skip_type_scaling", False))

    def drift(ty, u, z, den, t, i):
        dt = (ty - den["D_type"]) / t
        residual = u - den["D_u"]
        if coord_target == "wrapped":
            residual = wrap_exact(residual, skel["row"], tb)
        du = residual / t
        dz = (z - den["D_z"]) / t
        if "types" in scales:
            dt = scales["types"][i] * dt
        if "coords" in scales:
            du = scales["coords"][i] * du
        if "lattice" in scales:
            dz = scales["lattice"][i] * dz
        return dt, torch.where(dof_mask, du, 0.0), torch.where(fm, dz, 0.0)

    type_next, u_next, z_next = type_x * steps[0], u_x * steps[0], z_x * steps[0]
    for i, (tc, tn) in enumerate(zip(steps[:-1], steps[1:])):
        gamma = (min(setting("S_churn") / num_steps, math.sqrt(2.0) - 1.0)
                 if tc >= setting("S_min") and tc <= setting("S_max") else 0.0)
        th = tc + gamma * tc
        ns = (th**2 - tc**2).sqrt()
        sn = setting("S_noise")
        tyh = type_next + ns * sn * torch.randn(type_next.shape, device=device, dtype=type_next.dtype, generator=generator)
        uh = u_next + ns * sn * map3(N, torch.randn(u_next.shape, device=device, dtype=u_next.dtype, generator=generator))
        zh = z_next + ns * sn * torch.randn(z_next.shape, device=device, dtype=z_next.dtype, generator=generator)
        uh = torch.where(dof_mask, uh, 0.0)
        if canonical_chart:
            uh = uh - torch.round(uh)
        zh = torch.where(fm, zh, 0.0)
        sh = torch.full((bsz,), float(th), device=device)
        den = denoise(tyh, uh, zh, sh)
        dt, du, dz = drift(tyh, uh, zh, den, th, i)
        type_next = tyh + (tn - th) * dt
        u_next = uh + (tn - th) * du
        z_next = zh + (tn - th) * dz
        u_next = u_next - torch.round(u_next)
        if i < num_steps - 1:
            snext = torch.full((bsz,), float(tn), device=device)
            den2 = denoise(type_next, u_next, z_next, snext)
            dt2, du2, dz2 = drift(type_next, u_next, z_next, den2, tn, i)
            type_next = tyh + (tn - th) * (0.5 * dt + 0.5 * dt2)
            u_next = uh + (tn - th) * (0.5 * du + 0.5 * du2)
            z_next = zh + (tn - th) * (0.5 * dz + 0.5 * dz2)
            u_next = u_next - torch.round(u_next)

    finite = ((torch.isfinite(type_next) | pad[..., None]).all(dim=(1, 2))
              & (torch.isfinite(u_next) | ~dof_mask).all(dim=(1, 2))
              & (torch.isfinite(z_next) | ~fm).all(dim=1))
    type_next = torch.where(real[..., None], type_next, 0.0)
    u_next = torch.where(dof_mask, u_next, 0.0)
    z_next = torch.nan_to_num(z_next, nan=0.0, posinf=10.0, neginf=-10.0)
    type_next = torch.nan_to_num(type_next, nan=0.0, posinf=10.0, neginf=-10.0)
    u_next = torch.nan_to_num(u_next, nan=0.0, posinf=1.0, neginf=0.0)
    z_next = lattice.end_clamp(z_next, skel["family"])
    type_next = type_next.clamp(-50.0, 50.0)
    return {"type": type_next, "u": u_next, "z": z_next, "finite": finite}
