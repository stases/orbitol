"""Known clean states imposed through a composable denoiser override."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from orbitol import lattice
from orbitol.edm import denoise_orbit, wrap_exact
from orbitol.sampler import orbit_edm_sampler


@dataclass(frozen=True)
class Constraint:
    """Optional mask/value pairs in padded sampler coordinates.

    Types have mask (B,K) and value (B,K,16); coordinates have mask/value
    (B,K,3); lattice entries have mask/value (B,6). True means fixed.
    """

    type_mask: torch.Tensor | None = None
    type_value: torch.Tensor | None = None
    u_mask: torch.Tensor | None = None
    u_value: torch.Tensor | None = None
    z_mask: torch.Tensor | None = None
    z_value: torch.Tensor | None = None

    def __post_init__(self):
        for block in ("type", "u", "z"):
            mask, value = getattr(self, block + "_mask"), getattr(self, block + "_value")
            if (mask is None) != (value is None):
                raise ValueError(f"{block}_mask and {block}_value must be supplied together")


def _validate(constraint, state, skel):
    for block in ("type", "u", "z"):
        mask, value = getattr(constraint, block + "_mask"), getattr(constraint, block + "_value")
        if mask is None:
            continue
        noisy = state[block + "_noisy"]
        shape = noisy.shape[:-1] if block == "type" else noisy.shape
        if mask.shape != shape or value.shape != noisy.shape:
            raise ValueError(f"{block} constraint shapes must match the sampler state")
        if mask.dtype != torch.bool:
            raise ValueError(f"{block}_mask must be bool")
        if mask.device != noisy.device or value.device != noisy.device:
            raise ValueError(f"{block} constraint must be on the sampler state's device")
        if block == "type":
            active = ~skel["pad_mask"]
        elif block == "u":
            active = skel["dof_mask"] & ~skel["pad_mask"][..., None]
        else:
            active = lattice.free_mask(skel["family"])
        if (mask & ~active).any():
            raise ValueError(f"{block}_mask selects padding or a frozen entry")


def constrain_denoiser(denoise_fn, constraint):
    """Replace masked predictions without changing the noisy network input."""
    validated = False

    def denoise(state, sigma, skel, tb, *, objective, sigma_data):
        nonlocal validated
        if not validated:  # One sampler run keeps its skeleton and state shapes; each check syncs the GPU.
            _validate(constraint, state, skel)
            validated = True
        # No in-place changes or retained outputs: the wrapped call may replay a CUDA graph.
        den = dict(denoise_fn(state, sigma, skel, tb, objective=objective, sigma_data=sigma_data))
        # Keep sampler dtypes: promoting a block would change subsequent randn draws.
        if constraint.type_mask is not None:
            den["D_type"] = torch.where(constraint.type_mask[..., None],
                                         constraint.type_value.to(den["D_type"]), den["D_type"])
        if constraint.u_mask is not None:
            u = state["u_noisy"]
            known = u - wrap_exact(u - constraint.u_value.to(u), skel["row"], tb)
            den["D_u"] = torch.where(constraint.u_mask, known, den["D_u"])
        if constraint.z_mask is not None:
            den["D_z"] = torch.where(constraint.z_mask, constraint.z_value.to(den["D_z"]), den["D_z"])
        return den

    return denoise


def constrained_edm_sampler(model, skel, tb, *, constraint, objective, settings,
                            generator=None, denoise_fn=None):
    """Run the unchanged EDM integration with known clean predictions.

    Constrained blocks require zero anti-annealing scale. The sampler's final
    clamps still apply, so fixed values must lie within its output bounds.
    """
    def setting(name, default=None):
        return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)

    for block, target in (("type", "types"), ("u", "coords"), ("z", "lattice")):
        if getattr(constraint, block + "_mask") is not None and setting("aa_rho_" + target, 0.0) > 0:
            raise ValueError(f"aa_rho_{target} must be zero for a constrained {block} block")
    if denoise_fn is None:
        def denoise_fn(state, sigma, skel, tb, *, objective, sigma_data):
            return denoise_orbit(model, state, sigma, skel, tb, objective=objective, sigma_data=sigma_data,
                                 autocast_dtype=setting("autocast_dtype"),
                                 skip_type_scaling=setting("skip_type_scaling", False))
    return orbit_edm_sampler(model, skel, tb, objective=objective, settings=settings,
                             generator=generator, denoise_fn=constrain_denoiser(denoise_fn, constraint))


def element_constraint(fixed_A0, pad_mask, encoding, allowed_mask=None) -> Constraint:
    """Encode fixed atomic numbers (B,K); zero leaves an orbit free."""
    if fixed_A0.ndim != 2 or fixed_A0.shape != pad_mask.shape:
        raise ValueError("fixed_A0 must have the same (B,K) shape as pad_mask")
    if fixed_A0.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("fixed_A0 must contain integer atomic numbers")
    mask = fixed_A0 != 0
    if (mask & pad_mask).any():
        raise ValueError("fixed_A0 selects padding")
    if ((fixed_A0 < 0) | (fixed_A0 > encoding.max_z)).any():
        raise ValueError(f"fixed_A0 must be between 0 and encoding.max_z={encoding.max_z}")
    if allowed_mask is not None:
        if allowed_mask.dtype != torch.bool or allowed_mask.shape != (encoding.vz,):
            raise ValueError(f"allowed_mask must be bool with length {encoding.vz}")
        if not allowed_mask.to(fixed_A0.device)[fixed_A0[mask].long() - 1].all():
            raise ValueError("fixed_A0 contains an element disallowed by allowed_mask")
    # The encoder requires positive atomic numbers at every non-padding position.
    value = encoding.encode_from_A0(fixed_A0, ~mask)
    return Constraint(type_mask=mask, type_value=value)


class ConstrainedSampler:
    """Build a tensor constraint for each pool chunk, then sample it."""

    def __init__(self, build):
        self.build = build

    def __call__(self, model, skel, tb, *, objective, settings, generator=None, denoise_fn=None):
        return constrained_edm_sampler(model, skel, tb, constraint=self.build(skel), objective=objective,
                                        settings=settings, generator=generator, denoise_fn=denoise_fn)


class FixedElements(ConstrainedSampler):
    """Select pool-aligned atomic numbers using the chunk's pool indices."""

    def __init__(self, fixed_A0, encoding, allowed_mask):
        def build(skel):
            selected = fixed_A0[skel["index"].to(fixed_A0.device)].to(skel["pad_mask"].device)
            return element_constraint(selected, skel["pad_mask"], encoding, allowed_mask)

        super().__init__(build)
