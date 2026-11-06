# Constrained sampling

`orbitol/constrained.py`
wraps the denoiser to fix selected element codes, chart coordinates, or free lattice
parameters. The model still receives the current noisy state. Masked predictions are
replaced by their known clean values; coordinates use
`u_noisy - wrap_exact(u_noisy - u_value, row, tb)` to select the nearest periodic image.
The model, packed adapter, training, and EDM integration are unchanged.

Public API:

```python
Constraint(type_mask=None, type_value=None, u_mask=None, u_value=None,
           z_mask=None, z_value=None)
constrain_denoiser(denoise_fn, constraint)
constrained_edm_sampler(model, skel, tb, *, constraint, objective, settings,
                        generator=None, denoise_fn=None)
element_constraint(fixed_A0, pad_mask, encoding, allowed_mask=None)
ConstrainedSampler(build)
FixedElements(fixed_A0, encoding, allowed_mask)
```

`Constraint` is frozen. Supply each boolean mask with its value tensor on
the sampler device: types `(B,K)` / `(B,K,16)`, coordinates `(B,K,3)` / `(B,K,3)`,
lattice `(B,6)` / `(B,6)`. Values are cast to the sampler's precision to preserve
its random draws. Omitted blocks are free. Coordinates are centered chart
coordinates, and lattice values are the sampler's six-slot latent, not lengths and
angles. Masks on padding, frozen chart axes, or dependent lattice slots raise
`ValueError`. A supplied mask requires the corresponding `aa_rho_types`,
`aa_rho_coords`, or `aa_rho_lattice` to be zero, including an all-False mask.

`ConstrainedSampler(build)` calls `build(skel)` for each pool chunk. Its call
signature is `(model, skel, tb, *, objective, settings, generator=None,
denoise_fn=None)`. `FixedElements` uses the same interface and selects rows of a
pool-aligned `(n,K)` integer tensor using `skel["index"]`. Zero means free; positive
atomic numbers must be encodable, allowed by the decoder, and on real orbits.
It never writes `Skeleton.A0`.

Example: fix the second orbit to nitrogen wherever that orbit exists. Pool
positions refer to the bank draw with the same `n` and `seed`.

```python
import orbitol.core as core
import numpy as np
import torch
from orbitol import contracts as C
from pathlib import Path
from orbitol.constrained import FixedElements
from orbitol.sample import isolated_rng, load_snapshot, sample_pool

core.verify_core()
with isolated_rng():
    inputs, metadata = load_snapshot(
        Path(C.local_path("ORBITOL_RUNS_ROOT")) / "RUN/eval_snapshots/step_0020000.pt",
        device="cpu",
    )
    n, seed = 16, 123
    skeletons = inputs["bank"].sample(n, generator=np.random.default_rng(seed))
    fixed_A0 = torch.zeros(n, inputs["cfg"].K, dtype=torch.long)
    for i, skeleton in enumerate(skeletons):
        if len(skeleton.rows) >= 2:
            fixed_A0[i, 1] = 7
    sampler = FixedElements(fixed_A0, inputs["encoding"], inputs["allowed_mask"])
    result = sample_pool(**inputs, n=n, seed=seed, device="cpu", sampler=sampler)
    result["metadata"].update(metadata)
```

With a fixed prediction `c`, the Euler predictor satisfies
`x' - c = (x - c) * t' / t`. The Heun corrector sees the same slope, so it gives
the same update. Churn adds the usual trajectory noise; no extra random draws or
resampling are introduced. The final step to zero noise lands on `c` (coordinates:
on its periodic image), up to floating-point error. Positive anti-annealing scales
would rescale that slope and invalidate the argument.

Empty constraints and all-False masks preserve all outputs bitwise. The
default pool sampler and its random stream remain unchanged, including chunking.
Fixed values remain subject to the existing final clamps: type features in
`[-50,50]`, lattice slots 0/2/5 in `[-5,5]`, and slots 1/3/4 in `[-10,10]`.
The element builder's unit-norm codes lie within those bounds.

The trajectory carries its own noise around the known value. Exact
standard-normal forward marginals are not guaranteed: initial states are drawn
around zero rather than `c`, and `S_noise` can differ from one. The recurrence is
exact for an unwrapped fixed entry; coordinate wrapping can switch periodic images.
Free entries see constraints only through noisy inputs, so the effect is mainly at
low noise. There is no resampling, and the result is not an exact conditional sample.
