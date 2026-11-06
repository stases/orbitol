# DNG evaluation adapter

Implementation: orbitol/eval_adapter.py
and orbitol/symmetry_panel.py.
Tests: tests/test_eval_adapter.py.

The authority is the pinned train-time path:
$ORBITOL_CORE_ROOT/src/train_crystalite.py:315-410
and :695-720, which calls
$ORBITOL_CORE_ROOT/src/eval/sample_runtime.py:454-639.
The adapter verifies the core manifest before building its references. It uses
full train structures for novelty and full val structures for Wasserstein,
falling back to train only for an empty val cache. Missing caches are errors;
the adapter never preprocesses or downloads data. Counts in supplied samples
are preserved; empirical atom-count sampling belongs to the generator.

## SUN and MSUN

Default protocol (`sun_k=0`; `orbitol.eval_protocol.score`): every
constructed sample is relaxed. Relaxed endpoints are deduplicated and then
checked for novelty against the train structures (the core's
`compute_uniqueness_novelty`; the first attempt index wins). `sample_ema/SUN` and
`sample_ema/MSUN` count relaxed-unique ∧ relaxed-novel ∧ stable (E_hull ≤ 0) or
metastable (E_hull ≤ 0.1 eV/atom) endpoints, divided by all attempts, including
construction, relaxation and hull failures. `joint/SUN` and `joint/MSUN` are the
same numbers. Two variants keep the raw-sample U/N mask (the core's
novelty-then-dedup `is_un`): `sample_ema/SUN_raw_UN` and `sample_ema/MSUN_raw_UN`
use it with relaxed stability, and `sample_ema/SUN_oneshot` and
`sample_ema/MSUN_oneshot` use it with the raw structure's E_hull at its initial
energy (only when the relaxer reports that energy). They are logged as
`eval/SUN`, `eval/MSUN`, `eval/SUN_raw_UN`, `eval/MSUN_raw_UN`, `eval/SUN_oneshot`
and `eval/MSUN_oneshot` ([metrics.md](metrics.md)).

Legacy subset protocol (`sun_k > 0`, selected explicitly): the adapter
calls the pinned core's `compute_sun_metrics`, **not**
`compute_sun_msun_from_thermo_rates`. The former selects at most `sun_k` from
`novelty_metrics["un_structs"]` with seed `123 + step + sum(ord(c) for c in
"sample_ema")`. SUN and MSUN are products of the raw UN rate and the subset's
stability rate; the relaxed subset is already raw UN, unlike a random subset of
generated samples. No `un_style` option exists in this pin; its train-time path
explicitly uses `minimum_nary=1` and novelty-then-dedup UN. Here `joint/SUN` and
`joint/MSUN` count raw UN flags AND stable/metastable endpoint flags over the
relaxed subset only, so they are **conditional on UN** and are not unconditional
generation yields. They are logged under `eval_legacy/`
([eval_jobs_contract.md](eval_jobs_contract.md)).

`ThermoSettings` defaults: device `cuda`, batch size 128, 500 fixed FIRE
steps with the Frechet cell filter, MP2020-like corrections, and the OEQ backend.
Neither relaxer stops early on fmax; both always execute the configured number of
steps. `nequip_fmax=0.005` is passed to the sequential TorchScript relaxer and the
core's `_ThermoConfig`; the sequential relaxer also runs every step. The historical Crystalite baseline settings (TorchScript,
200 steps) remain selectable with `nequip_backend="torchscript"` and
`thermo_relax_steps=200`. The constructor's `device` does not override the
explicit `thermo_stability_device` field. There is no checkpoint/model_args
fallback for thermodynamics. The CLI sets the thermo device from its explicit
`--device`.

## Relaxation backends and failure isolation

orbitol/nequip_relax.py
builds the default OEQ relaxer (`make_fast_batch_relaxer`: torch-sim FIRE,
Frechet, artifact `ORBITOL_NEQUIP_OEQ`) or, with `nequip_backend="torchscript"`,
Crystalite's `NequIPRelaxer` (ASE FIRE, Frechet, fmax 0.005) or
`NequIPBatchRelaxer` (torch-sim `ase_fire`, Frechet) with artifact
`ORBITOL_NEQUIP_TS` (default `$ORBITOL_DATA_HOME/mlip/nequip_ts/NequIP-OAM-L.nequip.pth`).
Both artifacts are TorchScript files (`nequip_model_kind="torchscript"`);
`nequip_model_kind="aoti"` retains the old core factory and compile path.
The TorchScript path requires FIRE/Frechet and rejects other optimizer/filter
settings. An injected relaxer always takes precedence, without constructing an
MLIP. Initialization remains lazy until the thermo gate is reached. No core
logger, hull calculation, sample selection, or metric definition is replaced.

NequIP 0.17 supports this artifact in **both** integrations. The shared
loader in
<project environment>/lib/python3.12/site-packages/nequip/integrations/mixins.py
calls `load_compiled_model`, whose extension dispatch in
<project environment>/lib/python3.12/site-packages/nequip/model/inference_models/compiled.py
selects TorchScript for `.nequip.pth`. AOTI input/output targets apply only to
`.nequip.pt2`. The batch calculator uses its standard alchemiops neighbor list;
the ASE calculator uses its standard matscipy neighbor list. No artifact is
downloaded or recompiled. GPU energy parity of this batch path is measured
below; loader support alone does not establish it.

Local calculator subclasses preserve float64 ASE geometry before
neighbor finding, restore the caller's default dtype after loading/conversion,
make torch-sim row-vector cells contiguous, and refresh system membership on
every batch call. The last two avoid Warp layout errors and stale batch metadata
when a split changes system indices without changing the atom-number vector.
These are subclasses/composition; neither the core nor NequIP is monkeypatched.
The dtype context is intended for this serial evaluation process, not concurrent
calculator calls from Python threads.

`RobustRelaxer` bisects every failed batch down to singleton structures,
including unexpected Python exceptions. A failed singleton raises
`RuntimeError("non-finite relaxation (wrapped <Type>: <message>)")`, preserving
the original cause. The unchanged logger records the failure. Its subset
retries reuse results from the failed batch to avoid relaxing healthy structures
again; a fresh/full batch or changed relaxation options invalidates this cache.
Batch error text omits child-local offender indices, preventing accidental
removal of unrelated structures. Each singleton retains its own failure reason.
KeyboardInterrupt propagates. Native segfaults cannot be caught in-process.
Model-loading/configuration failures remain visible initialization errors.

## Relaxation speed

Inspection of
$ORBITOL_CORE_ROOT/src/utils/sample_stats.py
shows a post-step `get_forces()` check immediately followed by another force
request at the next FIRE step. ASE caches model outputs, but its Frechet filter
still computes nine matrix derivatives on each request. A private Frechet
subclass now reuses that force array until `set_positions` invalidates it. It
returns copies so optimizer code cannot modify the cached value. The model,
precision, timestep, step count, and convergence policy are unchanged.

The CPU test in
tests/test_nequip_relax.py
uses the actual core relaxer and ASE EMT on a four-atom Cu cell. Eight FIRE steps
give exactly identical final cell, coordinates and energy with the original and
cached filters. Matrix-derivative calls fall from 144 to 81. This isolates the
redundant filter work; it is not a NequIP GPU throughput measurement.

The TorchScript reference timing (13.7 s/structure) runs 200 steps even when a
structure has converged. Neither early stopping nor a lower step count is
introduced, because either would change the evaluated endpoint. NequIP already
sets a dynamic JIT fusion strategy in
<project environment>/lib/python3.12/site-packages/nequip/utils/global_state.py.
No additional global profiling-executor/fusion switches are changed in
production. Calculators and their loaded TorchScript modules persist across
structures. The post-hoc job uses one BLAS thread for small matrix operations.

The default relaxer is now the OpenEquivariance build of NequIP-OAM-L
(`--nequip-backend oeq`, batched FIRE, 500 steps), about 18× faster than the
TorchScript model; the TorchScript paths above remain selectable.
tools/bench_nequip.py
benchmarks a relaxer on a GPU. It reports load time separately, cold and repeated
per-call timing, seconds per structure, final energy differences in eV and eV/atom,
and failure counts. `--profile` writes CPU call profiles of model, neighbor-list,
filter and optimizer costs. `--jit-optimization off` isolates JIT overhead and
`--reference-filter` the filter cache. Profiling overhead affects wall timings, so
measure throughput without `--profile`. Batch/sequential endpoint agreement is
measured, not assumed; torch-sim and ASE are distinct FIRE/filter implementations.

TorchScript batch and sequential relaxation agreed to about 1e-6
eV/atom. A post-hoc evaluation of a Crystalite baseline snapshot (step 20k) took
38.8 minutes, including a one-off 7-minute symmetry-reference rebuild; post-hoc
evaluation now uses the prebuilt reference.

GPU benchmark command:

```bash
export ORBITOL_RUNS_ROOT="$(python -m orbitol.contracts ORBITOL_RUNS_ROOT)"
slurm/submit --cpus-per-task=1 --time=01:00:00 --job-name=nequip-bench slurm/gpu.sbatch tools/bench_nequip.py --n 24 --steps 200 --batch-size 24 --repeats 2 --include-garbage --profile --out $ORBITOL_RUNS_ROOT/nequip_bench
```

## Snapshot evaluation

For Crystalite snapshots (`--kind crystalite`), orbitol/posthoc_eval.py
invokes Crystalite's unchanged sampler as a subprocess, after binding
`orbitol.core`. It requires EMA weights and fixed snapshot sampler settings (150
steps, churn 60, S_noise 1.003, all three aa_rho values zero, nmax 20). Mismatched
settings fail explicitly. Other sampler values come from snapshot `model_args`
with the core's defaults; BF16 and chunk size follow the snapshot too. Atom
counts use the core's empirical training distribution and decoded elements use
training-set support. Existing raw splits and processed caches are required;
missing inputs never trigger preprocessing or downloads.

`--stage all|sample|evaluate` selects the two importable functions
`sample_stage` and `evaluate_stage`; `all` remains the default. Crystalite's
emitted list is stacked into the adapter's batched dict. Orbitol calls
`orbitol.sample.sample_from_checkpoint` and saves its entire result, including
rows, requested groups, skeletons, raw tensors, containment flags and failures.
Sampling uses EMA weights. Sample count, seed, `sun_k` and relaxation settings
default to the snapshot's training configuration (otherwise 10,000 attempts,
seed 123, `sun_k=0`, OEQ, 500 steps, batch 128); flags override them. In-training
evaluation, synchronous or asynchronous, samples with the same seed: the run's
`extra.sample_seed`, default 123, at every evaluation step.

For an output directory `OUT` (`--out`), the principal outputs are
`OUT/metrics.json`, `OUT/evaluation_settings.json` and `OUT/samples.pt`.
Settings record every thermo field, resolved model and snapshot paths with SHA256,
effective sampler settings, sampler/evaluator/SUN seeds, core root, device and
subset policy. Undefined scalar metrics are JSON null. The core sampler runs in
a private staging directory because its own writes are non-atomic.

Sampling commits the tensors and settings with temporary files, file
fsync, atomic replacement and directory fsync. It publishes `OUT/pool.json`
last. That manifest records kind, original snapshot path/hash/size/mtime_ns/inode,
checkpoint step (checked against the numbered filename), n, seed, sampler
settings, sample hash, sample-stage seconds and status `complete`. Evaluation
checks the sample hash and MLIP hash, then uses the pool alone; the snapshot can
already be deleted. Metrics are also committed atomically. Added timing keys
are `posthoc/sample_seconds` and `posthoc/evaluate_seconds`. Orbitol additionally
uses exactly training's `orbit/contains_a_rate` and `orbit/failures/<reason>`
assembly, including zero counts for the fixed failure vocabulary.

Evaluation also atomically replaces `OUT/relaxed_records.jsonl` before
committing the metrics. With the default protocol, evaluation writes
`OUT/relaxed.csv.gz` (one relaxation record per attempt: `index`, `constructed`,
`relax_status`, `reason`, `final_structure`, `energy_eV`, `nsteps` and relaxer
telemetry) and `OUT/per_sample.csv.gz`, and `relaxed_records.jsonl` holds the same
per-attempt scoring rows as `per_sample.csv.gz`: raw validity and raw U/N,
relaxation status, E_hull, stability, relaxed U/N, geometry and one-shot fields,
the SUN/MSUN flags of every variant, and raw and relaxed space groups.

With the legacy subset (`sun_k > 0`), `relaxed_records.jsonl` has one row
per attempted index of the relaxed subset, including failures: `pool_index`,
`is_un`, `raw_struct_valid`, `raw_comp_valid`, `raw_valid`, `relaxation_outcome`,
`failure_reason`, `failure_message`, `e_above_hull`, `energy_per_atom`,
initial/relaxed volume per atom, `relaxed_min_distance`, `relaxation_steps`,
`final_max_force`, and the pinned `stable`/`metastable` flags. Raw validity comes
directly from that sample's core evaluator Crystal; it is not inferred from its
relaxed endpoint.

The minimum distance is computed in numpy over the 27 nearest periodic
images of the LLL-reduced cell, including periodic self-images and coincident
atoms; pymatgen's neighbor list is not used. Energies are eV/atom, distances Å,
volumes Å³/atom, forces eV/Å. The OEQ relaxer reports initial energy, initial
and final atomic force maxima (`initial_fmax`, `final_max_force`) and the final
Frechet filter force maximum (`final_fmax`). The TorchScript relaxers report none
of these, so those fields are null, and the sequential backend may report zero
nsteps despite its manual step loop. Geometry/telemetry extraction failures are
listed in `diagnostic_errors` with null measurements.

The `diag/*` keys are variants, leaving all pinned/headline numbers
unchanged. Counted stable/metastable endpoints are suspect if hull energy is
< −0.5 eV/atom, PBC distance < 0.7 Å, or volume < 3 Å³/atom. Counts are
`diag/stable_suspect_count`, `diag/metastable_suspect_count`, and
`diag/suspect_{e_above_hull,min_distance,volume_per_atom}_count`; overlapping
criteria count independently, but each suspect is excluded only once.
`diag/SUN_plausible` and `diag/MSUN_plausible` exclude suspects from the joint
numerators. `diag/SUN_valid` and `diag/MSUN_valid` restrict those numerators to
raw-valid samples independently of the suspect test. All four divide by the
unchanged `joint/checked_count`, including failed attempts; they still require
`is_un`, the relaxed-endpoint flag by default and the raw flag in the legacy
subset. `diag/valid_checked_count` includes all raw-valid attempts, including
non-UN or failed ones. `diag/e_above_hull_median` and
`diag/e_above_hull_p05` use finite hull energies of successful relaxations.
Unknown measurements do not imply suspect; these heuristic variants are not
a guarantee of physical plausibility. The inherited pinned stable test has no
lower hull-energy bound, and the UN pool filters finite rather than valid
structures, motivating these additional views.

`--symmetry-reference` defaults to the prebuilt
`ORBITOL_MP20_SYMMETRY_REFERENCE` (default `$ORBITOL_DATA_HOME/mp20/symmetry_reference.json`).
Missing references fail immediately. Before constructing an evaluator, training's
reference validator rejects stale metadata or invalid histograms. The validated
histograms are pinned in memory so the symmetry panel cannot rebuild them inside
post-hoc evaluation. Both model kinds share the same evaluator and thermo settings.

`--wandb-name` enables W&B. A deterministic ID derived from project and
run name plus `resume="allow"` resumes the same evaluation run across snapshot
output directories. `wandb.define_metric("*", step_metric="train_step")` binds
charts to training steps. The entity is `ORBITOL_WANDB_ENTITY` (unset: your W&B default entity). Reusing a name
means reusing that evaluation series; a new series needs a new name. Offline
W&B operation does not provide server-side resume until logs are synced.

An already-sampled pool can be evaluated without any snapshot argument:

```bash
export ORBITOL_RUNS_ROOT="$(python -m orbitol.contracts ORBITOL_RUNS_ROOT)"
slurm/submit --job-name=pool-evaluate slurm/gpu.sbatch -m orbitol.posthoc_eval --kind orbitol --stage evaluate --out $ORBITOL_RUNS_ROOT/posthoc/RUN/step_0020000
```

## Resumable queue and retention

orbitol/posthoc_queue.py
uses only standard-library imports until actual sampling, evaluation or W&B
logging. `--plan` only reads directory entries, stats and JSON; it does not hash
large files, import torch, create locks, or write status. On a first visit it
correctly reports that a second unchanged poll is required. `--once` performs
one actual pass. `--dry-run` writes status and verified "would delete" audit
entries, but performs no sampling, evaluation, W&B writes or deletions.

Output roots default to `$ORBITOL_RUNS_ROOT/posthoc/<run name>`, where `<run name>`
is the name of the run directory passed with `--run`. Each numbered step has its own
subdirectory. An all-role pass observes only
regular, non-symlink files matching `^step_\d{7}\.pt$`, samples all eligible
unpooled snapshots oldest first, prunes, evaluates one pending pool in increasing
step order across runs, then logs completed metrics. Readiness requires age of
at least 120 seconds plus unchanged size, nanosecond mtime and inode across two
polls. Poll observations and attempt counts persist across job restarts.

A sampling failure is retried on later polls. After at least three
failures and snapshot age greater than 15 minutes, a pool-error record preserves
the traceback and file identity; that unchanged snapshot is kept and quarantined.
A changed file can become eligible again after another stability poll. Evaluation
failures are terminal for that pool: an error record prevents endless retries
while the queue processes other pools. Operators must resolve the cause and
clear the corresponding error marker to request another attempt.

`--keep-snapshots 1` protects each run's highest numbered step. Deletion
requires a committed pool, verified sample hash, matching snapshot size/mtime_ns/
inode, and matching snapshot SHA256. The newest-step and identity checks repeat
immediately before unlink. The queue never deletes resume states, end-of-run
aliases or temporary checkpoint names. Each run's audit log is `retention.log` in
its output root. A durable intent is appended before deletion, followed by a completed entry;
both contain time, absolute path, step, size and snapshot hash. An unwritable
audit log prevents deletion. Only the all-role process samples and prunes.

Main and helper workers claim each pool using an O_CREAT|O_EXCL lock with
host, pid, Slurm job ID and timestamp. Python success, failure and handled signals
release it. Restart reclaims a lock only when its local owner is dead or Slurm
confirms a remote job is absent; unverifiable locks are retained. The all-role
queue also has an exclusive ownership lock, ensuring a single retention/W&B
writer. `--role evaluate` samples nothing, deletes nothing and logs nothing to
W&B; it processes highest-step pending pools first. Each process creates at most
one DNGEvaluator, reused across both kinds. In-process Orbitol sampling releases
the sampling model's unused GPU memory before evaluation resumes.

W&B entity is `ORBITOL_WANDB_ENTITY` (unset: your W&B default entity); names are
`<run name>-posthoc`, with stable IDs and custom x-axis `train_step`. Config
includes the training run directory and training W&B ID when discoverable. The all
worker reserves each history index in a durable intent before sending metrics.
After interruption it compares the resumed W&B history step with that reservation,
avoiding a duplicate when the server accepted data before the local receipt was
written. Only after completion is the pool marked logged. Uncertain logging blocks
later history reservations for that run. Online W&B is required for remote resume;
disabled mode is reserved for tests. Do not write independently to these queue-owned
W&B runs.

Status is atomically refreshed at
$ORBITOL_RUNS_ROOT/posthoc/queue_status.json,
including snapshots, pool/evaluation/logging steps, error paths, current action
and accumulated stage times. SIGTERM and SIGUSR1 abandon the current uncommitted
stage, release locks and exit zero; committed pools/metrics remain reusable.
The queue exits after every run's final-step pool is evaluated and its work is
drained, or after the idle timeout with nothing pending. Failed pools remain
visible as errors and do not retry indefinitely.

The one-GPU loop does not impose a continuous two-checkpoint ceiling:
new snapshots can accumulate during a long, indivisible NequIP evaluation. The
next pass pools them all and restores the configured retention level. Unstable,
unpooled, changed, corrupt or quarantined snapshots are always retained, even if
this exceeds the storage target. A second evaluation worker increases throughput
but does not make the main worker's evaluation preemptible. A strict cap would
require a sampler-only worker or evaluation preemption, which this queue does not implement.

Launch the main queue with one `--run KIND:RUN_DIR` per training run
(`KIND` is `crystalite` or `orbitol`):

```bash
export ORBITOL_RUNS_ROOT="$(python -m orbitol.contracts ORBITOL_RUNS_ROOT)"
slurm/submit --job-name=posthoc-queue slurm/posthoc_queue.sbatch --run crystalite:$ORBITOL_RUNS_ROOT/BASELINE_RUN --run orbitol:$ORBITOL_RUNS_ROOT/ORBITOL_RUN
```

Start a second GPU helper using the same pool root and evaluation flags:

```bash
export ORBITOL_RUNS_ROOT="$(python -m orbitol.contracts ORBITOL_RUNS_ROOT)"
slurm/submit --job-name=posthoc-helper slurm/posthoc_queue.sbatch --role evaluate --run crystalite:$ORBITOL_RUNS_ROOT/BASELINE_RUN --run orbitol:$ORBITOL_RUNS_ROOT/ORBITOL_RUN
```

## Exact scalar key catalog

Every returned value is a float. Conditional keys are omitted exactly
where the core omits them. Undefined Orbitol rates/distances are NaN in Python
and null in CLI JSON. The following prefix/suffix sets enumerate every key;
failure-reason suffixes are supplied dynamically by the core.

**Crystalite**, prefix `sample_ema/`:

- `valid_rate`
- `comp_valid_rate`
- `struct_valid_rate`
- `wdist_nary`
- `wdist_density_mass`
- `unique_rate`
- `novel_rate`
- `un_rate`
- `SUN`, `MSUN` (thermo enabled and gate reached; relaxed-endpoint yields by
  default, the core's estimate in the legacy subset when a novelty result exists)
- `SUN_raw_UN`, `MSUN_raw_UN` (default protocol, thermo enabled)
- `SUN_oneshot`, `MSUN_oneshot` (default protocol, relaxer reports initial energy)

**Orbitol stability**, default protocol, prefix `sample_ema/stability/`:

- `attempt_count`, `constructed_count`, `construction_failure_count`
- `relaxed_ok_count`, `relax_failure_count`, `hull_ok_count`, `hull_failure_count`
- `stable_rate`, `metastable_rate` (over attempts)
- `stable_rate_relaxed_ok`, `metastable_rate_relaxed_ok`
- `e_above_hull_median`, `e_above_hull_p10`, `e_above_hull_p90`
- `e_above_hull_oneshot_median`, `e_above_hull_oneshot_p10`,
  `e_above_hull_oneshot_p90` (relaxer reports initial energy)

**Orbitol relaxation telemetry**, default protocol, prefix `sample_ema/relax/`
(only when the relaxer reports initial energy): `telemetry_count`, geometry and
force statistics, `converged_rate`, `sg_kept_rate` and `sg_kept_rate_msun`; the
full list is in [metrics.md](metrics.md) under `eval_relax/`.

**Thermo aggregates**, prefix `sample_ema/all_relaxed/` (default protocol,
all constructed samples) or `sample_ema/sun/` (legacy subset):

- `thermo_checked_total`
- `thermo_success_total`
- `thermo_failed_total`
- `thermo_divergence_total`
- `thermo_success_rate`
- `thermo_failure_rate`
- `thermo_stable_rate`
- `thermo_metastable_rate`
- `thermo_e_above_hull_mean` (at least one successful hull result)
- `thermo_fail/<reason>` (observed failures only; e.g. `relax_divergence`,
  `relax_non_finite`, `exception:<type>`, `hull_exception:<type>`,
  `mp2020_exception:<type>`, `nan_eabove`, `mp2020_removed`)

**Backend aliases**, under that same thermo prefix, followed
by `nequip_stability/` (or `chgnet_stability/` if explicitly configured):

- `checked_total`
- `success_total`
- `failed_total`
- `divergence_total`
- `success_rate`
- `failure_rate`
- `stability`
- `metastability`
- `e_above_hull_mean` (at least one successful hull result)

The core's standalone reference comparisons (`sample_ema/e_above_hull/*`
and the `reference_*` / `*_minus_reference` stability keys) are not emitted:
neither protocol calls the core's standalone thermo path.

**Orbitol extras**, prefix `joint/` (default: relaxed-endpoint U/N over all
attempts; legacy: raw U/N over the relaxed subset):

- `SUN`
- `MSUN`
- `SUN_count`
- `MSUN_count`
- `checked_count`
- `relaxed_count`
- `UN_count`

**Orbitol extras**, prefix `diag/`: the variants described under Snapshot
evaluation (`*_suspect_count`, `suspect_*_count`, `valid_checked_count`,
`SUN_plausible`, `MSUN_plausible`, `SUN_valid`, `MSUN_valid`,
`e_above_hull_median`, `e_above_hull_p05`).

**Orbitol extras**, each of the four prefixes
`sym/raw/0.1/`, `sym/raw/0.01/`, `sym/relaxed/0.1/`, `sym/relaxed/0.01/`:

- `count`
- `detected_count`
- `failure_count`
- `failure_rate`
- `p1_share`
- `crystal_system_tv`
- `space_group_jsd`
- `effective_groups`
- `crystal_system/triclinic`
- `crystal_system/monoclinic`
- `crystal_system/orthorhombic`
- `crystal_system/tetragonal`
- `crystal_system/trigonal`
- `crystal_system/hexagonal`
- `crystal_system/cubic`

**Orbitol extras**, only when requested_G is supplied, prefix
`sym/relaxed/0.1/`:

- `retention_count`
- `retention_total`
- `retention`
- `crystal_system_retention`

## Definitions and integration

Symmetry uses spglib at 0.1 and 0.01 Angstrom with angle tolerance 5
degrees. Detection failures occupy a separate count, never SG 1. P1 shares and
distributions condition on successful detection; failure rates use all input
indices. JSD uses natural logs; effective groups is exp(entropy). Empty or
entirely failed detected populations have undefined distributions. Retention
uses all available relaxed endpoints, counts failed detections as non-retained,
and tests equality or membership in the transitive inverse of pyxtal's maximal
t- and k-subgroup tables. This is group-type containment, not an embedding or
orientation certificate. Hull failures with a returned endpoint still appear
in the relaxed symmetry panel.

The train reference is computed once with the same token decoder and
spglib settings. Post-hoc stages and the queue require the prebuilt reference
at `ORBITOL_MP20_SYMMETRY_REFERENCE` (default `$ORBITOL_DATA_HOME/mp20/symmetry_reference.json`)
or an explicit `--symmetry-reference`. Metadata validates source path, size,
mtime, item count, decoder, spglib version and tolerances. These paths never
build or modify that reference. The generic adapter's optional cache-building
API remains available for separate preparation and small test fixtures.

`bootstrap_ci(values_fn, n, reps=1000, seed=0)` returns 95% percentile
bounds by resampling sample indices with replacement. For joint SUN, construct
one UN-and-stable flag per attempted index, inserting zero for failed attempts,
and return the mean of selected flags. For TV, pass selected detected SG numbers
to `crystal_system_tv` against the fixed reference histogram. The helper does
not redo novelty matching inside each bootstrap replicate. Undefined
replicates are not silently dropped.

## Core calls and deliberate adaptations

Direct calls made by the adapter, `orbitol.eval_protocol` and the symmetry
module:

- `orbitol.core.verify_core` (Orbitol's manifest guard).
- `src.utils.dataset.dataset_to_structures`.
- `src.data.mp20_tokens.tokens_to_structure`.
- `src.eval.dng_eval.compute_evaluator_metrics`.
- `src.eval.dng_eval.compute_novelty_metrics`.
- `src.eval.uniqueness_novelty.compute_uniqueness_novelty`, `_filter_by_nary` and
  `_is_finite_structure` (relaxed-endpoint U/N).
- `src.eval.stability._aggregate_thermo_records` (default-protocol thermo aggregates).
- `src.eval.stability.load_phase_diagram`.
- `src.utils.sample_stats.compute_e_above_hull_mp2020_like` and
  `compute_e_above_hull_uncorrected` (endpoint and one-shot hulls; legacy
  per-index records).
- Legacy subset only: `src.eval.dng_eval.compute_sun_metrics` (which calls
  `src.eval.stability._compute_thermo_metrics`), and
  `src.utils.stability_logger._ThermoConfig` and `StabilityLogger.__init__`,
  `StabilityLogger.update` through subclassing.

Indirect core metric calls include `tokens_to_eval_dict`,
`array_dict_to_crystal`, `_compute_wasserstein_metrics`, and in the legacy subset
the logger's original `_update_thermo`, backend factories and hull helpers. No
core metric is reimplemented; the relaxed-endpoint SUN/MSUN combine these core
helpers. The adapter reproduces only the scalar log assembly because
`evaluate_dng_sampling_batch` logs globally and returns only its best-checkpoint
MSUN payload. No core module, global function, or training-script state is
monkeypatched.

Deliberate adaptations: direct read-only token-cache loading instead of
MP20Tokens' potentially preprocessing constructor; lazy MLIP initialization at
the first eligible evaluation; omission of non-scalar W&B images
(`element_histogram` and `sample_vs_ref`); explicit computation even without a
live W&B session. The train-time logger exposes only mass-density and n-ary
Wasserstein distances, so the other two distances computed internally are not
added to its namespace. It does not log the diagnostic metrics or scalar
structure statistics available elsewhere in the core.

Default protocol: `score` computes each relaxed endpoint's hull once from
the saved relaxation records, plus the raw structure's one-shot hull when an
initial energy exists; nothing is relaxed twice. Failed relaxations remain in the
denominators.

Legacy subset: relaxation wrappers retain sample identity and the exact
returned endpoint, and return the original result unchanged to the core logger.
Since the logger's hull-recording function is a private closure, the adapter
recomputes the per-index hull value using the same core helper after
aggregation. Failed relaxations remain in joint denominators. The CPU injection
bypasses only MLIP construction; it still uses real core hull corrections and
failure aggregation.

Verification scope: the equivalence test uses 64 actual MP-20 validation
samples with full train novelty and full validation Wasserstein references.
Only its extra symmetry reference is a tiny fixture to avoid the full spglib
build on a login node. Separate tests exercise reference-cache creation/reuse,
batch and sequential relaxation capture, both hull methods, partial batches,
divergence retries, hull failures, original indices after invalid conversion,
seeded UN selection, hand-computed joint rates, rutile/perturbed symmetry,
bootstrap behavior, and CLI serialization. GPU nondeterminism and full-train
symmetry reference generation are not covered by these tests.
