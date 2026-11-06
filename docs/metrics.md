# Metrics

These are the in-training diagnostics. In-house MSUN uses the selected dataset's hull and training-set
novelty.

W&B and the training JSONL log use `section/metric` names, with exactly one slash;
`step` and `train_step` are axes. Evaluation JSON and per-sample artifacts retain producer
names. Read old or new history rows with `orbitol.metric_names.canonical(row)`. Values,
units and denominators are unchanged. Missing measurements remain missing; decode zeros
are logged. Unknown keys become `other/<internal name with slashes replaced by __>` and
warn once. Collisions warn and retain the first value.

The following tables describe the all-attempt protocol (`sun_k=0`). Let A be all
attempts, C constructed structures, R returned relaxed endpoints (including hull failures),
and H endpoints with finite hull scores. Failures stay in A. Rates are fractions, not percent.

## Headline — `eval/`

| Key | Definition / denominator |
|---|---|
| `MSUN`, `SUN` | Relaxed E_hull ≤ 0.1 / ≤ 0 eV/atom, unique and novel **relaxed** endpoint / A; deduplication precedes novelty |
| `MSUN_raw_UN`, `SUN_raw_UN` | Relaxed E_hull with uniqueness/novelty on the **raw** structure / A |
| `MSUN_oneshot`, `SUN_oneshot` | Raw energy and raw-structure E_hull with raw U/N / A; relax-failed attempts count as not one-shot stable |
| `best_MSUN` | Highest finite in-process MSUN seen by the trainer |
| `{M}SUN_valid`, `{M}SUN_plausible` | Corresponding relaxed yield requiring raw validity / excluding suspect endpoints, over A |
| `stable_rate`, `metastable_rate` | E_hull ≤ 0 / ≤ 0.1 eV/atom, irrespective of U/N, over A |
| `valid_rate`, `comp_valid_rate`, `struct_valid_rate` | Raw joint / composition / geometry validity, over A |
| `unique_rate`, `novel_rate`, `un_rate` | Pinned core raw U/N flags over its finite, constructable, n-ary eligible structures |
| `wdist_nary`, `wdist_density_mass` | Wasserstein distances of raw n-ary count / mass density to reference; core sample filtering, no rate denominator |

## Relaxation — `eval_relax/`

Per relaxed-ok structure, from the relaxer's initial and final state and the exact structure it relaxed
(`orbitol/relax_geometry.py`, `eval_protocol._relax_metrics`). Statistics skip missing values; runs or records
without relaxer telemetry emit none of these keys.

| Quantity | Definition / units |
|---|---|
| `rms_disp`, `max_disp` | Raw→relaxed atomic displacement in Å, in the relaxed cell, minimum image, mean translation removed; RMS / maximum over atoms |
| `strain` | ‖½(FᵀF − I)‖_F, Green–Lagrange strain of the cell |
| `volume_change_pct` | Signed 100(V_relaxed/V_raw − 1) |
| `energy_drop_meV` | 1000(E_raw − E_relaxed)/atom, meV/atom |
| `initial_fmax`, `final_fmax` | eV/Å; max atomic force on the raw structure / Frechet filter fmax (atoms and cell) at the end |

| Key | Pool |
|---|---|
| `{quantity}_{median,mean,p90}` | All relaxed-ok structures, for every quantity above |
| `{rms_disp,max_disp,strain,volume_change_pct,energy_drop_meV}_median_msun` | MSUN structures |
| `{rms_disp,strain,energy_drop_meV}_median_sg_kept`, `..._median_sg_changed` | Raw and relaxed space group (symprec 0.1) detected and equal / different |
| `converged_rate` | Final filter fmax ≤ 0.05 eV/Å, over relaxed-ok structures with a final fmax |
| `sg_kept_rate`, `sg_kept_rate_msun` | Space group kept, over relaxed-ok (or MSUN) structures with both groups detected |
| `telemetry_count` | Relaxed-ok structures with a finite initial energy |

## Hull — `eval_hull/`

| Key | Definition / denominator |
|---|---|
| `e_above_hull_{mean,median,p05,p10,p90}` | Relaxed E_hull distribution, eV/atom, over H |
| `e_above_hull_oneshot_{median,p10,p90}` | Raw E_hull distribution for successful one-shot scores |
| `{meta}stable_rate_relaxed_ok` | Stable / metastable endpoints divided by R, including hull failures in R |

## Training — `train/`

| Key | Definition / denominator |
|---|---|
| `loss_total`, `loss_{type,coord,lattice}` | Optimisation objective and weighted components; current minibatch, core objective reductions |
| `lr`, `sigma_mean`, `sigma_std`, `grad_scale` | Optimizer learning rate; minibatch noise-level mean/std; AMP scale when enabled |
| `type_acc`, `coord_l2`, `lattice_l2` | Correct species / real tokens; squared coordinate error sum / real tokens; mean squared lattice error |

## Validation — `val/`, `val_sigma/`, `val_coord_class/`

| Key | Definition / denominator |
|---|---|
| `val/loss_*` | Mean over validation batches: weighted total, unweighted type/coordinate/lattice components |
| `val_sigma/{loss_type,loss_coord,loss_lattice,coord_do_nothing}_s{s}` | Weighted losses / do-nothing coordinate reference, averaged over validation batches at fixed sigma |
| `val_sigma/coord_ratio_s{s}` | Coordinate loss / do-nothing reference at sigma s |
| `val_coord_class/{d1,d2,d3,wrap,nondiag}_s{s}` | Same ratio restricted to that coordinate class; classes can overlap |

## Perf — `perf/`

| Key | Definition / denominator |
|---|---|
| `steps_per_second`, `samples_per_second` | Completed training steps / samples per elapsed logging interval |
| `gpu_memory_{allocated,reserved}_gb`, `lll_cap_exceedances` | GPU bytes / 1e9; recorded LLL iteration-cap counter |
| `eval_seconds`, `nequip_eval_seconds`, `posthoc_{sample,evaluate}_seconds` | Wall-clock seconds for the named operation |

## Counts — `eval_count/`

| Key | Definition (absolute counts; no denominator) |
|---|---|
| `attempts`, `constructed`, `relaxed_ok`, `hull_ok` | A, C, R, H |
| `construction_failures`, `relax_failures`, `hull_failures` | A−C, C−R, R−H |
| `SUN`, `MSUN`, `relaxed_UN` | Relaxed endpoint SUN, MSUN and U∧N numerators |
| `raw_valid` | Raw-valid checked structures (constructed checks in diagnostic producer) |
| `suspect_stable`, `suspect_metastable` | Stable/metastable endpoints meeting any suspect criterion, independent of U/N |
| `suspect_low_hull`, `suspect_min_distance`, `suspect_volume` | Stable or metastable endpoints with hull < −0.5 eV/atom, distance < 0.7 Å, volume < 3 Å³/atom; criteria overlap |
| `relax_fail_{reason}` | Core thermo failures by reason, including hull failures |
| `relax_divergence_total` | Force-divergence **plus non-finite** relaxation failures; not a copy of `relax_fail_relax_divergence` |

## Symmetry — `sym_raw/`, `sym_relaxed/`

| Key | Definition / denominator; suffix tolerance is 0.01 or 0.1 Å |
|---|---|
| `p1_share_{tol}`, `cs_{system}_{tol}` | P1 / seven crystal-system shares, conditional on successful detection |
| `space_group_jsd_{tol}`, `crystal_system_tv_{tol}` | SG Jensen–Shannon divergence (natural logs) / crystal-system total variation to reference, conditional on detection |
| `effective_groups_{tol}`, `failure_rate_{tol}` | exp(SG entropy); detection failures / A (raw) or R (relaxed) |
| `retention_{tol}`, `cs_retention_{tol}` | Relaxed SG is requested SG or supergroup / same crystal system as requested, over R; currently emitted at 0.1 only |
| `count_{tol}` | Input count retained only when the usual attempt/relaxed count is unavailable |

## Decode — `decode/`

| Key | Definition / denominator |
|---|---|
| `decode/fail_{reason}`, `decode/contains_a_rate` | Decode failures per reason, including zeros; containment-pass fraction over sampled skeletons |

## Status — `status/`

| Key | Definition (metadata; no denominator) |
|---|---|
| `status/eval_error`, `status/eval_submit_failed`, `status/eval_skipped_backlog` | Evaluation exception / submission failure / backlog skip indicators |
| `status/eval_snapshot_step`, `status/eval_job_id`, `status/pool_sha256` | Snapshot step, Slurm job identifier, pool fingerprint; metadata, no denominator |

Duplicate joint yields/counts, backend thermo copies, constructed-denominator
thermo rates, and reconstructable symmetry counts are omitted. The diagnostic median and
symmetry input counts survive when their twins are absent. With `sun_k>0`, joint yields and
checked/relaxed counts survive as `eval_legacy/joint_*`; other historical subset thermo
aggregates use `eval_legacy/sun_*`. Legacy joint yields divide by checked subset size;
legacy headline yields estimate full-pool raw-UN yield. They are not comparable to the
current protocol. `canonical()` conservatively treats joint rows lacking full-pool attempt
counts as legacy; a row alone cannot recover missing configuration.

## Main renames and saved view

| Old internal name | Logged name |
|---|---|
| `loss/total`, `lr` | `train/loss_total`, `train/lr` |
| `sample_ema/MSUN`, `sample_ema/MSUN_raw` | `eval/MSUN`, `eval/MSUN_raw_UN` |
| `orbit/best_msun`, `diag/MSUN_valid` | `eval/best_MSUN`, `eval/MSUN_valid` |
| `sample_ema/stability/e_above_hull_median` | `eval_hull/e_above_hull_median` |
| `val/sigma_0.1/coord_ratio/d1` | `val_coord_class/d1_s0.1` |
| `sym/raw/0.01/crystal_system/cubic` | `sym_raw/cs_cubic_0.01` |

The saved-view builder is `tools/wandb_workspace.py`.
Set `ORBITOL_WANDB_ENTITY` in the local file (or pass `--entity YOUR_ENTITY`), then run with
`--project orbitol_alex_mp20 --name Orbitol --dry-run` to preview JSON offline; omit `--dry-run` to save a view. Headline panels include old internal aliases.
The sections are Headline, Relaxation (open), Relaxation detail (closed), Hull, Training,
Validation (open), then Perf, Counts, Symmetry, Decode, Status (closed). Validation groups seven sigmas per metric;
symmetry groups two tolerances per metric. Open-ended failure reasons use regex panels.
