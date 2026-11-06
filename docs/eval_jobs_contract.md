# Orbitol evaluation: protocol and job contract

Paths, names, signatures and file formats here are fixed.

## Protocol
- Each evaluation samples `sample_count` (default 10,000) structures with packed sampling (auto token budget) from the
  EMA weights.
- **Every constructed sample is relaxed** with the fast relaxer: `orbitol.nequip_relax.make_fast_batch_relaxer`,
  OEQ artifact `DEFAULT_OEQ_PATH` (`ORBITOL_NEQUIP_OEQ`), batched FIRE with the Frechet cell filter for exactly
  `thermo_relax_steps` steps (default **500**; no early stopping), through `RobustRelaxer` with `steps` passed
  explicitly. `--nequip-backend` (`oeq` or `torchscript`), `--thermo-relax-steps` and `--nequip-relax-mode` select
  the relaxer.
- **Stability over all relaxed structures** (E_hull with the dataset's hull and corrections, as before): stable =
  E_hull ≤ 0; metastable = E_hull ≤ 0.1. Logged under `<tag>/stability/*`, with explicit denominators (over attempts
  and over relaxed_ok), plus E_hull median/p10/p90 and failure counts.
- **Headline SUN/MSUN (`<tag>/SUN`, `<tag>/MSUN`) use relaxed endpoints:**
  - uniqueness by deduplicating the relaxed endpoints of successfully relaxed samples, with the core's
    StructureMatcher conventions (`src/eval/uniqueness_novelty.py`, the same settings as before); first index
    wins;
  - novelty of relaxed endpoints against the dataset's novelty reference (the same reference as before);
  - SUN = relaxed-unique ∧ relaxed-novel ∧ stable; MSUN = relaxed-unique ∧ relaxed-novel ∧ metastable (inclusive);
  - denominator = `sample_count` attempts.
- **Raw-based SUN/MSUN** (the earlier definition: raw novelty-then-dedup `is_un` mask ∧ stability) is logged as
  `<tag>/SUN_raw_UN`, `<tag>/MSUN_raw_UN`. Raw validity and U/N rates are unchanged.
- `<tag>` is the existing evaluator tag (e.g. `sample_ema`).
- Legacy subset protocol (`sun_k > 0`, the TorchScript backend, 200 steps) remains selectable only explicitly, for
  historical comparison.

## Modes (`TrainConfig.eval_mode`)
- **`"async"` (default):** at each `sample_frequency` step, training writes an eval snapshot and submits a GPU eval
  job; training never waits. If `eval_backlog_max` (default 2) or more eval snapshots are already pending on disk,
  it skips this evaluation and logs `orbit/eval_skipped_backlog=1`.
- **`"sync"`:** training evaluates in-process (same protocol, same functions) and waits. Use it for single-GPU runs.

## Files and layout, under the run directory `RUN` (the trainer's run dir)
- **Eval snapshot:** `RUN/eval_snapshots/step_<step:07d>.pt`, written atomically (temp file plus `os.replace`). It
  holds only what sampling needs: the trainer's snapshot minus optimizer, scheduler, scaler and data-sampler state
  (EMA weights, model config, step, plus whatever `orbitol.sample`'s loader reads). The GPU stage deletes it after
  `relaxed.csv.gz` is complete.
- **Stage directory** `RUN/eval/step_<step:07d>/`:
  - `samples.pt`: raw sample tokens plus `requested_G`, in the format `DNGEvaluator.evaluate` consumes.
  - `relaxed.csv.gz`: one row per attempt index: `index, constructed, relax_status (ok|failed|not_constructed),
    reason, final_structure (pymatgen JSON), energy_eV, nsteps`, plus relaxer telemetry (`initial_energy_eV,
    initial_fmax, final_fmax, final_max_force`; empty for the TorchScript relaxers).
  - `gpu.done.json`: timings, job id, versions.
  - `metrics.json`: the full metric dict, written by the scoring stage.
  - `per_sample.csv.gz`: index, raw flags, stability, E_hull, relaxed U/N, raw U/N, SUN/MSUN flags, space groups.
  - `score.done.json`.
- **Retention:** keep the stage directory's small files (metrics, per_sample, relaxed.csv.gz, samples.pt, which are
  tens of MB) and delete the eval snapshot. No full trainer snapshots are created for evaluation.

## Python interfaces
- **`orbitol/eval_protocol.py`**, pure functions shared by sync and async:
  - `relax_all(items, relaxer, steps) -> list[dict]`: one record per attempt index (constructed or not), with
    final structure, energy and status; the `RobustRelaxer` contract.
  - `score(items, relaxed_records, *, references, ppd, settings, tag, step, requested_G=None) -> (metrics: dict,
    per_sample: list[dict])`: raw validity and U/N, stability, relaxed and raw SUN/MSUN, joint metrics and the
    symmetry panel. `references`, `ppd` and `settings` are whatever `DNGEvaluator` holds.
  - `DNGEvaluator.evaluate` (sync) = sample → `relax_all` → `score`.
- **`orbitol/contracts.py`**: `normalize_historical_config(cfg_dict) -> dict` fills fields
  missing from legacy checkpoint configs with their historical meaning (TorchScript backend, 200 steps,
  sync, `sun_k`/`sample_count` as stored), so old snapshots load unchanged. posthoc_eval and sample loading call
  it.
- **`orbitol/eval_snapshot.py`**:
  - `write_eval_snapshot(trainer, run_dir, step) -> Path`
  - `pending_eval_snapshots(run_dir) -> list[Path]`
  - `load_eval_snapshot(path)`: what `eval_gpu` needs to build the sampler.
- **`orbitol/eval_jobs.py`**:
  - `submit_eval(run_dir, step, cfg) -> str` (Slurm job id) runs
    `slurm/submit slurm/eval_gpu.sbatch --run-dir RUN --step STEP`, with partition/account/gres from
    `cfg.eval_partition` and `cfg.eval_account` (both default empty), and `cfg.eval_gres` (default
    `gpu:1`). Empty placement fields defer to `ORBITOL_SLURM_GPU_PARTITION` /
    `ORBITOL_SLURM_GPU_ACCOUNT` in the local configuration, then cluster defaults.
  - Submission failure logs `orbit/eval_submit_failed=1` and does not crash training.
- **CLIs**:
  - `python -m orbitol.eval_gpu --run-dir RUN --step STEP`: load snapshot → sample → `relax_all` → write `samples.pt`,
    `relaxed.csv.gz` and `gpu.done.json` → delete the snapshot → submit
    `slurm/submit slurm/eval_score.sbatch --run-dir RUN --step STEP` → exit.
  - `python -m orbitol.eval_score --run-dir RUN --step STEP`: `score` → `metrics.json`, `per_sample.csv.gz` → W&B log
    to `ORBITOL_WANDB_ENTITY` (unset: your W&B default entity), the run's project, companion run `<wandb_name>-eval`, axis `train_step` → `score.done.json`.
  - Both are idempotent: a stage with its `.done.json` exits immediately; reruns overwrite partial outputs.
- **Slurm wrappers**:
  - `slurm/eval_gpu.sbatch`: configured GPU placement, 1 GPU, ~1.5 h, `source slurm/oeq_env.sh`, `$ORBITOL_PYTHON`.
  - `slurm/eval_score.sbatch`: `ORBITOL_SLURM_CPU_PARTITION` / `ORBITOL_SLURM_CPU_ACCOUNT`,
    16 CPUs, ~1 h. CPU jobs do not inherit GPU placement.
  - Both pass their args through to the CLI.

All submissions, including nested jobs, go through `slurm/submit`; log destinations use
`ORBITOL_SLURM_LOG_DIR`. Interpreter/toolkit settings come from the local file. Data references
resolve through `C.DATA_LAYOUT` under `ORBITOL_DATA_HOME`: `tables/v1`, `<dataset>/processed`,
`<dataset>/orbit_cache`, `<dataset>/symmetry_reference.json`, `mlip/nequip_oeq/NequIP-OAM-L-oeq-ts.nequip.pth`,
and the dataset hull (`hulls/mp20/2023-02-07-ppd-mp.pkl` or `hulls/alex_mp2020/variants/v1.pkl`).
Individual overrides use `ORBITOL_TABLES_DIR`, `ORBITOL_NEQUIP_OEQ` and
`ORBITOL_{MP20,ALEX_MP20}_{DATA_ROOT,ORBIT_CACHE_DIR,SYMMETRY_REFERENCE,PHASE_DIAGRAM}`.
