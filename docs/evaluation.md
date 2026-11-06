# Evaluation

Orbitol scores generated crystals with an in-house protocol: relax them with NequIP, then count the stable or
metastable, unique and novel structures (MSUN) against the selected dataset's hull and training set. The paper's main
tables use LeMat-GenBench and a separate symmetry analysis.

## Protocol

At each `--sample-frequency` step of training, and in post-hoc evaluation, a pool is evaluated as follows:

1. Sample `--sample-count` structures (default 10,000) with the EMA weights.
2. Relax **all** of them with NequIP-OAM-L for `--thermo-relax-steps` FIRE steps (default 500), with a Frechet cell
   filter and unconstrained symmetry.
3. Score E_hull against the dataset's hull: stable means E_hull ≤ 0, metastable means E_hull ≤ 0.1 eV/atom. The
   headline stability rates are over all attempts; `*_rate_relaxed_ok` gives them over the returned relaxed structures.
4. Deduplicate the relaxed structures and check them for novelty against the training set. **SUN** = unique ∧ novel ∧
   stable and **MSUN** = unique ∧ novel ∧ metastable, both over all attempts.
   - `SUN_raw_UN` and `MSUN_raw_UN` combine relaxed stability with uniqueness and novelty judged on the raw samples.
   - `--sun-k` > 0 selects a subset protocol whose SUN/MSUN are not directly comparable with this one.

[metrics.md](metrics.md) defines every logged metric and its denominator.

## Post-hoc evaluation

Sample from a checkpoint, relax and score, with the protocol stored in the checkpoint:

```bash
python -m orbitol.posthoc_eval --kind orbitol --snapshot <checkpoint.pt> --out <dir> --sample-count 1000
```

`--stage sample` writes the pool only and `--stage evaluate` scores an existing pool in `--out`;
`--sampler-overrides` takes a JSON object such as `configs/paper/sampler_mp20_fast.json`. The metrics are written to
`<dir>/metrics.json`; `--wandb-name` also logs them to W&B. On Slurm, `slurm/posthoc_eval.sbatch` evaluates several
snapshots:

```bash
slurm/submit slurm/posthoc_eval.sbatch orbitol <out_root> <snapshot.pt> [<snapshot.pt> ...]
```

Optional environment variables override single settings, for example `EVAL_SAMPLE_COUNT`, `EVAL_RELAX_STEPS` or
`EVAL_NEQUIP_BACKEND`. Set `EVAL_WANDB_NAME` to log to W&B. The full list is in the script header.

## Evaluation during training

**Where evaluation runs** (`--eval-mode`):

| Mode | What happens |
|---|---|
| `async` (default) | Training writes a small EMA-only snapshot to `eval_snapshots/` and submits one GPU job (`slurm/eval_gpu.sbatch`) that samples and relaxes. That job deletes the snapshot and hands scoring and W&B logging to a CPU job (`slurm/eval_score.sbatch`). Training never waits, and no GPU sits idle between evaluations. If `--eval-backlog-max` (2) snapshots are already waiting, the evaluation is skipped and logged. |
| `sync` | Training evaluates in-process and waits. Use it for single-GPU runs and without Slurm. |

In async mode, results are written to `<run>/eval/step_XXXXXXX/` (`samples.pt`, `relaxed.csv.gz`, `metrics.json`,
`per_sample.csv.gz`) and logged to a companion W&B run `<run_name>-eval`, against `train_step`
([eval_jobs_contract.md](eval_jobs_contract.md)). In sync mode (the paper configs), the trainer saves the sample
tensors to `<run>/samples/step_XXXXXXX.pt` and the per-attempt relaxation records to
`<run>/samples/step_XXXXXXX.relaxed.jsonl`, and logs the metrics with the training metrics (W&B and
`<run>/metrics.jsonl`).

## Relaxation

The default relaxer is NequIP-OAM-L with OpenEquivariance kernels (`--nequip-backend oeq`). It runs batched FIRE with
a Frechet cell filter for a fixed number of steps (`orbitol/nequip_relax.py`). It is much faster than the
TorchScript model.

| Option | Values |
|---|---|
| `--nequip-backend` | `oeq` (default), `torchscript` |
| `--thermo-relax-steps` | 500 by default, as in the paper; a different step count can change relaxed endpoints and MSUN |
| `--nequip-relax-mode` | `batch` (default); `sequential` only with `torchscript` |

[evaluation_adapter.md](evaluation_adapter.md) describes the evaluator in detail.
