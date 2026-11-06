# Running Orbitol on different GPUs

2026-10-12, torch 2.8.0+cu128, Triton 3.4.0, OpenEquivariance 0.5.4. Model: d512,
14 layers, batch 128, Alex-MP-20 (`--dataset alex_mp20 --objective posterior_periodic --gem-mode distance
--loss-weights 0.1 50 5`).
Configure `ORBITOL_SLURM_LOG_DIR` for logs named `<job name>-<job id>.out`.

## Which configuration to use

| GPU class | Examples | Training | Sampling a bf16 checkpoint | Relaxation |
|---|---|---|---|---|
| Ampere and newer (sm_80+) | RTX 6000 Ada, RTX 3090, RTX A5000 | defaults (bf16, `reduce-overhead`, packed 8,192) | defaults | `oeq`, shipped artifact |
| Turing (sm_75) | Titan RTX | `--precision fp16` (fastest) or `--precision fp32` | `--precision fp16` or `fp32` | `oeq` with the sm_75 artifact |
| Pascal (sm_61) | GTX 1080 Ti | does not run | does not run | does not run |

Configure `ORBITOL_SLURM_GPU_PARTITION` and `ORBITOL_SLURM_GPU_ACCOUNT` in the local file
for the target GPU class, then submit from the checkout. Cluster-specific GPU resource names can
be passed with `--gres` before the script and `--eval-gres` after it for asynchronous children.

```bash
slurm/submit slurm/train_orbitol.sbatch RUN \
    --dataset alex_mp20 --objective posterior_periodic --gem-mode distance --loss-weights 0.1 50 5
```

Turing has no bf16. Use fp16 or fp32 and rebuild OEQ on that GPU; set `ORBITOL_NEQUIP_OEQ`
to its artifact in the local file. The shipped sm_89 artifact requests too much shared memory on Turing.

```bash
slurm/submit slurm/build_nequip.sbatch --only oeq --force
slurm/submit slurm/train_orbitol.sbatch RUN \
    --dataset alex_mp20 --objective posterior_periodic --gem-mode distance --loss-weights 0.1 50 5 \
    --precision fp16
```

Sampling a checkpoint with execution choices other than the stored ones (`orbitol/sample.py`): the weights are
the snapshot's EMA weights; only precision, compilation and packing change, and the choice is recorded in the
pool's `metadata["overrides"]`. Omitted flags keep the snapshot's values. The same mechanism also overrides the
integration settings `--sample-num-steps`, `--S-churn`, `--S-min`, `--S-max` and `--S-noise`.

```bash
$ORBITOL_PYTHON -m orbitol.sample SNAPSHOT.pt --out POOL.pt --n 10000 --precision fp16   # Titan RTX
$ORBITOL_PYTHON -m orbitol.sample SNAPSHOT.pt --out POOL.pt --n 10000 \
    --precision fp32 --compile-trunk none --sample-token-budget 0                       # no Triton at all
```

fp16 training uses PyTorch's dynamic loss scaling. On a gradient overflow the step is skipped and the scale
halved; `perf/grad_scale` is logged at every `--log-every`. A scale that keeps falling means the forward pass
itself overflows; switch that run to fp32.

## Throughput

### Training

800 steps without validation or evaluation (`--sample-count 0 --val-every 0 --ckpt-every 0
--nequip-eval false --no-wandb`). Steps/s is the median over the 50-step logging windows from step 150 on, with the
range in brackets. The Ada shared its node with four training runs (about 31 steps/s reported for those runs).
Time to first step is the first window minus 49 steady-state steps: compilation and CUDA-graph
recording with a cold Inductor cache for that GPU type, unless marked warm. Memory is the largest
`perf/gpu_memory_reserved_gb` at the logging points (not a peak). Values separated by semicolons come from separate
jobs.

| GPU | Precision | `compile_trunk` | Steps/s | vs Ada | Time to first step | Reserved GB |
|---|---|---|---:|---:|---:|---:|
| RTX 6000 Ada | bf16 | reduce-overhead | 29.3 [28.2, 30.9] | 1.00 | 7 s (warm) | 2.8 |
| RTX 3090 | bf16 | reduce-overhead | 15.4 [14.5, 15.8] | 0.53 | 144 s | 2.8 |
| RTX 3090 | fp32 | reduce-overhead | 12.9 [12.0, 13.4] | 0.44 | 117 s | 3.6 |
| RTX 3090 | bf16 | none | 9.3 [9.0, 10.1] | 0.32 | 8 s | 2.9 |
| RTX A5000 | bf16 | reduce-overhead | 13.5 [12.4, 14.9] | 0.46 | 139 s | 2.8 |
| Titan RTX | fp16 | reduce-overhead | 14.1 [13.3, 14.3]; 12.5 [12.0, 13.4] | 0.48; 0.43 | 151 s; 13 s (warm) | 2.8 |
| Titan RTX | fp32 | reduce-overhead | 10.1 [9.5, 10.9]; 7.4 [7.0, 9.5]; 7.7 and 8.2 | 0.25-0.35 | 129 s; 11-20 s (warm) | 3.6 |
| Titan RTX | fp32 | none | 7.9 [6.6, 8.7] | 0.27 | 12 s | 3.9 |
| Titan RTX | bf16 | any | fails, see limitations | | | |
| GTX 1080 Ti | fp32 | none, reduce-overhead | fails at the first kernel | | | |

fp16 against fp32 on Turing, paired: one job on one Titan RTX trained fp32, fp16 and fp32 again
in sequence, 800 steps each: 7.7, 12.5 and 8.2 steps/s, so fp16 is about 1.6x fp32. The fp16 scale fell from 65,536
to 256 in the first 50 steps (8 skipped updates) and then stayed there to step 800, in this job and in the separate
fp16 job.
nvidia-smi reported power-cap and thermal slowdown on that GPU (throttle reasons 0x4 and 0x4C), which is the likely
reason the separate Titan RTX jobs differ by up to 35%.

Losses at step 800 agree across GPUs and precisions (50.8-51.2; same seed and data order).

### Sampling

A benchmark on the step-200k EMA snapshot of an Alex-MP-20
run with the recipe above (bf16-trained): 150 Heun steps, seed 123, a 256-crystal warm-up and then one timed
`sample_pool` call for 2,048 crystals, including skeleton drawing, decoding and transfers. Every pool was finite
and decoded all 2,048 crystals, except one coinciding-orbit failure on the A5000. With T = 8,192 a 2,048-crystal
pool needs two chunks of which the second is only about 40% full, so 10,000-crystal pools run faster per crystal
(about 149 crystals/s on the Ada).

| GPU | Precision | `compile_trunk` | Packing | Crystals/s | vs Ada | Peak allocated GiB |
|---|---|---|---|---:|---:|---:|
| RTX 6000 Ada | bf16 | reduce-overhead | 8,192 | 134.7 | 1.00 | 1.72 |
| RTX 3090 | bf16 | reduce-overhead | 8,192 | 65.6 | 0.49 | 1.72 |
| RTX 3090 | fp32 | reduce-overhead | 8,192 | 30.4 | 0.23 | 1.72 |
| RTX A5000 | bf16 | reduce-overhead | 8,192 | 60.5 | 0.45 | 1.72 |
| Titan RTX | fp16 | reduce-overhead | 8,192 | 24.8 | 0.18 | 1.72 |
| Titan RTX | fp32 | reduce-overhead | 8,192 | 23.4; 19.8 | 0.17; 0.15 | 1.72 |
| Titan RTX | fp32 | none | padded, 256 | 7.7 | 0.06 | 0.99 |
| Titan RTX | bf16 | reduce-overhead | 8,192 | fails, see limitations | | |
| GTX 1080 Ti | fp32 | none | padded, 256 | fails at the first kernel | | |

### Relaxation

Same jobs: NequIP-OAM-L, batched FIRE with the Frechet cell filter (`relax_all` in batches of 128) on the
first 512 sampled crystals, after a 2-step warm-up. The marginal rate differences 100-step and 10-step runs, so it
excludes construction and FIRE initialisation; the 100-step rate includes them. Every constructed crystal relaxed
(512, and 511 on the A5000). The last column is 10,000 crystals x 500 steps at the marginal rate (the production
evaluation).

| GPU | Backend | Artifact | Marginal structure-steps/s | At 100 steps | vs Ada | 10k x 500 |
|---|---|---|---:|---:|---:|---:|
| RTX 6000 Ada | oeq | sm_89 (default) | 4,783 | 4,300 | 1.00 | 17 min |
| RTX 3090 | oeq | sm_89 (default) | 2,467 | 2,246 | 0.52 | 34 min |
| RTX 3090 | torchscript | | 237 | 231 | 0.05 | 5.9 h |
| RTX A5000 | oeq | sm_89 (default) | 2,115 | 1,868 | 0.44 | 39 min |
| Titan RTX | oeq | sm_89 (default) | fails, see limitations | | | |
| Titan RTX | oeq | sm_75 | 1,551 | 1,445 | 0.32 | 54 min |
| Titan RTX | torchscript | | 144 | 142 | 0.03 | 9.6 h |
| GTX 1080 Ti | any | | not reached (sampling fails first) | | | |

The sm_75 artifact, sha256 71e47f1ac21eabbf6c1bbe72b15d280c8f760b704adf684ac5e20c4505335c49, was built
on a Titan RTX with the recipe of the default one, which `tools/build_nequip.py --only oeq` also runs:
`$ORBITOL_PYTHON -m nequip.scripts.compile <NequIP-OAM-L 0.1 package> OUT --mode torchscript --device cuda --no-tf32
--modifiers enable_OpenEquivariance`. OpenEquivariance fixes its kernel schedule from the shared memory of the GPU it
is compiled on (`openequivariance/core/LoopUnrollConv.py`, `maxSharedMemPerBlock`).

## Verification

Each recommended configuration trained 200 steps from scratch with one evaluation of 64 crystals at
step 200 (`--max-steps 200 --sample-frequency 200 --sample-count 64 --val-every 0 --ckpt-every 0 --no-wandb`):
sampling with the run's own settings, 500-step NequIP relaxation of every crystal, and scoring.

| Configuration | GPU | Evaluation | Steps/s at step 200 | Result |
|---|---|---|---:|---|
| Ampere defaults: bf16, `reduce-overhead`, packed 8,192, oeq sm_89 | RTX 3090 | sync | 15.4 | evaluation complete in 223 s; startup logs the sm_89 artifact's sha256 |
| Turing: fp16, `reduce-overhead`, packed 8,192, oeq sm_75 | Titan RTX | async, `--eval-gres` requesting a Titan RTX | 13.9 | GPU stage on a Titan RTX (sampling 52 s, relaxation 75 s) with the fp16 sampler and the sm_75 artifact (71e47f1a...); CPU scoring complete |
| Turing: fp32, `reduce-overhead`, packed 8,192, oeq sm_75 | Titan RTX | sync | 10.9 | evaluation complete in 257 s; startup logs the sm_75 artifact's sha256 |
| Sampler CLI on the bf16 checkpoint, `--precision fp32` and `--precision fp16` | Titan RTX | | | 512 crystals each, no decode failures, `metadata["overrides"]` recorded |

After 200 steps the model's crystals are mostly unphysical: in all three runs 1 of 63-64 constructed
crystals relaxed and the rest diverged or became non-finite (15 and 47-48), the same on both GPU types. These runs
check the pipeline, not quality. The throughput jobs relaxed every crystal sampled from the trained checkpoint.

## Known limitations

- **Pascal (GTX 1080 Ti) does not run at all.** The installed torch 2.8.0+cu128 is compiled for
  sm_70, sm_75, sm_80, sm_86, sm_90, sm_100 and sm_120, so the first CUDA kernel fails with
  `torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`. It would need a separate environment with a torch build that still targets sm_61 (the cu126
  wheels), and even then Triton 3.4 does not target sm_61: torch.compile, FlexAttention (packed sampling) and
  ExactAdamW (`orbitol/optim.py`, a Triton kernel used whenever weight decay is zero) would all fail. That needs
  `--compile-trunk none --sample-token-budget 0`, a fallback to stock foreach AdamW (which ExactAdamW reproduces)
  and an OpenEquivariance artifact built on sm_61 or the
  TorchScript relaxer. Not attempted.
- **bf16 on Turing fails** inside torch.compile with
  `torch._dynamo.exc.Unsupported: Attempt to trace generator` (packed sampling). Use fp16 or fp32.
- **The default OpenEquivariance artifact aborts the process on Turing**:
  `cuKernelSetAttribute( CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, ...) failed with error
  CUDA_ERROR_INVALID_VALUE`. It is a native abort, not a Python exception, so an in-process (sync)
  evaluation would end the training job. Use the sm_75 artifact through `extra.nequip_model`.
- **Runs do not move between Turing and the newer GPUs.** `precision` and `extra` are part of the resume
  check (`Resume mismatch: cfg`), and sync runs also compare `nequip_model_sha256`. Only execution choices may
  change on resume: `compile_trunk`, `pad_multiple`, `sample_token_budget`, the decoder (`decode_impl`,
  `decode_overlap`) and the training-step toggles `periodic_target_impl`, `gem_compile`, `fast_update`,
  `fast_encode` and `compile_step`. Ampere and Ada runs share one configuration.
- **Post-hoc evaluation** (`orbitol/posthoc_eval.py`) samples with the snapshot's stored precision unless
  `--sampler-overrides` changes it, for example `--sampler-overrides '{"precision": "fp16"}'` for a bf16 snapshot
  on a Titan RTX; the accepted keys are `SAMPLER_OVERRIDES` in `orbitol/sample.py`. `--nequip-oeq-path` selects the
  relaxation artifact.
- **Titan RTX throughput varies between jobs** (power and thermal throttling): fp32 `reduce-overhead`
  training gave 7.4-10.1 steps/s in three jobs on one node, and sampling 23.4 and 19.8 crystals/s.
- TF32 is disabled everywhere, so fp32 on Ampere is slower than bf16 and is not recommended there.
- fp16 sampling of the bf16-trained checkpoint was finite and decoded every crystal; its sample quality
  (MSUN) was not compared with bf16.
- A cold Inductor cache costs about 2-2.5 min of compilation per GPU type for `reduce-overhead`
  training; a warm one about 10-20 s.

## How it was measured

Training: `slurm/train_orbitol.sbatch` with `EVAL_MODE=sync` and the flags above, 800 steps. Sampling and
relaxation: timed calls of `sample_pool` and `relax_all` with the precision,
`compile_trunk`, packing and relaxer of each row.

The sampler overrides `--precision`, `--compile-trunk` and `--sample-token-budget` are part of
`SAMPLER_OVERRIDES` in `orbitol/sample.py` (tests in `tests/test_sample_overrides.py`). fp16 gradient overflows are
skipped by the scaler in `orbitol/train.py`
(`tests/test_train.py::test_fp16_overflow_skips_the_update_and_training_continues`).
