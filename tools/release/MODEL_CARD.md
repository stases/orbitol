---
license: cc-by-4.0
tags:
- materials-science
- crystal-structure-generation
- diffusion
---

# Orbitol release bundle

Weights and data for the paper [Orbitol: Crystal Generation on Wyckoff Orbits](https://arxiv.org/abs/2610.30086)
by Ivan Žugec, Joshua Rosenthal, Ivor Lončarić, Jan-Willem van de Meent and Tin Hadži Veljković. Orbitol is a
diffusion transformer that generates crystals with a chosen space group, describing each structure by its Wyckoff
orbits. Code (MIT): https://github.com/stases/orbitol. Instructions for reproducing the paper's results are in
`docs/reproduce.md` in the code repository.

## Models

EMA weights only; optimizer and trainer state are not included.

| File | Model | Training step |
|---|---|---|
| `checkpoints/mp20_x_postper_2500k/step_1500000.pt` | Orbitol-X, MP-20 | 1,500,000 (of a 2.5M schedule) |

## Contents

| Folder | Contents |
|---|---|
| `configs/paper/` | Training and sampler settings; `models.json` lists provenance and expected results |
| `tables/v1/` | Wyckoff tables (PyXtal, spglib) |
| `mp20/` | Processed train and validation tokens, orbit caches, symmetry reference |
| `mlip/` | NequIP-OAM-L 0.1 relaxer: TorchScript, and OpenEquivariance compiled for compute capability 8.9, which runs on Ampere and Ada GPUs (rebuild for other GPUs, such as Turing, with `tools/build_nequip.py`) |

`MANIFEST.json` lists every file with its size, SHA-256, role and origin. `SHA256SUMS` checks the download.
`ATTRIBUTION.md` gives licences and sources.

Not included: the Alex-MP-20 model and data, the Orbitol-I and control models, processed test splits, the MP-20 raw
CSVs and phase diagram (public
downloads that the fetch tool retrieves; the raw CSVs include the test split, which the released training and
evaluation recipes do not use and the paper uses only for skeleton coverage), and generated sample pools.

## Download

With the code repository set up, the fetch tool downloads the bundle, checks every file against `MANIFEST.json`
and places it at the configured data paths. `docs/reproduce.md` gives the revision used for the paper.

```bash
python -m tools.release.fetch --bundle https://huggingface.co/stases/orbitol/resolve/<revision> \
    --public mp20-raw,mp20-hull
```

Or download the files directly and check them:

```bash
uvx --from huggingface_hub hf download stases/orbitol --revision <revision> --local-dir orbitol-bundle
cd orbitol-bundle && sha256sum -c SHA256SUMS
```

## Licence

Weights and data: CC BY 4.0. The structures and the relaxer keep the terms of their sources; see `ATTRIBUTION.md`.
If you use this bundle, cite the Orbitol paper and the sources listed there.

## Citation

```bibtex
@article{zugec2026orbitol,
  title   = {Orbitol: Crystal Generation on Wyckoff Orbits},
  author  = {{\v{Z}}ugec, Ivan and Rosenthal, Joshua and Lon{\v{c}}ari{\'c}, Ivor and van de Meent, Jan-Willem and
             Had{\v{z}}i Veljkovi{\'c}, Tin},
  journal = {arXiv preprint arXiv:2610.30086},
  year    = {2026}
}
```
