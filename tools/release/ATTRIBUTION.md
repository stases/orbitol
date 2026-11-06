# Orbitol release bundle: licence and attribution

The bundle (model weights and data files) is licensed under Creative Commons Attribution 4.0 International
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). The Orbitol code is licensed under MIT (see LICENSE in the
code repository). If you use the bundle, cite the Orbitol paper and the sources below.

## Contents and sources

- **Model weights.** EMA weights of the Orbitol-X MP-20 model of the main paper (step 1,500,000). Optimizer and
  trainer state are not included.
- **Wyckoff tables** (`tables/v1`). Generated with PyXtal 1.1.2 (MIT) and spglib 2.7.0 (BSD-3-Clause).
- **MP-20** (`mp20/`). Structures from the Materials Project (Jain et al., 2013; CC BY 4.0) in the MP-20 split of
  CDVAE (Xie et al., 2022), as distributed in the Crystalite dataset
  (https://huggingface.co/datasets/joshrosie/crystalite-datasets; the dataset card states MIT; the structures keep the
  terms of their sources).
  The bundle holds processed tokens, orbit caches and the symmetry reference derived from it. The raw CSVs and the
  MP-20 phase diagram are downloaded separately.
- **Matbench Discovery data files** (Riebesell et al.; figshare article 22715158; CC BY 4.0). Source of the MP-20
  phase diagram (downloaded separately).
- **NequIP-OAM-L 0.1** (`mlip/`; Seán R. Kavanagh and MIR Group @ Harvard, "NequIP & Allegro Foundation
  Potentials", doi:10.5281/zenodo.16980200; CC BY 4.0). TorchScript and OpenEquivariance compilations of the
  published weights, with the TorchScript debug tables removed. The OpenEquivariance build targets compute capability
  8.9 and runs on Ampere and Ada GPUs; rebuild it for other GPUs, such as Turing, with `tools/build_nequip.py`.
- **Crystalite** (https://github.com/joshrosie/crystalite, commit 03603fe; MIT). The pinned core used to process
  tokens and evaluate structures. Fetched separately; not included.
- **Software used to build the bundle.** pymatgen 2024.11.13 (MIT; Ong et al., 2013) for structure processing and
  symmetry detection; nequip 0.17.0 (MIT) and OpenEquivariance 0.5.4 (BSD-3-Clause) for the relaxer compilations.
  Code derived from pymatgen in the Orbitol repository keeps its notice in `THIRD_PARTY_NOTICES.md` there.

## Not included

The Alex-MP-20 model and data, the Orbitol-I and control models, test splits, and generated sample pools.
