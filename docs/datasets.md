# Training datasets

The frozen `DatasetSpec` registry and shared `resolve_dataset_paths` resolver
live in `orbitol/contracts.py`. `TrainConfig.dataset`
defaults to `mp20`; old snapshots without the field also mean `mp20`. Resuming a
checkpoint with a different dataset is refused before opening the new caches.
Legacy path constants remain aliases of the MP-20 entry.

Paths derive from `ORBITOL_DATA_HOME` via `C.DATA_LAYOUT`; each listed key can override its
own path. The local file and environment are resolved by `python -m orbitol.contracts`.

| Field / key suffix | MP-20 key → layout | Alex-MP-20 key → layout |
|---|---|---|
| Input root | `ORBITOL_MP20_DATA_ROOT` → `mp20` | `ORBITOL_ALEX_MP20_DATA_ROOT` → `alex_mp20` |
| CSV format | MP-20 CIF | MatterGen Cartesian arrays |
| Orbit cache | `ORBITOL_MP20_ORBIT_CACHE_DIR` → `mp20/orbit_cache` | `ORBITOL_ALEX_MP20_ORBIT_CACHE_DIR` → `alex_mp20/orbit_cache` |
| Atom cache | `ORBITOL_MP20_ATOM_CACHE_DIR` → `mp20/atom_cache` | `ORBITOL_ALEX_MP20_ATOM_CACHE_DIR` → `alex_mp20/atom_cache` |
| Symmetry reference | `ORBITOL_MP20_SYMMETRY_REFERENCE` → `mp20/symmetry_reference.json` | `ORBITOL_ALEX_MP20_SYMMETRY_REFERENCE` → `alex_mp20/symmetry_reference.json` |
| Hull | `ORBITOL_MP20_PHASE_DIAGRAM` → `hulls/mp20/2023-02-07-ppd-mp.pkl` | `ORBITOL_ALEX_MP20_PHASE_DIAGRAM` → `hulls/alex_mp2020/variants/v1.pkl` |
| W&B project | `orbitol` | `orbitol_alex_mp20` |

Shared keys: `ORBITOL_TABLES_DIR` → `tables/v1`, `ORBITOL_NEQUIP_TS` →
`mlip/nequip_ts/NequIP-OAM-L.nequip.pth`, `ORBITOL_NEQUIP_OEQ` →
`mlip/nequip_oeq/NequIP-OAM-L-oeq-ts.nequip.pth`. Runs use `ORBITOL_RUNS_ROOT`;
W&B uses `ORBITOL_WANDB_ENTITY` (unset: your W&B default entity).

Alex scores against the fixed Alex-MP + MP2020 reference `v1`, built with
`tools/build_alex_reference.py`; the released MatterGen source corresponds to the `original`
variant. The release bundle includes no Alex data. See
[docs/alex_reference.md](alex_reference.md).

Select Alex with `--dataset alex_mp20` in the training CLI. For example,
after all CPU builds below finish, launch:

```bash
slurm/submit slurm/train_orbitol.sbatch orbitol_alex_mp20 --dataset alex_mp20
```

The resolver uses explicit CLI path flags first, then `cfg.extra`, then the
selected registry entry. The supported extra keys are `cache_dir`, `data_root`,
`atom_cache_dir`, `symmetry_reference`, and `phase_diagram`. Each override affects
only its own path. `cache_dir` selects the training cache and the skeleton bank
rebuilt from that cache. To train in atom view, explicitly set `cache_dir` to
$ORBITOL_DATA_HOME/alex_mp20/atom_cache.
`--extra` replaces the whole dictionary; preserve existing entries on resume.

Training derives allowed elements from the selected train tokens and saves
the resulting mask in its checkpoint. Checkpoint sampling restores that mask and
rebuilds the bank from the snapshot's dataset/cache override. Posthoc defaults
come from Orbitol's saved `cfg.dataset` or Crystalite's saved
`model_args.dataset_name`; `--data-root`, `--symmetry-reference`, and `--ppd-path`
override them. Committed pools retain their resolved paths after checkpoint
pruning, and the queue rebuilds its evaluator when those references change.
The standalone evaluator also accepts `--dataset alex_mp20` and these path flags.

Both evaluator reference splits follow the selected input root. The pinned
metric behavior is preserved: train supplies novelty structures and val supplies
distribution metrics (with a train fallback for empty val). They are not combined
into a new novelty corpus. The symmetry panel uses train tokens, the existing
Crystalite decoder, spglib at symprec 0.1 and 0.01 Å, and angle tolerance 5°.

## Alex builds

Inputs are read-only. Alex has 540,162 / 67,521 / 67,521 train/val/test
structures with at most 20 atoms, supplied under
$ORBITOL_DATA_HOME/alex_mp20. Full builds belong on Slurm; none
are part of the implementation smoke checks. Run from the checkout after applying the setup exports.
These commands use the default layout;
if you override an individual key, use its resolved value instead. Export the shell root first:

```bash
export ORBITOL_DATA_HOME="$(python -m orbitol.contracts ORBITOL_DATA_HOME)"
```

The CPU wrapper binds the pinned core and the project Python environment. The release bundle
holds no Alex-MP-20 data. The builds need two inputs under $ORBITOL_DATA_HOME/alex_mp20:

- `raw/{train,val,test}.csv`, the MatterGen splits, obtained separately. The orbit-cache build reads them.
- `processed/mp20_tokens_{train,val,test}.pt`, the pinned Crystalite core's token files. The atom-cache
  build reads them. The symmetry reference build and training (allowed elements, symmetry-reference
  check) read the train file; evaluation reads train (novelty) and val (distribution metrics).

Orbitol only reads the token files and stops if one is missing; it never preprocesses. The core's
`MP20Tokens` constructor (`$ORBITOL_CORE_ROOT/src/data/mp20_tokens.py`) writes them. For a missing
`processed/mp20_tokens_<split>.pt` it reads `raw/<split>.csv` (MatterGen's Cartesian rows included),
Niggli-reduces each cell with 32 worker processes, and keeps structures with at most 20 atoms and atomic
number at most 94. It also writes `raw/<split>_tokens.pt` and a filter summary
`processed/mp20_tokens_<split>_info.txt`. This is the only step that writes into the input root, so run
it first:

```bash
for split in train val test; do
    slurm/submit --job-name="alex-tokens-$split" --cpus-per-task=32 \
        slurm/cpu.sbatch \
        -c 'import sys, orbitol.core as core; core.verify_core()
from src.data.mp20_tokens import MP20Tokens
MP20Tokens(sys.argv[1], split=sys.argv[2])' "$ORBITOL_DATA_HOME/alex_mp20" "$split"
done
```

For the splits above, each summary reports 0 structures filtered out.

Build all orbit splits (zero tolerated failures by default):

```bash
for split in train val test; do
    slurm/submit --job-name="alex-orbit-$split" --cpus-per-task=8 \
        slurm/cpu.sbatch \
        -m orbitol.data build-cache --dataset alex_mp20 --split "$split" --workers 8 \
        --out $ORBITOL_DATA_HOME/alex_mp20/orbit_cache
done
```

Failure reports include material IDs. If a build fails, inspect its causes
before rerunning with an explicit `--max-failures N`; accepted exclusions are
recorded in each manifest. No partial cache is published when this limit is exceeded.

Build all atom splits directly from the chosen Crystalite token files:

```bash
for split in train val test; do
    slurm/submit --job-name="alex-atom-$split" --cpus-per-task=1 \
        slurm/cpu.sbatch \
        -m orbitol.data build-cache --dataset alex_mp20 --view atom --split "$split" \
        --out $ORBITOL_DATA_HOME/alex_mp20/atom_cache
done
```

After the train orbit cache finishes, build the serialized skeleton bank:

```bash
slurm/submit --job-name=alex-bank --cpus-per-task=1 \
    slurm/cpu.sbatch \
    -m orbitol.skeleton_prior build \
    --cache $ORBITOL_DATA_HOME/alex_mp20/orbit_cache \
    --out $ORBITOL_DATA_HOME/alex_mp20/skeleton_bank.json
```

Training and checkpoint sampling currently rebuild this same bank from the
train cache; the serialized file is an inspectable artifact. For an atom-view bank,
use $ORBITOL_DATA_HOME/alex_mp20/atom_cache and save to
$ORBITOL_DATA_HOME/alex_mp20/atom_skeleton_bank.json.

Build the symmetry reference independently of orbit/atom cache builds:

```bash
slurm/submit --job-name=alex-symmetry --cpus-per-task=1 \
    slurm/cpu.sbatch \
    -m orbitol.symmetry_panel build --dataset alex_mp20 \
    --out $ORBITOL_DATA_HOME/alex_mp20/symmetry_reference.json
```

## Reader

MatterGen stores Cartesian positions in Å, lattice vectors as rows, and
atomic numbers in multiline numpy-repr fields. The reader parses numeric text
without executing Python, increases the CSV field-size limit, and validates array
shapes, finite values and the 20-atom limit. Space-group labels are resolved against
spglib's short/full/international Hermann–Mauguin aliases for all 530 Hall settings.
Whitespace, underscores, Unicode subscripts and Unicode minus are normalized;
tabulated setting suffixes are accepted. Unknown or ambiguous symbols become
ordinary cache failures; the reader never substitutes a redetected group for the
CSV label. MP-20 retains its original CIF path and cache manifest schema.
