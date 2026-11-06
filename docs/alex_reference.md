# Alex-MP hull reference: original and fixed variants

This is how Orbitol scores stability for Alex-MP-20. It covers what MatterGen ships, what we found
wrong with it, what each fix does, and how to build or switch any of it.

## Quick use

- **The source.** We score candidates against MatterGen's released Alex-MP DFT reference,
  `reference_MP2020correction.gz`: 845,997 MP2020-corrected entries from Materials Project and Alexandria.
- **The variants.** The released archive is never modified. A build turns it into one per-entry table, and each
  hull variant is a pickle derived from that table:

| Variant | Flags | Use |
|---|---|---|
| `original` | none | Reproduces the released archive exactly; comparable with MatterGen's evaluator |
| `v1` | `known_bad`, `recorrect`, `potcar_align` | The fixed reference; Orbitol's default for Alex-MP-20 |
| e.g. `known_bad+recorrect` | any subset, sorted, joined by `+` | Ablations |

Files:
- table: `$ORBITOL_DATA_HOME/hulls/alex_mp2020/entries.parquet`, with manifest.json and
  diagnostics.jsonl next to it;
- variants: `$ORBITOL_DATA_HOME/hulls/alex_mp2020/variants/<variant>.pkl`; the active `v1` hull can
  be overridden with `ORBITOL_ALEX_MP20_PHASE_DIAGRAM`.

Each pickle is an `orbitol.alex_hull.AlexHull`. It loads through the evaluator's existing `load_phase_diagram`, so
selecting a variant means selecting a path: `--ppd-path` for post-hoc evaluation, `DatasetSpec.phase_diagram` for
training. `hull.manifest` records:
- the variant and its flags;
- the sha256 of the table manifest and of both assets;
- the build's git commit;
- entry counts.

The provenance strings inside `potcar_shifts.json` and `known_bad.json` were redacted after the hull build, so
checksums recorded by previously built hull manifests refer to the pre-redaction files. Shifts, targets, statuses and
exclusions are unchanged, and a rebuilt table has identical arrays; only the recorded asset hashes differ:

| Asset | sha256 before redaction | sha256 now |
|---|---|---|
| `potcar_shifts.json` | `656e50b1cb54833bd9b23e64517cae2aa33c7d2949ceb2166b7e03caf6b55a5a` | `348bbb2772a5fe0558c32e611bd2332016ce267b8720f8186db4964db1b2fc04` |
| `known_bad.json` | `c93c152cc3ba35cfc56b7f8b53aed6c454a532c2083e5015c53312befe77018a` | `8ca4fb23dd1d842c0829140a3fc32e2000927ff5a0d3ebfa6a16eaf06ec08333` |

Post-hoc evaluation also records the pickle's sha256 (`phase_diagram_sha256` in evaluation_settings.json) and stops if
the file changed between the pool and evaluate stages. Training configs record the path; the pickle's manifest holds
the rest.

**Rule:** a named variant is frozen. Changing any rule or asset defines a new name (`v2`, …), never an edit of `v1`.

## What MatterGen ships

MatterGen documents both Alex-MP references as 845,997 ordered DFT-relaxed entries
(`mattergen/evaluation/reference/presets.py` in https://github.com/microsoft/mattergen).
- The MP part, 145,891 entries labelled `mp-…`, is described as calculations retrieved from Materials Project
  v2022.10.28. In fact 13,803 of them carry `msft-` replacement tasks under their MP ID.
- The non-MP part, 700,102 entries labelled `msft-…`, is Alexandria (release 2022.126) structures recomputed with MP
  settings (MPRelaxSet; MatterGen Appendix C).
- Entries are MP2020-corrected. The build reads the decompressed archive, an LMDB (`--lmdb` under Build).

## What we found

**Scoring an MLIP against a DFT hull is valid here**.
- On 300 on-hull binaries per reference, NequIP-OAM-L agrees with raw DFT to a median +0.005 eV/atom (MAD 0.006).
- Re-referencing elements to MLIP chemical potentials makes the binaries worse, not better (mean |error| 0.016 →
  0.018 eV/atom), so no same-MLIP hull and no elemental recalibration is used.

The defects are in the reference data:

1. **One corrupt entry**. `Co-O.CoO2.34` (mp-796278-GGA+U) has 6 sites and a raw energy of −11.84 eV/atom.
   - The other 38 CoO2 entries lie between −4.43 and −5.74 eV/atom.
   - MP's own record of the same material is −5.58 eV/atom.
   - NequIP on the stored structure is 6.14 eV/atom above the stored energy.
   - It is an old MP task, with the same POTCARs and U as MP, not a cell-normalisation slip.
2. **Candidates and reference phases receive MP2020 corrections by different rules**.
   - Every archive entry stores `data.oxidation_states = {}`. pymatgen then applies anion corrections only to the
     most electronegative element.
   - Candidates scored through the core
     (`$ORBITOL_CORE_ROOT/src/utils/sample_stats.py:1047`) carry no such key,
     so pymatgen guesses oxidation states instead.
   - The build reproduces all 845,997 stored corrections exactly under the archive's rule. Under the candidate rule,
     39,189 corrections (4.6%) differ.
   - Example: Ce(SiAg)2 gets +0.142 eV (the Si anion correction) as a candidate but 0 as a reference. A known stable
     phase scored as a candidate would land 28 meV/atom above its own hull.
   - MatterGen's evaluator shares this inconsistency, so `original` stays the MatterGen-comparable number.
3. **Two POTCAR libraries are mixed in one hull**.
   - 271,575 msft and 7,632 mp entries use the potpaw_PBE (2010) files for at least one of Li, Mg, Sm, Pd, Na, Ce,
     Ge, Eu or Gd. The rest use MP's legacy files.
   - The registered hashes identify the 2010 library, not PBE_52 or PBE_54.
   - Switching library shifts every atom of the element by a near-constant energy. Two independent same-structure
     pair sets give the same shifts:

     | Element | MP side | Alexandria side |
     |---|---:|---:|
     | Ge | +0.109 | +0.105 |
     | Mg | +0.092 | +0.093 |
     | Sm | +0.068 | +0.068 |
     | Eu | −0.058 | −0.054 |

     Shifts are in eV per atom of the element. The asset applies the MP-side value for all nine elements; the other
     five are small: Gd −0.021, Li −0.016, Na −0.004, Pd −0.002, Ce −0.001 (Li below).

**Checked and left unchanged:**
- **The Alexandria part matches its source**. A stratified 20k sample agrees with the original Alexandria
  release to a median −0.2 meV/atom (MAD 2.2). Its tails are mostly on the harmless too-high side.
- **The 218 same-formula gaps above 1 eV/atom exist in the source data.**
- **One entry stays flagged:** Rb2MnFeF6 is retained but flagged in the known-bad asset.
- **MP's own snapshot has broken records too:** SiO2 mp-555394 and Cr2CuO4 mp-504573. They affect only the legacy MP
  hull (the MP-20 hull, `hulls/mp20/2023-02-07-ppd-mp.pkl`): Cr2CuO4 depresses the Cr–Cu–O hull by
  about 1.4 eV/atom. That pickle is not edited.

## The fix-sets (normative)

For entry i with stored raw total energy `raw_i`, the corrected energy under flag set F is:

    E_i = (raw_i − potcar_shift_i  if potcar_align ∈ F  else raw_i)
        + (recorrected_i           if recorrect ∈ F     else stored_correction_i)

The flags commute.

- **`known_bad`** drops the keys listed with `"action": "exclude"` in
  `orbitol/assets/alex_reference/known_bad.json` (today only `Co-O.CoO2.34`). Entries
  marked `retain_flagged`, or without a key, are documentation only.
- **`recorrect`.**
  - The correction becomes what the core's candidate path assigns to the same structure and raw energy:
    `_build_mp2020_parameters`, empty `data`, and `MaterialsProject2020Compatibility(check_potcar=False)`.
  - This makes candidates and reference phases consistent by construction.
  - The build counts the 1 entry whose synthesised run type disagrees with the stored one, and the 4,966 whose oxide
    type is reclassified.
- **`potcar_align`.**
  - An element's shift applies only when the stored POTCAR **hash** equals the asset's `new.hash` and the element's
    `status` is `applied`. Then `potcar_shift_i = Σ_e n_i(e)·s_e`, with s_e = E(new file) − E(legacy file), from
    `orbitol/assets/alex_reference/potcar_shifts.json`.
  - Every `target` is `legacy`: the NequIP scale probe puts NequIP on the legacy scale (below).
  - Entries with any other non-legacy hash, or with missing or ambiguous POTCAR metadata, are excluded: 39 entries.

## How the shifts were fitted

The shift table's method is recorded in `orbitol/assets/alex_reference/potcar_shifts.json`: per element, the pair counts, estimates and
95% CIs, `status`, `target`, `reason` and the scale-probe evidence. A shift is applied when an element has ≥ 50
MP-side pairs, or ≥ 20 MP-side pairs and overlapping MP and Alexandria 95% CIs (Ge and Sm, with 37 and 32 pairs).

**The probe's verdict**, with NequIP − DFT minus NequIP's baseline error:

| Element | Against legacy DFT | Against new-file DFT | Verdict |
|---|---:|---:|---|
| Eu | +0.003 | +0.020 | legacy |
| Ge | −0.003 | −0.036 | legacy |
| Sm | −0.006 | −0.024 | legacy |

Mg, Li and Gd point the same way but are inconclusive. Mg in particular (47k entries) is on the legacy scale by
analogy, not by its own verdict. A model-cancelled check on the probe's pairs reproduces the shifts: Eu −0.061,
Mg +0.091, Ge +0.105, Sm +0.068. NequIP cancels in it, but the DFT pairs are the same MP pairs the fit used, so it is
not independent of the fit.

**Li.** The MP-side value (−0.016) is applied. Alexandria's (−0.007) disagrees, which is recorded; the NequIP-cancelled
check (−0.022) sides with MP.

## Validation

**Self-consistency** (v1 build, 2026-10-08). 1,000 reference phases, stratified by source and POTCAR
library, were rescored as candidates through the core's path, with their own structure and raw energy.
- **Under `v1`, all 483 legacy-file phases are consistent:** every on-hull phase scores 0, none scores below its own
  hull, and the residual is 0.
- **The 478 new-file phases** score exactly their shift per atom (residual ≤ 8e-16), as designed, since they are fed
  their unshifted raw energy.
- **The 39 unknown-offset entries** are excluded.
- **Under `original`:** 1 of 131 on-hull phases scores off its own hull, and 13 of the 1,000 phases score below their
  own hull (mp: 4 legacy-file, 5 new-file; msft: 3 legacy-file, 1 new-file). No shift is applied under `original`,
  so the POTCAR stratum does not matter here; this is the correction-rule inconsistency.

## Build

These commands produce `v1`. Inputs: MatterGen's `data-release/alex-mp/reference_MP2020correction.gz`,
its decompressed LMDB, and Matbench Discovery's `2023-02-07-mp-elemental-reference-entries.json.gz`.
`slurm/cpu.sbatch` runs the project Python with PYTHONPATH set to the checkout. The variant loop calls sbatch
directly (add your placement options) and sets both itself, because the CLI's stdout is the manifest. The
variant jobs start only after the table job succeeds (`--parsable` returns its id), and their redirection
target directory is created first.

```bash
export ORBITOL_DATA_HOME="$(python -m orbitol.contracts ORBITOL_DATA_HOME)"
H=$ORBITOL_DATA_HOME/hulls/alex_mp2020
LMDB=/path/to/reference_MP2020correction   # the decompressed archive
S="slurm/submit --mem=64G"
mkdir -p $H/variants

# Table (16 workers, about 9 min); the assets default to orbitol/assets/alex_reference/
table_job=$($S --parsable --job-name alex-reference --cpus-per-task=16 slurm/cpu.sbatch \
  tools/build_alex_reference.py \
  --lmdb $LMDB \
  --mp-elemental /path/to/2023-02-07-mp-elemental-reference-entries.json.gz \
  --source-gz /path/to/mattergen/data-release/alex-mp/reference_MP2020correction.gz \
  --out $H --workers 16)

# One pickle per variant; the CLI prints the variant manifest (about 40 s each)
for v in original known_bad recorrect potcar_align known_bad+potcar_align known_bad+recorrect \
         potcar_align+recorrect v1; do
  sbatch --mem=64G --job-name alex-variants --dependency=afterok:$table_job \
    --wrap "cd $PWD && PYTHONPATH=$PWD $ORBITOL_PYTHON \
    -m orbitol.alex_hull --table $H/entries.parquet --variant $v --out $H/variants/$v.pkl \
    > $H/variants/$v.manifest.json"
done
```

**Build counts** (manifest.json):
- 845,997 entries;
- stored corrections reproduced: all;
- `recorrect` changes 39,189;
- shifts applied: Li 62,342, Mg 47,133, Sm 47,043, Pd 46,160, Na 45,488, Ce 33,866, Ge 33,217, Eu 6,702, Gd 1,165;
- 39 unlisted, 1 known-bad.

## Limitations

- **The pinned-Crystalite baseline cannot load the pickles.**
  `slurm/crystalite_baseline.sbatch` puts only the pinned core on PYTHONPATH, so it
  cannot unpickle AlexHull.
- **`recorrect` depends on pymatgen's oxidation-state guessing.** The build used pymatgen 2024.11.13 (recorded in the
  manifest). An upgrade can change the corrections on both the candidate and the reference side, so it needs a
  rebuild under a new variant name.
- **Coverage.** The Alex reference has 76 elemental terminals: none for the noble gases, Tc, Pm, or Z ≥ 84 (Po to
  Pu), matching MatterGen's element exclusions. Candidates containing these elements cannot be scored and count as
  failures under every Alex variant.
- **Not proven against independent DFT.** These fixes make the reference internally consistent and consistent with
  the candidate path. They are not a DFT verification of generated structures.
