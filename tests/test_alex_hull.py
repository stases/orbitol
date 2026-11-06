"""Small fixed-reference hulls and the pinned MP2020 candidate path."""

import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry
import pytest

import orbitol.core  # noqa: F401  bind the pinned core before src imports
from orbitol.alex_hull import AlexHull, FLAGS
from src.eval.stability import load_phase_diagram
from src.utils.sample_stats import _build_mp2020_parameters, compute_e_above_hull_mp2020_like


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@pytest.fixture
def workdir():
    parent = ROOT / ".pytest_cache"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        yield Path(directory)


def row(formula, raw, *, stored=0.0, recorrected=None, shift=0.0, bad=None, unlisted=False):
    comp = Composition(formula)
    # Preserve formula order: the reader must align counts to sorted chemsys.
    elements = [element.symbol for element in comp.elements]
    return {
        "key": f"{comp.chemical_system}.{formula}.0",
        "chemsys": comp.chemical_system,
        "elements": elements,
        "amounts": [comp[element] for element in elements],
        "natoms": comp.num_atoms,
        "raw_energy": raw,
        "stored_correction": stored,
        "recorrected_correction": stored if recorrected is None else recorrected,
        "potcar_shift": shift,
        "unlisted_nonlegacy": unlisted,
        "known_bad": bad,
    }


def write_table(directory, rows, *, recorrect_failures=0):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "entries.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    manifest = {
        "inputs": {
            "lmdb": {"sha256": "a" * 64},
            "source_gz": {"sha256": "b" * 64},
            "mp_elemental": {"sha256": "c" * 64},
            "known_bad": {"sha256": "d" * 64},
            "shifts": {"sha256": "e" * 64},
        },
        "pymatgen_version": "synthetic",
        "git_commit": "f" * 40,
        "counts": {"entries": len(rows), "recorrect_failures": recorrect_failures},
        "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reproduction_check": {"passed": True},
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture
def ternary_table(workdir):
    return write_table(workdir / "reference", [
        row("Li", -1), row("Li2", -1), row("Mg", -2), row("O2", -6),
        row("Li2O", -10, stored=-0.6, recorrected=-1.2, shift=0.4),
        row("MgO", -10, stored=-1, recorrected=-2, shift=-0.6),
        row("LiMg", -3, stored=-0.5),
        row("LiMgO2", -16, stored=-2, recorrected=-3, shift=0.8),
        row("LiMgO2", -50, bad="exclude"),
        row("LiO", -30, unlisted=True),
    ])


def test_original_matches_pymatgen_and_includes_subsystems(ternary_table):
    hull = AlexHull.from_table(ternary_table, frozenset())
    reference = [
        PDEntry("Li", -1), PDEntry("Li2", -1), PDEntry("Mg", -2), PDEntry("O2", -6),
        PDEntry("Li2O", -10.6), PDEntry("MgO", -11), PDEntry("LiMg", -3.5),
        PDEntry("LiMgO2", -18), PDEntry("LiMgO2", -50), PDEntry("LiO", -30),
    ]
    assert hull._cache == {}
    for formula in ("Li", "O", "MgO", "Li3Mg", "LiMgO2", "Li2MgO4", "Li3Mg2O"):
        comp = Composition(formula)
        elements = set(comp.elements)
        pd_ref = PhaseDiagram([entry for entry in reference if set(entry.elements) <= elements])
        assert hull.get_hull_energy_per_atom(comp) == pytest.approx(pd_ref.get_hull_energy_per_atom(comp), abs=1e-9)
        candidate = ComputedEntry(comp, -18, correction=-0.7)
        assert hull.get_e_above_hull(candidate, allow_negative=True) == pytest.approx(pd_ref.get_e_above_hull(
            candidate, allow_negative=True,
        ), abs=1e-9)
    system_cached = hull._cache[("Li", "Mg", "O")]
    assert len(system_cached[1]) == len(reference)
    hull.get_hull_energy_per_atom(Composition("O2MgLi"))
    assert hull._cache[("Li", "Mg", "O")] is system_cached


@pytest.mark.parametrize(("flags", "compound_rows", "formula", "original", "changed", "n_entries"), [
    (frozenset({"known_bad"}), [row("Li2O", -6), row("LiO", -12, bad="exclude")], "LiO", -6, -1.5, 3),
    (frozenset({"recorrect"}), [row("Li2O", -6, stored=-3, recorrected=-6)], "Li2O", -3, -4, 3),
    (frozenset({"potcar_align"}), [row("Li2O", -6, stored=-3, shift=3)], "Li2O", -3, -4, 3),
    (frozenset({"potcar_align"}), [row("Li2O", -6, stored=-3, shift=-3)], "Li2O", -3, -2, 3),
    (frozenset({"potcar_align"}), [row("Li2O", -6), row("LiO", -12, unlisted=True)], "LiO", -6, -1.5, 3),
])
def test_independent_flags(workdir, flags, compound_rows, formula, original, changed, n_entries):
    path = write_table(workdir / "reference", [row("Li", 0), row("O2", 0), *compound_rows])
    comp = Composition(formula)
    assert AlexHull.from_table(path, frozenset()).get_hull_energy_per_atom(comp) == pytest.approx(original)
    variant = AlexHull.from_table(path, flags)
    assert variant.get_hull_energy_per_atom(comp) == pytest.approx(changed)
    assert variant.manifest["n_entries"] == n_entries
    for terminal in ("Li", "O"):
        assert variant.get_hull_energy_per_atom(Composition(terminal)) == pytest.approx(0, abs=1e-9)


def test_all_flags_compose_and_record_provenance(workdir):
    path = write_table(workdir / "reference", [
        row("Li", 0), row("Mg", 0), row("O2", 0),
        row("LiO", -40, bad="exclude"), row("MgO", -40, unlisted=True),
        row("LiMgO2", -12, stored=-4, recorrected=-8, shift=4),
    ])
    manifests = []
    table_bytes = (path.parent / "manifest.json").read_bytes()
    source = json.loads(table_bytes)
    for size in range(4):
        for subset in combinations(sorted(FLAGS), size):
            flags = frozenset(subset)
            hull = AlexHull.from_table(path, flags)
            manifest = hull.manifest
            expected = "original" if not flags else "v1" if flags == FLAGS else "+".join(subset)
            assert manifest["variant"] == expected
            assert manifest["flags"] == list(subset)
            assert manifest["table_manifest_sha256"] == hashlib.sha256(table_bytes).hexdigest()
            assert manifest["table_manifest"] == {key: source[key] for key in (
                "inputs", "counts", "git_commit", "pymatgen_version", "table_sha256",
            )}
            assert manifest["asset_sha256"] == {"known_bad": "d" * 64, "potcar_shifts": "e" * 64}
            assert len(manifest["arrays_sha256"]) == 64
            assert manifest["n_dropped_nan"] == 0
            assert manifest["missing_terminals"] == []
            manifests.append(manifest)
    assert len({manifest["arrays_sha256"] for manifest in manifests}) == 8
    v1 = AlexHull.from_table(path, FLAGS)
    assert v1.manifest["n_entries"] == 4
    assert v1.get_hull_energy_per_atom(Composition("LiMgO2")) == pytest.approx(-6, abs=1e-9)


def test_pickle_manifest_omits_large_diagnostic_key_lists(ternary_table):
    original = AlexHull.from_table(ternary_table, FLAGS)
    path = ternary_table.parent / "manifest.json"
    source = json.loads(path.read_bytes())
    source["diagnostics"] = {"oxide_type_changed": [f"diagnostic-entry-{i}" for i in range(2000)]}
    source["potcar_inventory"] = [{"hash": "f" * 64}] * 2000
    path.write_text(json.dumps(source))
    hull = AlexHull.from_table(ternary_table, FLAGS)
    assert hull.manifest["table_manifest"] == original.manifest["table_manifest"]
    assert hull.manifest["table_manifest_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert hull.manifest["table_manifest_sha256"] != original.manifest["table_manifest_sha256"]
    restored = pickle.loads(pickle.dumps(hull))
    assert len(json.dumps(restored.manifest)) < 4096
    assert len(pickle.dumps(hull)) == len(pickle.dumps(original))
    assert "diagnostic-entry" not in json.dumps(restored.manifest)


def test_negative_distance_and_pymatgen_kwargs(workdir):
    path = write_table(workdir / "reference", [row("Li", 0), row("O2", 0), row("Li2O", -6)])
    hull = AlexHull.from_table(path, frozenset())
    candidate = ComputedEntry("Li2O", -8, correction=-1)
    assert hull.get_e_above_hull(candidate, allow_negative=True) == pytest.approx(-1, abs=1e-9)
    assert hull.get_e_above_hull(candidate, allow_negative=True, check_stable=False) == pytest.approx(-1, abs=1e-9)
    assert hull.get_hull_energy_per_atom(candidate.composition) == pytest.approx(-2, abs=1e-9)
    assert len(hull._cache[("Li", "O")][1]) == 3
    with pytest.raises(ValueError, match="No valid decomposition"):
        hull.get_e_above_hull(candidate)
    assert hull.get_e_above_hull(candidate, on_error="ignore") is None
    with pytest.warns(UserWarning, match="No valid decomposition"):
        assert hull.get_e_above_hull(candidate, on_error="warn") is None
    # A later candidate still sees the original hull, even after a lower-energy candidate.
    assert hull.get_e_above_hull(ComputedEntry("Li2O", -3)) == pytest.approx(1, abs=1e-9)


def test_known_bad_excludes_only_exclusion_actions(workdir):
    rows = [row("Li", 0), row("O2", 0), row("Li2O", -6, bad="retain_flagged"), row("LiO", -12, bad="exclude")]
    path = write_table(workdir / "reference", rows)
    hull = AlexHull.from_table(path, frozenset({"known_bad"}))
    assert hull.manifest["n_entries"] == 3
    assert hull.get_hull_energy_per_atom(Composition("Li2O")) == pytest.approx(-2, abs=1e-9)


def test_unsupported_elements_fail_before_build(ternary_table, monkeypatch):
    hull = AlexHull.from_table(ternary_table, FLAGS)

    def forbidden_build(*args, **kwargs):
        pytest.fail("A diagram must not be built without all unary terminals")

    monkeypatch.setattr("orbitol.alex_hull.linprog", forbidden_build)
    with pytest.raises(ValueError, match=r"Unsupported elements.*Si"):
        hull.get_hull_energy_per_atom(Composition("LiSi"))
    with pytest.raises(ValueError, match=r"Unsupported elements.*Si"):
        hull.get_e_above_hull(ComputedEntry("Si", -1), allow_negative=True)
    silicon = Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]])
    energy, reason = compute_e_above_hull_mp2020_like(hull, silicon, -1)
    assert energy is None
    assert reason == "mp2020_exception:ValueError"
    assert hull._cache == {}


def test_terminal_removed_by_flag_is_unsupported(workdir):
    path = write_table(workdir / "reference", [
        row("Li", 0), row("O2", 0, unlisted=True), row("Li2O", -6), row("Si", 0, unlisted=True),
    ])
    assert AlexHull.from_table(path, frozenset()).manifest["missing_terminals"] == []
    hull = AlexHull.from_table(path, frozenset({"potcar_align"}))
    assert hull.manifest["missing_terminals"] == ["O"]
    with pytest.raises(ValueError, match=r"no terminal reference.*O"):
        hull.get_hull_energy_per_atom(Composition("Li2O"))


def test_nan_terminal_is_reported_as_missing(workdir):
    path = write_table(workdir / "reference", [
        row("Li", 0), row("O2", 0, recorrected=np.nan), row("Li2O", -6),
    ], recorrect_failures=1)
    hull = AlexHull.from_table(path, frozenset({"recorrect"}))
    assert hull.manifest["n_dropped_nan"] == 1
    assert hull.manifest["missing_terminals"] == ["O"]


def test_cache_evicts_oldest_system_at_capacity(ternary_table):
    hull = AlexHull.from_table(ternary_table, FLAGS)
    expected = hull.get_hull_energy_per_atom(Composition("Li"))
    oldest = hull._cache[("Li",)]
    # Populate unused slots without building thousands of diagrams in a CPU test.
    hull._cache.update({("unused", str(i)): oldest for i in range(4095)})
    assert hull.get_hull_energy_per_atom(Composition("Li")) == expected
    hull.get_hull_energy_per_atom(Composition("Mg"))
    assert len(hull._cache) == 4096
    assert ("Li",) not in hull._cache
    assert hull.get_hull_energy_per_atom(Composition("Li")) == expected
    assert hull._cache[("Li",)] is not oldest
    assert len(hull._cache) == 4096
    assert ("unused", "0") not in hull._cache


def test_pickle_is_self_contained_and_drops_cache(ternary_table, workdir):
    hull = AlexHull.from_table(ternary_table, FLAGS)
    cold = pickle.dumps(hull, protocol=pickle.HIGHEST_PROTOCOL)
    comp = Composition("LiMgO2")
    candidate = ComputedEntry(comp, -50)
    expected = hull.get_e_above_hull(candidate, allow_negative=True)
    assert hull._cache
    assert pickle.dumps(hull, protocol=pickle.HIGHEST_PROTOCOL) == cold
    path = workdir / "reference.pkl"
    path.write_bytes(cold)
    # The loader must not reopen either source artifact.
    ternary_table.unlink()
    (ternary_table.parent / "manifest.json").unlink()
    restored = load_phase_diagram(path)
    assert isinstance(restored, AlexHull)
    assert restored._cache == {}
    assert restored.manifest == hull.manifest
    assert restored.get_hull_energy_per_atom(comp) == hull.get_hull_energy_per_atom(comp)
    assert restored.get_e_above_hull(candidate, allow_negative=True) == expected


def test_v1_legacy_entry_is_self_consistent_with_core(workdir):
    # Legacy POTCAR entries have zero shift and are not unlisted in the table.
    structure = Structure(
        Lattice.cubic(5), ["Li", "Li", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
    )
    raw = -12.0
    compat = MaterialsProject2020Compatibility(check_potcar=False)
    entry = ComputedStructureEntry(
        structure, raw, parameters=_build_mp2020_parameters(structure.composition, compat),
    )
    corrected = compat.process_entry(entry.copy(), on_error="raise")
    assert corrected is not None and corrected.correction != 0
    path = write_table(workdir / "reference", [
        row("Li", 0), row("O2", 0),
        row("Li2O", raw, stored=0, recorrected=corrected.correction, shift=0),
    ])
    hull = AlexHull.from_table(path, FLAGS)
    energy, reason = compute_e_above_hull_mp2020_like(hull, structure, raw, mp2020_compat=compat)
    assert reason is None
    assert energy == pytest.approx(0, abs=1e-12)
    original = AlexHull.from_table(path, frozenset())
    old_energy, reason = compute_e_above_hull_mp2020_like(original, structure, raw, mp2020_compat=compat)
    assert reason is None
    assert old_energy == pytest.approx(corrected.correction / len(structure), abs=1e-12)


def test_nan_reference_drops_count_only_assembled_kept_energies(workdir):
    path = write_table(workdir / "reference", [
        row("Li", 0), row("O2", 0), row("LiO", -6, recorrected=np.nan),
        row("Si", np.nan, bad="exclude"),
    ], recorrect_failures=1)
    assert AlexHull.from_table(path, frozenset({"known_bad"})).manifest["n_dropped_nan"] == 0
    hull = AlexHull.from_table(path, frozenset({"known_bad", "recorrect"}))
    assert hull.manifest["n_dropped_nan"] == 1
    assert hull.manifest["n_entries"] == 2
    assert hull.get_hull_energy_per_atom(Composition("LiO")) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize(("compound", "flags"), [
    (row("LiO", np.nan), frozenset()),
    (row("LiO", -6, recorrected=np.nan), frozenset({"recorrect"})),
])
def test_nan_drops_reject_zero_recorded_recorrection_failures(workdir, compound, flags):
    path = write_table(workdir / "reference", [row("Li", 0), row("O2", 0), compound])
    with pytest.raises(ValueError, match=r"NaN reference energies.*counts.recorrect_failures is 0"):
        AlexHull.from_table(path, flags)


def test_invalid_flags_and_compositions(workdir):
    path = write_table(workdir / "reference", [row("Li", 0)])
    with pytest.raises(ValueError, match="Unknown AlexHull flags: typo"):
        AlexHull.from_table(path, frozenset({"typo"}))
    invalid = row("Li2O", -6)
    invalid["natoms"] = 2
    write_table(path.parent, [invalid])
    with pytest.raises(ValueError, match="Invalid composition amounts or natoms"):
        AlexHull.from_table(path, frozenset())


def test_table_requires_single_file_and_required_columns(workdir):
    path = write_table(workdir / "reference", [row("Li", 0)])
    with pytest.raises(ValueError, match="Expected a single Parquet file"):
        AlexHull.from_table(path.parent, frozenset())
    table = pd.read_parquet(path).drop(columns="stored_correction")
    table.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Missing AlexHull table columns: stored_correction"):
        AlexHull.from_table(path, frozenset())


@pytest.mark.parametrize("variant", ["original", "v1", "recorrect+known_bad"])
def test_cli_pickle_loads_without_torch_or_core(ternary_table, workdir, variant):
    output = workdir / "cli.pkl"
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               MPLCONFIGDIR=str(workdir / "matplotlib"))
    result = subprocess.run(
        [PYTHON, "-m", "orbitol.alex_hull", "--table", str(ternary_table),
         "--variant", variant, "--out", str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    manifest = json.loads(result.stdout)
    assert load_phase_diagram(output).manifest == manifest
    assert manifest["variant"] == ("known_bad+recorrect" if "+" in variant else variant)
    subprocess.run(
        [PYTHON, "-c", "import pickle, sys; "
         "hull = pickle.load(open(sys.argv[1], 'rb')); "
         "assert type(hull).__module__ == 'orbitol.alex_hull'; "
         "assert 'torch' not in sys.modules; assert 'src' not in sys.modules", str(output)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
