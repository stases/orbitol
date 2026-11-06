"""CPU parity with the pinned *train-time* scalar evaluation path."""
from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedEntry

import orbitol.core  # noqa: F401
from orbitol.contracts import DATA_ROOT
from orbitol.eval_adapter import (
    CapturingStabilityLogger, DNGEvaluator, ThermoSettings, _ConfiguredLogger,
    _sample_items,
)
from orbitol.symmetry_panel import (
    SymmetryPanel, bootstrap_ci, crystal_system_tv, detect_group,
    distribution_metrics, group_histogram, supergroup_closure,
)
from src.data.mp20_tokens import lattice_to_Y, tokens_to_structure
from src.eval.dng_eval import compute_evaluator_metrics, compute_novelty_metrics, compute_sun_metrics
from src.eval.stability import _compute_thermo_metrics
from src.utils.dataset import dataset_to_structures


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def workdir():
    # Keep even test artifacts inside the checkout.
    parent = ROOT / ".pytest_cache"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as name:
        yield Path(name)


@pytest.fixture
def rutile():
    return Structure.from_spacegroup(
        136, Lattice.tetragonal(4.594, 2.959),
        ["Ti", "O"], [[0, 0, 0], [0.305, 0.305, 0]],
    )


def pack(structures):
    a = torch.zeros((len(structures), 20), dtype=torch.long)
    f = torch.zeros((len(structures), 20, 3))
    y = torch.zeros((len(structures), 6))
    mask = torch.ones((len(structures), 20), dtype=torch.bool)
    for i, s in enumerate(structures):
        n = len(s)
        a[i, :n] = torch.tensor(s.atomic_numbers)
        f[i, :n] = torch.tensor(s.frac_coords)
        y[i] = torch.from_numpy(lattice_to_Y(s.lattice.abc, s.lattice.angles))
        mask[i, :n] = False
    return dict(A0=a, F1=f, Y1=y, pad_mask=mask)


@pytest.fixture
def tiny_root(workdir, rutile):
    processed = workdir / "data/processed"
    processed.mkdir(parents=True)
    items = _sample_items(pack([rutile]))
    for split in ("train", "val"):
        torch.save(items, processed / f"mp20_tokens_{split}.pt")
    return processed.parent


@pytest.mark.data
@pytest.mark.slow
def test_64_mp20_val_exact_train_time_equivalence(workdir, tiny_root):
    """Full MP-20 train novelty/val Wasserstein references, no MLIP.

    Only the expensive *extra* symmetry reference is replaced with a tiny
    panel. Full-train spglib cache building belongs on Slurm (~4+ minutes).
    No core functions are replaced. Every shared scalar is exactly equal;
    there is no stochastic numerical tolerance in this CPU path.
    """
    evaluator = DNGEvaluator(DATA_ROOT, device="cpu", thermo=None, sample_count=2048, sun_k=512,
                             symmetry_cache=workdir / "unused_full_reference.json")
    train = torch.load(Path(DATA_ROOT) / "processed/mp20_tokens_train.pt", weights_only=False)
    val = torch.load(Path(DATA_ROOT) / "processed/mp20_tokens_val.pt", weights_only=False)
    # Build the baseline references independently, following
    # train_crystalite.py:345-365. Reusing the adapter's references here would
    # hide an incorrect choice of split or a truncated novelty reference.
    baseline_train = dataset_to_structures(train)
    baseline_val = dataset_to_structures(val)
    assert len(evaluator.novelty_ref_structs) == len(baseline_train) == len(train)
    assert len(evaluator.ref_structs) == len(baseline_val) == len(val)
    tiny_items = torch.load(tiny_root / "processed/mp20_tokens_train.pt", weights_only=False)
    evaluator.symmetry_panel = SymmetryPanel(
        tiny_root / "processed/mp20_tokens_train.pt", tiny_items, workdir / "tiny_reference.json",
    )
    items = val[:64]
    samples = {k: torch.stack([s[k] for s in items]) for k in ("A0", "F1", "Y1", "pad_mask")}

    # Literal scalar call sequence from sample_runtime.py:494-541, called by
    # train_crystalite.py:720. Do not compare against checkpoint evaluator keys.
    validity = compute_evaluator_metrics(
        items, enabled=True, limit=2048, ref_structs=baseline_val,
        sample_seed=123, include_wasserstein=True, wasserstein_max_samples=2048,
    )
    novelty = compute_novelty_metrics(
        items, baseline_train, limit=2048, minimum_nary=1,
    )
    expected = {
        "sample_ema/valid_rate": validity.valid_rate,
        "sample_ema/comp_valid_rate": validity.comp_valid_rate,
        "sample_ema/struct_valid_rate": validity.struct_valid_rate,
        "sample_ema/wdist_nary": validity.dist_metrics["sample_dist/wdist_nary"],
        "sample_ema/wdist_density_mass": validity.dist_metrics["sample_dist/wdist_density_mass"],
        "sample_ema/unique_rate": novelty.unique_rate,
        "sample_ema/novel_rate": novelty.novel_rate,
        "sample_ema/un_rate": novelty.un_rate,
    }
    actual = evaluator.evaluate(samples, step=20000)
    shared = {k: v for k, v in actual.items() if k.startswith("sample_ema/")}
    assert shared == expected
    assert len(shared) == 8


class MockRelaxer:
    """Element-indexed, deterministic energies; one divergence and one hull failure."""

    energy = {"Li": -0.01, "Na": 0.05, "K": 0.2, "Al": 0.0, "Ti": -2.0}

    def __init__(self):
        self.calls = []

    def relax(self, structure, *, steps, verbose):
        assert steps == 200 and verbose is False
        symbol = structure[0].specie.symbol
        self.calls.append(symbol)
        if symbol == "Mg":
            raise RuntimeError("force divergence")
        final = structure.copy()
        final.scale_lattice(structure.volume * 1.01)
        return {"final_structure": final,
                "trajectory": SimpleNamespace(energies=[self.energy[symbol] * len(final)])}

    def relax_many(self, structures, *, steps, verbose):
        offenders = [i for i, s in enumerate(structures) if s[0].specie.symbol == "Mg"]
        if offenders:
            raise RuntimeError(f"force divergence systems {offenders}")
        return [self.relax(s, steps=steps, verbose=verbose) for s in structures]


@pytest.fixture
def settings(workdir):
    # Al is intentionally absent: its endpoint is captured even though the
    # actual core hull calculation fails. No mocked hull/metric functions.
    pd = PhaseDiagram([ComputedEntry(el, 0.0) for el in ("Li", "Na", "K", "Mg", "Ti", "O")])
    path = workdir / "phase_diagram.pkl"
    with path.open("wb") as f:
        pickle.dump(pd, f)
    return replace(ThermoSettings(nequip_backend="torchscript", thermo_relax_steps=200), thermo_ppd_mp=str(path),
                   thermo_stability_device="cpu", thermo_stability_batch=2)


def metal(symbol):
    return Structure(Lattice.cubic(4.0), [symbol], [[0.13, 0.24, 0.31]])


@pytest.mark.parametrize("mode", ["batch", "sequential"])
@pytest.mark.parametrize("method", ["mp2020_like", "uncorrected"])
def test_hook_preserves_original_aggregate_metrics(settings, mode, method):
    settings = replace(settings, nequip_relax_mode=mode, thermo_ehull_method=method)
    structs = [metal(s) for s in ("Li", "Na", "K", "Mg", "Al")]
    unhooked = _ConfiguredLogger(settings, MockRelaxer())
    hooked = CapturingStabilityLogger(settings, MockRelaxer())
    hooked.begin({id(s): (i + 10, True) for i, s in enumerate(structs)})
    kwargs = dict(tag="sample_ema/sun", step=20000, enabled=True)
    expected = _compute_thermo_metrics(unhooked, structs, **kwargs)
    actual = _compute_thermo_metrics(hooked, structs, **kwargs)
    assert actual == expected  # bit-for-bit, including corrected hulls and failure counters
    assert actual["sample_ema/sun/thermo_checked_total"] == 5
    assert actual["sample_ema/sun/thermo_success_total"] == 3
    assert actual["sample_ema/sun/thermo_failed_total"] == 2
    assert actual["sample_ema/sun/thermo_divergence_total"] == 1
    records = {r.sample_index: r for r in hooked.records()}
    assert set(records) == {10, 11, 12, 14}
    assert records[10].stable and records[11].metastable and not records[11].stable
    assert not records[12].metastable and records[14].failure_reason
    assert records[10].final_structure.volume == pytest.approx(structs[0].volume * 1.01)
    assert hooked.thermo_cfg.batch_size == 2  # partial final batch restored by core


@pytest.mark.parametrize("sun_k", [512, 0])
def test_joint_sun_hand_count_and_original_indices(tiny_root, workdir, settings, rutile, sun_k):
    structs = [metal(s) for s in ("Li", "Li", "Li", "Na", "K", "Mg", "Al")] + [rutile]
    samples = pack(structs)
    samples["A0"][0, 0] = 0  # conversion failure before every retained sample index
    evaluator = DNGEvaluator(tiny_root, device="cpu", thermo=settings,
                             relaxer=MockRelaxer(), sun_k=sun_k, sample_count=2048,
                             no_thermo_before_steps=20000,
                             symmetry_cache=workdir / "reference.json")
    actual = evaluator.evaluate(samples, step=20000, requested_G=np.ones(8, dtype=int))
    # Historical subset: Li representative, Na, K, Mg divergence, Al hull failure.
    # All-attempt protocol also relaxes duplicate Li and nonnovel rutile.
    denominator = 5 if sun_k > 0 else 8
    assert actual["joint/SUN"] == 1 / denominator
    assert actual["joint/MSUN"] == 2 / denominator
    assert actual["joint/SUN_count"] == 1
    assert actual["joint/MSUN_count"] == 2
    assert actual["joint/checked_count"] == denominator
    assert actual["joint/UN_count"] == (5 if sun_k > 0 else 4)
    assert actual["joint/relaxed_count"] == (4 if sun_k > 0 else 6)
    assert set(evaluator.attempted_indices) == ({1, 3, 4, 5, 6} if sun_k > 0 else set(range(1, 8)))
    records = {r.sample_index: r for r in evaluator.relaxed_records}
    assert records[6].failure_reason  # hull failure still has a relaxed endpoint
    assert actual["sym/relaxed/0.1/retention"] == 1.0
    assert actual["sym/raw/0.1/failure_count"] == 1.0
    if sun_k > 0:
        assert actual["sample_ema/SUN"] == (5 / 7) * (1 / 5)
        assert actual["sample_ema/MSUN"] == (5 / 7) * (2 / 5)
    else:
        assert not records[2].is_un and not records[7].is_un
        assert actual["sample_ema/all_relaxed/thermo_checked_total"] == 7
        assert actual["sample_ema/stability/hull_ok_count"] == 5
        assert actual["sample_ema/SUN"] == actual["sample_ema/SUN_raw_UN"] == 1 / 8
    # Diagnostics observe actual per-index core validity and failure reasons;
    # computing them must not change pinned/headline bytes on re-evaluation.
    before = json.dumps(actual, sort_keys=True).encode()
    diagnostics = {row["pool_index"]: row for row in evaluator.relaxation_diagnostics()}
    assert set(diagnostics) == set(evaluator.attempted_indices)
    validity = compute_evaluator_metrics(
        _sample_items(samples), limit=2048, ref_structs=[], sample_seed=123,
        include_wasserstein=False,
    )
    for crystal in validity.pred_crys_list:
        if crystal.sample_idx in diagnostics:
            row = diagnostics[crystal.sample_idx]
            assert row["raw_valid"] == crystal.valid
            assert row["raw_struct_valid"] == crystal.struct_valid
            assert row["raw_comp_valid"] == crystal.comp_valid
    assert diagnostics[5]["failure_reason"] == "relax_divergence"
    assert diagnostics[5]["relaxation_outcome"] == "relaxation_failed"
    assert diagnostics[6]["relaxation_outcome"] == "hull_failed"
    assert diagnostics[1]["final_max_force"] is None
    assert diagnostics[1]["initial_volume_per_atom"] == pytest.approx(structs[1].volume)
    assert diagnostics[1]["relaxed_volume_per_atom"] == pytest.approx(structs[1].volume * 1.01)
    repeated = evaluator.evaluate(samples, step=20000, requested_G=np.ones(8, dtype=int))
    assert json.dumps(repeated, sort_keys=True).encode() == before
    # A later early-step evaluation must not leak records from the previous call.
    early = evaluator.evaluate(samples, step=19999)
    assert early["joint/checked_count"] == 0
    assert np.isnan(early["joint/SUN"])
    assert not any("thermo_" in key for key in early)
    assert evaluator.relaxed_records == []
    assert evaluator.relaxation_diagnostics() == []


def test_seeded_sun_selection_matches_core(tiny_root, workdir, settings):
    samples = pack([metal(s) for s in ("Li", "Na", "K", "Mg", "Al")])
    evaluator = DNGEvaluator(tiny_root, device="cpu", thermo=settings,
                             relaxer=MockRelaxer(), sun_k=2, sample_count=2048,
                             symmetry_cache=workdir / "reference.json")
    novelty = compute_novelty_metrics(_sample_items(samples), evaluator.novelty_ref_structs,
                                      limit=2048, minimum_nary=1)
    expected = compute_sun_metrics(
        novelty.novelty_metrics, thermo_logger=_ConfiguredLogger(settings, MockRelaxer()),
        tag="sample_ema", step=40000, enabled=True, base_seed=40123, sun_target=2,
        show_progress=False,
    )
    actual = evaluator.evaluate(samples, step=40000)
    for key, value in {**expected.thermo_metrics, **expected.summary_metrics}.items():
        assert actual[key] == value
    indices = np.random.default_rng(40123 + sum(map(ord, "sample_ema"))).choice(5, size=2, replace=False)
    assert set(evaluator.attempted_indices) == set(indices)


def test_rutile_perturbation_failures_and_retention(rutile):
    assert detect_group(rutile, 0.1) == 136  # P4_2/mnm
    perturbed = rutile.copy()
    for i, offset in enumerate(np.random.default_rng(17).normal(0, 0.06, (len(rutile), 3))):
        perturbed.translate_sites(i, offset, frac_coords=True)
    assert detect_group(perturbed, 0.01) == 1
    assert detect_group(None, 0.1) == 0
    result = distribution_metrics([136, 0], group_histogram([136]))
    assert result["p1_share"] == 0
    assert result["failure_count"] == 1
    assert result["failure_rate"] == 0.5
    assert result["crystal_system/tetragonal"] == 1
    assert result["crystal_system_tv"] == result["space_group_jsd"] == 0
    assert result["effective_groups"] == 1
    closure = supergroup_closure()
    assert 136 in closure[1] and 136 in closure[136] and 1 not in closure[136]
    assert 225 in closure[221]  # a k-supergroup relation, not numerical ordering


def test_reference_cache_roundtrip_and_invalidation(tiny_root, workdir, monkeypatch):
    import orbitol.symmetry_panel as panel_module

    source = tiny_root / "processed/mp20_tokens_train.pt"
    items = torch.load(source, weights_only=False)
    path = workdir / "reference.json"
    panel = SymmetryPanel(source, items, path)
    original = panel.reference
    assert original["0.1"][136] == original["0.01"][136] == 1
    detector = panel_module.detect_group
    calls = []

    def counted(s, p):
        calls.append(p)
        return detector(s, p)

    monkeypatch.setattr(panel_module, "detect_group", counted)
    assert SymmetryPanel(source, items, path).reference == original
    assert calls == []
    data = json.loads(path.read_text())
    data["metadata"]["spglib"] = "stale"
    path.write_text(json.dumps(data))
    assert SymmetryPanel(source, items, path).reference == original
    assert calls == [0.1, 0.01]


def test_bootstrap_joint_and_tv():
    flags = np.asarray([1, 0, 1, 0, 0, 1])
    rng = np.random.default_rng(7)
    expected = np.percentile([flags[rng.integers(0, 6, 6)].mean() for _ in range(1000)], [2.5, 97.5])
    assert bootstrap_ci(lambda indices: flags[indices].mean(), 6, seed=7) == tuple(expected)
    groups = np.asarray([1, 1, 136, 136, 225, 225])
    reference = group_histogram([1, 136, 225])
    low, high = bootstrap_ci(lambda indices: crystal_system_tv(groups[indices], reference), 6)
    assert 0 <= low < high <= 1
    assert all(np.isnan(v) for v in bootstrap_ci(lambda _: 0, 0))


def test_cli_without_thermo(tiny_root, workdir, rutile):
    samples = workdir / "samples.pt"
    torch.save(pack([rutile]), samples)
    out = workdir / "run"
    result = subprocess.run([
        sys.executable, "-m", "orbitol.eval_adapter",
        "--samples", str(samples), "--out", str(out), "--data-root", str(tiny_root),
        "--symmetry-reference", str(out / "symmetry_reference.json"),
    ], env={**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        cwd=ROOT, capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["sample_ema/valid_rate"] == 1
    assert metrics["joint/SUN"] is None
    assert (out / "symmetry_reference.json").is_file()
    config = json.loads((out / "evaluation_settings.json").read_text())
    assert config["thermo"] is None and config["step"] == 20000


def test_missing_cache_never_preprocesses(workdir):
    with pytest.raises(FileNotFoundError, match="no preprocessing"):
        DNGEvaluator(workdir, device="cpu", thermo=None)
    assert not (workdir / "processed").exists()


@pytest.mark.parametrize("mode", ["batch", "sequential"])
@pytest.mark.parametrize("injected", [False, True])
def test_torchscript_lazy_construction(tiny_root, workdir, settings, monkeypatch, mode, injected):
    import orbitol.nequip_relax as relax_module

    calls = []

    class ArbitraryFailure(MockRelaxer):
        def relax(self, structure, **kwargs):
            if structure[0].specie.symbol == "Mg":
                raise KeyError("bad geometry")
            return super().relax(structure, **kwargs)

        def relax_many(self, structures, **kwargs):
            return [self.relax(s, **kwargs) for s in structures]

    def factory(path, device, **kwargs):
        calls.append((path, device, kwargs))
        return ArbitraryFailure()

    monkeypatch.setattr(relax_module, f"make_{mode}_relaxer", factory)
    settings = replace(settings, nequip_relax_mode=mode)
    backend = MockRelaxer() if injected else None
    evaluator = DNGEvaluator(tiny_root, device="cpu", thermo=settings, relaxer=backend,
                             sample_count=2048, sun_k=512,
                             symmetry_cache=workdir / "reference.json")
    assert settings.nequip_model_kind == "torchscript"
    assert calls == []
    samples = pack([metal(s) for s in ("Li", "Mg", "Na")])
    metrics = evaluator.evaluate(samples, step=20000)
    assert metrics["sample_ema/sun/thermo_success_total"] == 2
    assert metrics["sample_ema/sun/thermo_failed_total"] == 1
    assert {r.sample_index for r in evaluator.relaxed_records} == {0, 2}
    evaluator.evaluate(samples, step=40000)
    if injected:
        assert calls == []
    else:
        assert len(calls) == 1
        assert calls[0][0:2] == (settings.nequip_ts_path, "cpu")
        assert calls[0][2]["max_force_abort"] == 1e6
        if mode == "sequential":
            assert calls[0][2]["fmax"] == .005


def test_thermo_model_kind_validation():
    with pytest.raises(ValueError, match="model kind"):
        ThermoSettings(nequip_model_kind="unknown")
    assert ThermoSettings(nequip_model_kind="aoti").to_core().nequip_compile_path


def test_robust_batch_records_mixed_failure_reasons_individually(settings):
    from orbitol.nequip_relax import RobustRelaxer

    class MixedFailures(MockRelaxer):
        def relax(self, structure, **kwargs):
            if structure[0].specie.symbol == "Na":
                raise ValueError("unsupported geometry")
            return super().relax(structure, **kwargs)

        def relax_many(self, structures, **kwargs):
            return [self.relax(s, **kwargs) for s in structures]

    structs = [metal(s) for s in ("Li", "Mg", "Na", "K")]
    outputs = []
    for mode in ("sequential", "batch"):
        logger = CapturingStabilityLogger(
            replace(settings, nequip_relax_mode=mode, thermo_stability_batch=4),
            RobustRelaxer(MixedFailures()),
        )
        logger.begin({id(s): (i, True) for i, s in enumerate(structs)})
        outputs.append(_compute_thermo_metrics(logger, structs, tag="test", step=20000, enabled=True))
        assert {r.sample_index for r in logger.records()} == {0, 3}
    assert outputs[0] == outputs[1]
    assert outputs[1]["test/thermo_fail/relax_divergence"] == 1
    assert outputs[1]["test/thermo_fail/relax_non_finite"] == 1


def test_rows_without_atoms_count_as_invalid_samples(tiny_root, workdir, rutile):
    # A failed Orbitol decode is a row without atoms. The pinned Crystal class cannot
    # build an empty composition; the row must count as the core's invalid stand-in.
    samples = pack([rutile, rutile])
    samples["pad_mask"][1] = True
    evaluator = DNGEvaluator(tiny_root, device="cpu", thermo=None, sample_count=2,
                             symmetry_cache=workdir / "reference.json")
    metrics = evaluator.evaluate(samples, step=20000, requested_G=np.full(2, 136))
    assert metrics["sample_ema/valid_rate"] == 0.5
    assert metrics["sample_ema/struct_valid_rate"] == 0.5
    assert evaluator.raw_validity[1]["raw_valid"] is False
    assert evaluator.raw_validity[0]["raw_valid"] is True


def test_thermo_defaults_are_stored_and_round_trip():
    from dataclasses import asdict
    from orbitol import contracts as C
    from orbitol.eval_adapter import ThermoSettings
    settings = asdict(ThermoSettings())
    assert settings["thermo_ppd_mp"] == C.dataset_spec("mp20").phase_diagram
    assert settings["nequip_ts_path"] == settings["nequip_compile_path"] == C.local_path("ORBITOL_NEQUIP_TS")
    assert settings["nequip_oeq_path"] == ""
    assert asdict(ThermoSettings(**settings)) == settings
    explicit = {**settings, "thermo_ppd_mp": "/saved/hull", "nequip_ts_path": "/saved/ts",
                "nequip_compile_path": "/saved/compile", "nequip_oeq_path": "/saved/oeq"}
    assert asdict(ThermoSettings(**explicit)) == explicit
