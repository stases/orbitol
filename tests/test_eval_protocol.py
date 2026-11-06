"""CPU protocol gates: real core U/N/hulls, stub sampling and relaxation."""
from dataclasses import asdict, fields, replace
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.core import Lattice, Structure

import orbitol.core  # noqa: F401
from orbitol import contracts as C, nequip_relax, posthoc_eval, posthoc_queue, train
from orbitol.eval_adapter import DNGEvaluator, ThermoSettings, _sample_items
from orbitol.eval_diagnostics import variant_metrics
from orbitol.eval_protocol import complete_score, relax_all, score, score_raw
from orbitol.symmetry_panel import SymmetryPanel
from test_eval_adapter import metal, pack


class Relaxer:
    energy = {"Li": -.01, "Na": .1, "K": .2, "Al": 0., "Cu": 0.}

    def __init__(self):
        self.calls = []

    def relax_many(self, structures, *, steps, verbose):
        assert steps == 500 and verbose is False
        self.calls.extend(s[0].specie.symbol for s in structures)
        offenders = [i for i, s in enumerate(structures) if s[0].specie.symbol == "Mg"]
        if offenders:
            raise RuntimeError(f"force divergence systems {offenders}")
        return [dict(final_structure=s.copy(), nsteps=steps,
                     trajectory=SimpleNamespace(energies=[self.energy[s[0].specie.symbol] * len(s)]))
                for s in structures]


@pytest.fixture
def protocol(tmp_path):
    data = tmp_path / "data"
    (data / "processed").mkdir(parents=True)
    reference_items = _sample_items(pack([metal("Cu")]))
    for split in ("train", "val"):
        torch.save(reference_items, data / "processed" / f"mp20_tokens_{split}.pt")
    reference = tmp_path / "reference.json"
    SymmetryPanel(data / "processed/mp20_tokens_train.pt", reference_items, reference).reference
    phase = tmp_path / "phase.pkl"
    with phase.open("wb") as stream:
        pickle.dump(PhaseDiagram([ComputedEntry(el, 0.) for el in ("Li", "Na", "K", "Mg", "Cu")]), stream)
    model = tmp_path / "mock.nequip.pth"
    model.write_bytes(b"stub OEQ artifact")
    rows = pack([metal(el) for el in ("Li", "Li", "Li", "Na", "K", "Mg", "Al", "Cu")])
    rows["A0"][0, 0] = 0  # unconstructable attempt stays in the denominator
    settings = ThermoSettings(thermo_stability_device="cpu", thermo_ppd_mp=str(phase),
                              thermo_ehull_method="uncorrected", thermo_stability_batch=3,
                              nequip_oeq_path=str(model))
    return SimpleNamespace(root=tmp_path, data=data, reference=reference, phase=phase,
                           model=model, rows=rows, settings=settings)


def evaluator(protocol, relaxer=None, **kwargs):
    return DNGEvaluator(protocol.data, device="cpu", sample_count=8, no_thermo_before_steps=0,
                        thermo=protocol.settings, relaxer=relaxer or Relaxer(),
                        symmetry_cache=protocol.reference, **kwargs)


@pytest.mark.parametrize("thermo", [False, True])
def test_raw_phase_and_completion_equal_score(protocol, monkeypatch, thermo):
    import orbitol.eval_protocol as ep

    runner = evaluator(protocol)
    items = _sample_items(protocol.rows)
    records = relax_all(items, Relaxer(), 500)[::-1] if thermo else None
    kwargs = dict(references=runner.references, settings=runner.settings, tag=runner.tag)
    requested = [221] * len(items)
    expected = score(items, records, ppd=runner.ppd, step=0, requested_G=requested, **kwargs)
    detections = []
    detect = ep.detect_many

    def counted(structures):
        detections.append(len(structures))
        return detect(structures)

    monkeypatch.setattr(ep, "detect_many", counted)
    prepared = score_raw(items, requested_G=requested, **kwargs)
    actual = complete_score(prepared, records, ppd=runner.ppd, step=0, **kwargs)
    assert json.dumps(actual) == json.dumps(expected)
    assert detections == [len(items), len(items)]


def test_training_keeps_phase_diagram_cached_in_parent(protocol, monkeypatch):
    import orbitol.eval_adapter as adapter

    runner = evaluator(protocol)
    loads = []
    load = adapter.load_phase_diagram

    def counted(path):
        loads.append(path)
        return load(path)

    monkeypatch.setattr(adapter, "load_phase_diagram", counted)
    first = runner.evaluate(protocol.rows, step=0)
    assert runner._ppd is not None
    second = runner.evaluate(protocol.rows, step=0)
    assert json.dumps(first) == json.dumps(second)
    assert loads == [str(protocol.phase)]


def test_all_constructed_relaxed_and_full_attempt_denominators(protocol):
    backend = Relaxer()
    runner = evaluator(protocol, backend)
    metrics = runner.evaluate(protocol.rows, step=2, requested_G=torch.ones(8))
    assert set(backend.calls) == {"Li", "Na", "K", "Mg", "Al", "Cu"}
    assert set(runner.attempted_indices) == set(range(1, 8))
    records = {r.sample_index: r for r in runner.relaxed_records}
    assert not records[2].is_un and not records[7].is_un  # duplicate and training reference
    assert records[3].metastable and records[3].e_above_hull == .1
    assert records[6].failure_reason  # endpoint retained after hull failure
    assert metrics["sample_ema/SUN"] == metrics["joint/SUN"] == 1 / 8
    assert metrics["sample_ema/MSUN"] == metrics["joint/MSUN"] == 2 / 8
    assert metrics["sample_ema/SUN_raw_UN"] == 1 / 8
    assert metrics["sample_ema/MSUN_raw_UN"] == 2 / 8
    assert metrics["joint/checked_count"] == 8
    prefix = "sample_ema/stability/"
    expected = dict(attempt_count=8, constructed_count=7, construction_failure_count=1,
                    relaxed_ok_count=6, relax_failure_count=1, hull_ok_count=5, hull_failure_count=1,
                    stable_rate=3 / 8, metastable_rate=4 / 8,
                    stable_rate_relaxed_ok=3 / 6, metastable_rate_relaxed_ok=4 / 6)
    for key, value in expected.items():
        assert metrics[prefix + key] == value
    hulls = [-.01, -.01, .1, .2, 0.]
    for name, percentile in (("median", 50), ("p10", 10), ("p90", 90)):
        assert metrics[prefix + "e_above_hull_" + name] == pytest.approx(np.percentile(hulls, percentile))
    assert "sym/relaxed/0.1/retention" in metrics
    assert len(runner.relaxation_diagnostics()) == 7
    rows = runner.per_sample
    assert [row["index"] for row in rows] == list(range(8))
    assert rows[0]["relax_status"] == "not_constructed"
    assert rows[5]["relax_status"] == "failed" and rows[5]["reason"]
    assert rows[6]["relax_status"] == "ok" and rows[6]["hull_failure_reason"]
    assert [i for i, row in enumerate(rows) if row["SUN"]] == [1]
    assert [i for i, row in enumerate(rows) if row["MSUN"]] == [1, 3]
    assert rows[3]["raw_space_group_0.1"] == rows[3]["relaxed_space_group_0.1"] == 221


def test_all_failed_construction_and_no_thermo_do_not_reuse_records(protocol):
    runner = evaluator(protocol)
    runner.evaluate(protocol.rows, step=2)
    failed = {key: value.clone() for key, value in protocol.rows.items()}
    failed["pad_mask"][:] = True
    result = runner.evaluate(failed, step=3)
    assert result["sample_ema/SUN"] == result["sample_ema/MSUN"] == 0
    assert result["sample_ema/stability/construction_failure_count"] == 8
    assert result["sample_ema/stability/relax_failure_count"] == 0
    assert math.isnan(result["sample_ema/stability/stable_rate_relaxed_ok"])
    assert runner.relaxed_records == []
    runner.thermo = None
    early = runner.evaluate(protocol.rows, step=4)
    assert early["joint/checked_count"] == 0
    assert "sample_ema/SUN" not in early


@pytest.mark.parametrize("first_energy", [-.01, .1])
def test_merged_endpoints_use_first_index_even_when_later_endpoint_is_more_stable(protocol, first_energy):
    raw = [metal("Li"), Structure(Lattice.orthorhombic(4, 8, 16), ["Li"], [[0, 0, 0]])]

    class MergingRelaxer:
        def relax_many(self, structures, *, steps, verbose):
            assert len(structures) == 2 and steps == 500 and verbose is False
            return [dict(final_structure=metal("Li"), nsteps=steps,
                         trajectory=SimpleNamespace(energies=[energy])) for energy in (first_energy, -.01)]

    runner = evaluator(protocol, MergingRelaxer())
    metrics = runner.evaluate(pack(raw), step=0)
    assert metrics["sample_ema/un_rate"] == 1
    assert metrics["sample_ema/SUN"] == (0.5 if first_energy < 0 else 0)
    assert metrics["sample_ema/SUN_raw_UN"] == (1 if first_energy < 0 else 0.5)
    assert metrics["sample_ema/MSUN"] == 0.5
    assert metrics["sample_ema/MSUN_raw_UN"] == 1
    assert metrics["joint/UN_count"] == metrics["joint/MSUN_count"] == 1
    assert [row["raw_is_un"] for row in runner.per_sample] == [True, True]
    assert [row["relaxed_is_unique"] for row in runner.per_sample] == [True, False]
    assert [row["relaxed_is_novel"] for row in runner.per_sample] == [True, True]
    assert [r.is_un for r in runner.relaxed_records] == [True, False]
    # Persisted rows may arrive out of order. Winners follow original indices.
    items = _sample_items(pack(raw))
    records = relax_all(items, MergingRelaxer(), 500)
    reordered, rows = score(items, records[::-1], references=runner.references, ppd=runner.ppd,
                             settings=runner.settings, tag=runner.tag, step=0)
    assert reordered == metrics and rows == runner.per_sample


def test_endpoint_that_matches_reference_loses_novelty(protocol):
    class MatchingRelaxer:
        def relax_many(self, structures, *, steps, verbose):
            assert len(structures) == 1 and steps == 500 and verbose is False
            return [dict(final_structure=metal("Cu"), nsteps=steps,
                         trajectory=SimpleNamespace(energies=[0.]))]

    runner = evaluator(protocol, MatchingRelaxer())
    raw = Structure(Lattice.orthorhombic(4, 8, 16), ["Cu"], [[0, 0, 0]])
    metrics = runner.evaluate(pack([raw]), step=0)
    assert metrics["sample_ema/un_rate"] == 1
    assert metrics["sample_ema/SUN_raw_UN"] == metrics["sample_ema/MSUN_raw_UN"] == 1
    assert metrics["sample_ema/SUN"] == metrics["sample_ema/MSUN"] == 0
    assert metrics["sample_ema/stability/stable_rate"] == 1
    row = runner.per_sample[0]
    assert row["raw_is_novel"] and row["relaxed_is_unique"]
    assert not row["relaxed_is_novel"] and not row["relaxed_is_un"]


def test_endpoint_dedup_precedes_novelty_filtering(protocol):
    # The broad core tolerance matches these endpoints to each other, but only
    # the first to the cubic Cu reference. Raw novelty-then-dedup keeps index 1;
    # dedup-before-novelty must keep index 0 and then reject it as nonnovel.
    raw = [Structure(Lattice.orthorhombic(4, 4, c), ["Cu"], [[0, 0, 0]]) for c in (5, 6)]
    runner = evaluator(protocol)
    metrics = runner.evaluate(pack(raw), step=0)
    assert [row["raw_is_un"] for row in runner.per_sample] == [False, True]
    assert [row["relaxed_is_unique"] for row in runner.per_sample] == [True, False]
    assert [row["relaxed_is_novel"] for row in runner.per_sample] == [False, True]
    assert metrics["sample_ema/SUN_raw_UN"] == metrics["sample_ema/MSUN_raw_UN"] == .5
    assert metrics["sample_ema/SUN"] == metrics["sample_ema/MSUN"] == 0


def test_raw_invalid_but_constructed_samples_are_relaxed(protocol):
    invalid = Structure(Lattice.cubic(4), ["Li", "Li"], [[0, 0, 0], [0, 0, 0]])
    backend = Relaxer()
    runner = evaluator(protocol, backend)
    result = runner.evaluate(pack([invalid, metal("Cu")]), step=0)
    assert backend.calls == ["Li", "Cu"]
    assert not runner.raw_validity[0]["raw_struct_valid"]
    assert result["sample_ema/stability/constructed_count"] == 2
    assert result["sample_ema/stability/relaxed_ok_count"] == 2


def test_construction_failures_follow_core_guards(protocol):
    from src.eval.dng_eval import compute_evaluator_metrics

    tiny = Structure(Lattice.cubic(.3), ["Li"], [[0, 0, 0]])
    samples = pack([tiny, metal("Li"), metal("Cu")])
    samples["F1"][1, 0, 0] = float("nan")
    items = _sample_items(samples)
    expected = compute_evaluator_metrics(items, limit=3, ref_structs=[], sample_seed=0, include_wasserstein=False)
    backend = Relaxer()
    records = relax_all(items, backend, 500)
    assert [r["constructed"] for r in records] == [c.constructed for c in expected.pred_crys_list] == [False, False, True]
    assert backend.calls == ["Cu"]
    assert [r["relax_status"] for r in records] == ["not_constructed", "not_constructed", "ok"]


def test_all_relaxed_keeps_core_thermodynamic_aggregate_names(protocol):
    from orbitol.eval_adapter import _ConfiguredLogger
    from src.data.mp20_tokens import tokens_to_structure
    from src.eval.stability import _compute_thermo_metrics

    runner = evaluator(protocol)
    actual = runner.evaluate(protocol.rows, step=0)
    structures = [tokens_to_structure(item) for item in _sample_items(protocol.rows)[1:]]
    expected = _compute_thermo_metrics(_ConfiguredLogger(protocol.settings, Relaxer()), structures,
                                       tag="sample_ema/all_relaxed", step=0, enabled=True)
    shared = {key: value for key, value in actual.items() if key.startswith("sample_ema/all_relaxed/")}
    assert shared.keys() == expected.keys()
    for key, value in expected.items():
        # Summation over a single saved pool vs core batches can differ by roundoff.
        if key.endswith("e_above_hull_mean"):
            assert shared[key] == pytest.approx(value, rel=0, abs=1e-15)
        else:
            assert shared[key] == value


@pytest.mark.parametrize("width,compile_mode,budget", [
    (256, "reduce-overhead", 8192), (512, "reduce-overhead", 8192),
    (513, "max-autotune-no-cudagraphs", 6144), (1024, "max-autotune-no-cudagraphs", 6144),
])
def test_auto_defaults_are_resolved_and_serialized(width, compile_mode, budget):
    config = C.TrainConfig(d_model=width)
    assert config.compile_trunk == compile_mode and config.sample_token_budget == budget
    assert asdict(config)["compile_trunk"] == compile_mode
    assert asdict(config)["sample_token_budget"] == budget
    assert (config.pad_multiple, config.sample_count, config.sun_k, config.nequip_eval,
            config.eval_mode, config.nequip_backend, config.thermo_relax_steps) == (4, 10000, 0, True, "async", "oeq", 500)
    defaults = {field.name: field.default for field in fields(C.TrainConfig)}
    assert defaults["compile_trunk"] == "auto" and defaults["sample_token_budget"] == -1
    assert C.TrainConfig(sample_token_budget=0, compile_trunk="none").sample_token_budget == 0


@pytest.mark.parametrize("kwargs", [dict(sample_token_budget=-2), dict(sample_token_budget=True),
    dict(compile_trunk="invalid"), dict(eval_mode="invalid"), dict(nequip_backend="invalid"),
    dict(thermo_relax_steps=0), dict(nequip_relax_batch_size=0), dict(sun_k=-1),
    dict(nequip_relax_mode="sequential")])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        C.TrainConfig(**kwargs)


def test_training_cli_resolves_auto_after_width_override(monkeypatch):
    configs = []
    monkeypatch.setattr(train, "resolve_resume", lambda *a: None)
    monkeypatch.setattr(train, "Trainer", lambda cfg, **kw: configs.append(cfg) or
                        SimpleNamespace(run=lambda: None, close=lambda: None))
    train.main(["--run-name", "width-auto", "--d-model", "1024"])
    assert configs[0].compile_trunk == "max-autotune-no-cudagraphs"
    assert configs[0].sample_token_budget == 6144


def test_separate_relax_and_score_match_sync_after_record_roundtrip(protocol, monkeypatch):
    runner = evaluator(protocol)
    direct = runner.evaluate(protocol.rows, step=4, requested_G=torch.ones(8))
    items = _sample_items(protocol.rows)
    records = relax_all(items, Relaxer(), 500)
    assert [r["index"] for r in records] == list(range(8))
    assert {r["nsteps"] for r in records if r["relax_status"] == "ok"} == {500}
    assert set(records[0]) == {"index", "constructed", "relax_status", "reason",
                               "final_structure", "energy_eV", "nsteps", "initial_energy_eV",
                               "initial_fmax", "final_fmax", "final_max_force"}
    serialized = json.dumps([{**r, "final_structure": r["final_structure"].as_dict()
                              if r["final_structure"] is not None else None} for r in records])
    restored = json.loads(serialized)
    for record in restored:
        if record["final_structure"] is not None:
            record["final_structure"] = Structure.from_dict(record["final_structure"])
    monkeypatch.setattr(ThermoSettings, "make_relaxer", lambda *_: pytest.fail("scoring loaded a relaxer"))
    monkeypatch.setattr(Path, "write_text", lambda *_a, **_kw: pytest.fail("scoring wrote an artifact"))
    metrics, per_sample = score(items, restored, references=runner.references, ppd=runner.ppd,
                                settings=runner.settings, tag=runner.tag, step=4, requested_G=torch.ones(8))
    assert metrics == direct
    assert {key: value for key, value in metrics.items() if key.startswith("diag/")} == variant_metrics(
        runner.relaxation_diagnostics(), checked_count=len(items))
    assert per_sample == runner.per_sample
    assert json.dumps([{**r, "final_structure": r["final_structure"].as_dict()
                        if r["final_structure"] is not None else None} for r in restored]) == serialized


def test_fast_backend_is_lazy_and_uses_batch_size(protocol, monkeypatch):
    calls = []
    monkeypatch.setattr(nequip_relax, "DEFAULT_OEQ_PATH", str(protocol.model), raising=False)
    monkeypatch.setattr(nequip_relax, "make_fast_batch_relaxer",
                        lambda *a, **kw: calls.append((a, kw)) or Relaxer(), raising=False)
    runner = evaluator(protocol)
    runner.relaxer = None
    runner.thermo = replace(protocol.settings, nequip_oeq_path="")
    assert calls == []
    runner.evaluate(protocol.rows, step=0)
    assert calls == [((str(protocol.model), "cpu"), dict(max_force_abort=1e6, batch_size=3))]


@pytest.mark.parametrize("telemetry", [False, True])
def test_posthoc_and_sync_share_config_seed_pool_and_metrics(protocol, monkeypatch, telemetry):
    if telemetry:
        original = Relaxer.relax_many

        def observed(self, structures, **kwargs):
            results = original(self, structures, **kwargs)
            for result in results:
                result.update(initial_energy=result["trajectory"].energies[-1] + .02,
                              initial_fmax=.2, final_fmax=.01, final_max_force=.005)
            return results

        monkeypatch.setattr(Relaxer, "relax_many", observed)
    monkeypatch.setattr(nequip_relax, "make_fast_batch_relaxer", lambda *a, **kw: Relaxer(), raising=False)
    config = C.TrainConfig(d_model=32, compile_trunk="none", sample_token_budget=0,
                          sample_count=8, eval_mode="sync", nequip_relax_batch_size=3,
                          extra=dict(data_root=str(protocol.data), symmetry_reference=str(protocol.reference),
                                     phase_diagram=str(protocol.phase), nequip_model=str(protocol.model), sample_seed=777))
    protocol.settings = ThermoSettings.from_train_config(config, device="cpu")
    result = dict(rows=protocol.rows, metadata={}, requested_G=torch.ones(8, dtype=torch.int64),
                  contains_a=torch.ones(8, dtype=torch.bool), failure_counts={"invalid_species": 1})
    seeds = []

    def sample(*args, **kwargs):
        seeds.append((kwargs["n"], kwargs["seed"]))
        return result

    monkeypatch.setattr(train, "sample_pool", sample)
    monkeypatch.setattr(posthoc_eval, "sample_orbitol", sample)
    sync = train.Trainer.__new__(train.Trainer)
    sync.cfg, sync.device, sync.step = config, torch.device("cpu"), 4
    sync.run_dir = protocol.root / "sync"
    (sync.run_dir / "samples").mkdir(parents=True)
    sync.checkpoint_dir = sync.run_dir / "checkpoints"
    sync.checkpoint_dir.mkdir()
    monkeypatch.setattr(sync, "snapshot", lambda: dict(step=sync.step))
    for name in ("model", "ema", "tables", "tb", "encoding", "allowed_mask", "bank", "table_sha256"):
        setattr(sync, name, None)
    sync.evaluator = evaluator(protocol)
    sync.evaluator.sample_seed = 777
    direct = sync.evaluate_samples()
    assert direct["orbit/best_msun"] == direct["sample_ema/MSUN"]
    assert json.loads((sync.checkpoint_dir / "best_msun.json").read_text()) == {
        "MSUN": direct["sample_ema/MSUN"], "step": 4}
    snapshot = protocol.root / "step_0000004.pt"
    torch.save(dict(cfg=asdict(config), step=4, ema={"weight": torch.zeros(1)}), snapshot)
    out = protocol.root / "posthoc"
    args = posthoc_eval.parse_args(["--kind", "orbitol", "--snapshot", str(snapshot), "--out", str(out), "--device", "cpu"])
    posthoc_eval.sample_stage(args)
    queued = posthoc_eval.evaluate_stage(out)
    if telemetry:
        sync_rows = [json.loads(line) for line in
                     (sync.run_dir / "samples/step_0000004.relaxed.jsonl").read_text().splitlines()]
        posthoc_rows = [json.loads(line) for line in (out / "relaxed_records.jsonl").read_text().splitlines()]
        from orbitol.eval_diagnostics import RELAXATION_FIELDS
        for row in sync_rows:
            if row["initial_energy_eV"] is not None:
                assert {k: row[k] for k in RELAXATION_FIELDS} == {
                    k: posthoc_rows[row["pool_index"]][k] for k in RELAXATION_FIELDS}
                assert row["final_max_force"] == .005
        assert any(row["initial_energy_eV"] is not None for row in sync_rows)
    assert seeds == [(8, 777), (8, 777)]
    expected = {key: float(value) if math.isfinite(value) else None for key, value in direct.items()
                if key not in {"orbit/nequip_seconds", "orbit/eval_seconds", "orbit/best_msun"}}
    assert {key: value for key, value in queued.items() if not key.startswith("posthoc/")} == expected
    settings = json.loads((out / "evaluation_settings.json").read_text())
    assert settings["protocol"] == "relaxed_endpoints_v1"
    assert settings["thermo"] == asdict(protocol.settings)
    assert settings["sample_seed"] == 777 and settings["sample_count"] == 8
    pool_bytes = (out / "pool.json").read_bytes()
    snapshot.unlink()
    posthoc_eval.sample_stage(args)
    assert (out / "pool.json").read_bytes() == pool_bytes
    assert len(seeds) == 2
    queue_args = posthoc_queue.parse_args(["--run", f"orbitol:{protocol.root / 'training'}", "--device", "cpu"])
    runtime = posthoc_queue.Runtime(queue_args)
    from_queue = runtime.evaluate(queue_args.run[0], out)
    assert {key: value for key, value in from_queue.items() if not key.startswith("posthoc/")} == expected


def test_explicit_legacy_subset_and_torchscript_are_selectable(protocol, monkeypatch):
    from src.eval.dng_eval import compute_novelty_metrics, compute_sun_metrics
    from orbitol.eval_adapter import _ConfiguredLogger

    class LegacyRelaxer(Relaxer):
        def relax_many(self, structures, *, steps, verbose):
            assert steps == 200
            return super().relax_many(structures, steps=500, verbose=verbose)

    calls = []
    monkeypatch.setattr(nequip_relax, "make_batch_relaxer",
                        lambda *a, **kw: calls.append((a, kw)) or LegacyRelaxer())
    protocol.settings = replace(protocol.settings, nequip_backend="torchscript", thermo_relax_steps=200,
                                nequip_ts_path=str(protocol.model))
    runner = evaluator(protocol, sun_k=2)
    runner.relaxer = None
    items = _sample_items(protocol.rows)
    novelty = compute_novelty_metrics(items, runner.novelty_ref_structs, limit=8)
    expected = compute_sun_metrics(novelty.novelty_metrics,
                                  thermo_logger=_ConfiguredLogger(protocol.settings, LegacyRelaxer()),
                                  tag="sample_ema", step=4, enabled=True, base_seed=127,
                                  sun_target=2, show_progress=False)
    actual = runner.evaluate(protocol.rows, step=4)
    assert calls == [((str(protocol.model), "cpu"), dict(max_force_abort=1e6))]
    for key, value in {**expected.thermo_metrics, **expected.summary_metrics}.items():
        assert actual[key] == value
    assert actual["joint/checked_count"] == 2
    assert not any(key.startswith("sample_ema/stability/") for key in actual)


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "status", "constructed", "endpoint"])
def test_scoring_rejects_incomplete_or_inconsistent_records(protocol, corruption):
    runner = evaluator(protocol)
    items = _sample_items(protocol.rows)
    records = relax_all(items, Relaxer(), 500)
    message = "one record per attempt"
    if corruption == "missing":
        records.pop()
    elif corruption == "duplicate":
        records[0]["index"] = 1
    elif corruption == "status":
        records[0]["relax_status"] = "unknown"
        message = "Invalid relax_status"
    elif corruption == "constructed":
        records[0]["constructed"] = True
        message = "constructed and relax_status"
    else:
        records[1]["final_structure"] = None
        message = "requires final_structure"
    with pytest.raises(ValueError, match=message):
        score(items, records, references=runner.references, ppd=runner.ppd,
              settings=runner.settings, tag=runner.tag, step=0)


def test_historical_pool_keeps_its_backend_and_controls(protocol):
    args = posthoc_eval.parse_args([
        "--kind", "orbitol", "--snapshot", str(protocol.root / "step_0000004.pt"),
        "--out", str(protocol.root / "out"), "--data-root", str(protocol.data),
        "--symmetry-reference", str(protocol.reference), "--ppd-path", str(protocol.phase),
        "--nequip-backend", "torchscript", "--nequip-ts-path", str(protocol.model),
        "--sample-count", "8", "--sun-k", "2", "--relax-steps", "200", "--seed", "777", "--device", "cpu"])
    settings = posthoc_eval.evaluation_settings(args)
    del settings["protocol"], settings["thermo"]["nequip_backend"], settings["thermo"]["nequip_oeq_path"]
    settings["sampler"] = dict(cfg=dict(nequip_eval=False))  # historical train-time switch
    loaded = posthoc_eval.build_evaluator(settings)
    assert loaded.thermo.nequip_backend == "torchscript" and loaded.thermo.thermo_relax_steps == 200
    queue_args = posthoc_queue.parse_args(["--run", f"orbitol:{protocol.root / 'training'}", "--device", "cpu"])
    runtime = posthoc_queue.Runtime(queue_args)
    runtime.prepare(queue_args.run[0], settings)
    assert runtime.settings["sample_count"] == 8 and runtime.settings["sample_seed"] == 777
    assert runtime.settings["sun_k"] == 2
    assert runtime.settings["thermo"]["nequip_backend"] == "torchscript"
    assert runtime.settings["mlip_model_sha256"] == settings["mlip_model_sha256"]


@pytest.mark.parametrize("historical", [False, True])
def test_queue_uses_saved_project_and_stable_posthoc_name(protocol, monkeypatch, historical):
    monkeypatch.setenv("ORBITOL_WANDB_ENTITY", "fixture-entity")
    monkeypatch.setenv("WANDB_ENTITY", "generic-entity")
    import sys
    calls = []
    pool = dict(kind="orbitol", n=8, seed=777, step=4, samples_sha256="samples",
                sampler=dict(cfg=dict(wandb_name="custom-training", wandb_project="custom-project")))
    monkeypatch.setattr(posthoc_queue, "verify_pool", lambda _: pool)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(step=0, finish=lambda: None),
        define_metric=lambda *a, **kw: None, log=lambda *a, **kw: None))
    (protocol.root / "metrics.json").write_text("{}")
    settings = dict(thermo=asdict(protocol.settings))
    if not historical:
        settings["protocol"] = "relaxed_endpoints_v1"
    (protocol.root / "evaluation_settings.json").write_text(json.dumps(settings))
    args = posthoc_queue.parse_args(["--run", f"orbitol:{protocol.root / 'training'}"])
    posthoc_queue.log_wandb(args, args.run[0], protocol.root, {})
    assert calls[0]["name"] == "training-posthoc"
    assert calls[0]["project"] == "custom-project" and calls[0]["entity"] == "fixture-entity"
    assert calls[0]["config"]["sample_count"] == 8 and calls[0]["config"]["seed"] == 777
