"""Post-hoc CLI serialization with mocked sampling and MLIP only."""
import hashlib
from dataclasses import replace
import json
import math
import multiprocessing
import os
from pathlib import Path
import pickle
import signal
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Lattice, Structure
from pymatgen.entries.computed_entries import ComputedEntry
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import posthoc_eval as posthoc
from orbitol import contracts as C
from orbitol.symmetry_panel import SymmetryPanel
from orbitol.eval_diagnostics import minimum_distance, relaxation_records, variant_metrics
from src.data.mp20_tokens import decode_Y1, lattice_to_Y


@pytest.fixture
def context():
    parent = Path(__file__).resolve().parents[1] / ".pytest_cache"
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        root = Path(directory)
        data = root / "data"
        (data / "processed").mkdir(parents=True)
        (data / "raw").mkdir()

        def item(symbol):
            s = Structure(Lattice.cubic(4), [symbol], [[.1, .2, .3]])
            return dict(A0=torch.tensor(s.atomic_numbers), F1=torch.tensor(s.frac_coords),
                        Y1=torch.from_numpy(lattice_to_Y(s.lattice.abc, s.lattice.angles)),
                        pad_mask=torch.tensor([False]))

        for split in ("train", "val"):
            torch.save([item("Cu")], data / "processed" / f"mp20_tokens_{split}.pt")
            (data / "raw" / f"{split}.csv").write_text("cif\n")
        reference = root / "reference.json"
        SymmetryPanel(data / "processed/mp20_tokens_train.pt", [item("Cu")], reference).reference
        phase = root / "hull.pkl"
        with phase.open("wb") as stream:
            pickle.dump(PhaseDiagram([ComputedEntry(el, 0.) for el in ("Cu", "Li", "Na")]), stream)
        model = root / "model.nequip.pth"
        model.write_bytes(b"mock model")
        model_args = dict(sample_num_steps=150, S_churn=60., S_noise=1.003,
                          aa_rho_types=0., aa_rho_coords=0., aa_rho_lattice=0.,
                          sample_chunk_size=2048, bf16=True, lattice_repr="ltri")
        checkpoints = root / "baseline/checkpoints"
        checkpoints.mkdir(parents=True)
        snapshot = checkpoints / "step_0020000.pt"
        torch.save(dict(step=20000, ema_state_dict={"weight": torch.zeros(1)}, model_args=model_args), snapshot)
        yield SimpleNamespace(root=root, data=data, phase=phase, model=model, snapshot=snapshot, reference=reference,
                              model_args=model_args, items=[item("Li"), item("Na")])


def cli(context, out="eval"):
    return ["--kind", "crystalite", "--snapshot", str(context.snapshot),
            "--out", str(context.root / out), "--data-root", str(context.data),
            "--ppd-path", str(context.phase), "--nequip-ts-path", str(context.model),
            "--symmetry-reference", str(context.reference),
            "--device", "cpu", "--sample-count", "2", "--seed", "777",
            "--nequip-backend", "torchscript", "--relax-steps", "200", "--sun-k", "512"]


def test_parser_defaults_and_validation():
    args = posthoc.parse_args(["--kind", "crystalite", "--snapshot", "/snapshot", "--out", "/out"])
    assert (args.sample_count, args.sun_k, args.seed, args.relax_mode) == (10000, 0, 123, "batch")
    assert (args.nequip_backend, args.relax_steps) == ("oeq", 500)
    with pytest.raises(SystemExit):
        posthoc.parse_args(["--kind", "wrong", "--snapshot", "/snapshot", "--out", "/out"])
    with pytest.raises(SystemExit):
        posthoc.parse_args(["--kind", "crystalite", "--snapshot", "/snapshot", "--out", "/out", "--sample-count", "0"])


@pytest.mark.parametrize("kind", ["orbitol", "crystalite"])
def test_sample_stage_resolves_dataset_from_snapshot(context, monkeypatch, kind):
    spec = replace(C.dataset_spec("alex_mp20"), data_root=str(context.data),
                   symmetry_reference=str(context.reference), phase_diagram=str(context.phase))
    monkeypatch.setitem(C.DATASETS, "alex_mp20", spec)
    cfg = {"dataset": "alex_mp20", "extra": {}}
    model_args = {**context.model_args, "dataset_name": "alex_mp20"}
    payload = (dict(cfg=cfg, ema={"weight": torch.zeros(1)}) if kind == "orbitol" else
               dict(model_args=model_args, ema_state_dict={"weight": torch.zeros(1)}))
    torch.save(dict(step=20000, **payload), context.snapshot)
    calls = []

    def crystalite(snapshot, out, settings):
        calls.append(settings)
        return context.items

    monkeypatch.setattr(posthoc, "sample_crystalite", crystalite)
    monkeypatch.setattr(posthoc, "sample_orbitol", lambda *args, **kwargs: dict(
        rows=posthoc._batch_samples(context.items)))
    args = posthoc.parse_args(["--kind", kind, "--snapshot", str(context.snapshot),
                               "--out", str(context.root / "dataset-pool"), "--sample-count", "2",
                               "--nequip-ts-path", str(context.model), "--device", "cpu",
                               "--nequip-backend", "torchscript", "--relax-steps", "200", "--sun-k", "512"])
    posthoc.sample_stage(args)
    settings = json.loads((args.out / "evaluation_settings.json").read_text())
    assert settings["data_root"] == str(context.data)
    assert settings["symmetry_reference"] == str(context.reference)
    assert settings["thermo"]["thermo_ppd_mp"] == str(context.phase)
    if kind == "crystalite":
        assert calls[0]["dataset_name"] == "alex_mp20"
        assert calls[0]["data_root"] == str(context.data)
    assert args.data_root is None and args.symmetry_reference is None


def test_orbitol_sampler_overrides_reach_sampler_and_pool_provenance(context, monkeypatch):
    spec = replace(C.dataset_spec("alex_mp20"), data_root=str(context.data),
                   symmetry_reference=str(context.reference), phase_diagram=str(context.phase))
    monkeypatch.setitem(C.DATASETS, "alex_mp20", spec)
    torch.save(dict(step=20000, cfg={"dataset": "alex_mp20", "extra": {}}, ema={"weight": torch.zeros(1)}),
               context.snapshot)
    calls = []
    monkeypatch.setattr(posthoc, "sample_orbitol", lambda *args, **kwargs: calls.append(kwargs) or dict(
        rows=posthoc._batch_samples(context.items)))
    overrides = dict(sample_num_steps=25, S_churn=10.0, sigma_min=0.01, sigma_max=3.0)

    def args(extra):
        return posthoc.parse_args(["--kind", "orbitol", "--snapshot", str(context.snapshot),
                                   "--out", str(context.root / "override-pool"), "--sample-count", "2",
                                   "--nequip-ts-path", str(context.model), "--device", "cpu",
                                   "--nequip-backend", "torchscript", "--relax-steps", "200"] + extra)
    pool = posthoc.sample_stage(args(["--sampler-overrides", json.dumps(overrides)]))
    assert calls[0]["overrides"] == overrides
    assert pool["sampler"]["overrides"] == overrides
    assert {key: pool["sampler"]["cfg"][key] for key in overrides} == overrides
    with pytest.raises(ValueError, match="different sampling provenance"):
        posthoc.sample_stage(args([]))
    for bad in (["--sampler-overrides", '{"lr": 0.1}'],):
        with pytest.raises(SystemExit):
            args(bad)
    with pytest.raises(SystemExit):
        posthoc.parse_args(["--kind", "crystalite", "--snapshot", "/s", "--out", "/o",
                            "--sampler-overrides", '{"S_churn": 0}'])


@pytest.fixture(autouse=True)
def disabled_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")


@pytest.fixture(autouse=True)
def historical_normalizer(monkeypatch):
    # Fallback for a contracts module without normalize_historical_config.
    if not hasattr(C, "normalize_historical_config"):
        def normalize(cfg):
            return dict(dict(compile_trunk="none", pad_multiple=0, sample_token_budget=0,
                             sample_count=2048, sun_k=512, nequip_eval=False, eval_mode="sync",
                             nequip_backend="torchscript", thermo_relax_steps=200,
                             nequip_relax_mode="batch", nequip_relax_batch_size=128), **cfg)
        monkeypatch.setattr(C, "normalize_historical_config", normalize, raising=False)


@pytest.mark.parametrize("mode", ["batch", "sequential"])
def test_outputs_and_shared_evaluator(context, monkeypatch, mode):
    import orbitol.nequip_relax as relax_module

    sample_calls, relax_calls = [], []

    def sampler(snapshot, out, settings):
        sample_calls.append((snapshot, out, settings))
        return context.items

    class Relaxer:
        def relax(self, s, **kw):
            relax_calls.append(s[0].specie.symbol)
            if s[0].specie.symbol == "Na":
                raise ValueError("bad structure")
            return dict(final_structure=s.copy(), trajectory=SimpleNamespace(energies=[-1.]))

        def relax_many(self, structs, **kw):
            return [self.relax(s, **kw) for s in structs]

    monkeypatch.setattr(posthoc, "sample_crystalite", sampler)
    monkeypatch.setattr(relax_module, f"make_{mode}_relaxer", lambda *args, **kw: Relaxer())
    metrics = posthoc.main(cli(context) + ["--relax-mode", mode])
    out = context.root / "eval"
    assert json.loads((out / "metrics.json").read_text()) == metrics
    assert metrics["joint/checked_count"] == 2
    assert metrics["joint/relaxed_count"] == 1
    assert metrics["sample_ema/sun/thermo_failed_total"] == 1
    assert metrics["posthoc/sample_seconds"] >= 0
    assert metrics["posthoc/evaluate_seconds"] >= 0
    assert {"Li", "Na"} == set(relax_calls)
    rows = torch.load(out / "samples.pt", weights_only=False)
    assert rows["A0"].shape == (2, 1)
    settings = json.loads((out / "evaluation_settings.json").read_text())
    assert settings["step"] == 20000
    assert settings["no_thermo_before_steps"] == 0
    assert settings["thermo"]["nequip_model_kind"] == "torchscript"
    assert settings["mlip_model_path"] == str(context.model)
    assert settings["mlip_model_sha256"] == hashlib.sha256(b"mock model").hexdigest()
    assert settings["phase_diagram_sha256"] == hashlib.sha256(
        Path(settings["thermo"]["thermo_ppd_mp"]).read_bytes()).hexdigest()
    assert settings["seeds"]["sun_selection"] == 777 + 20000 + sum(map(ord, "sample_ema"))
    assert settings["sampler"]["sample_chunk_size"] == 2048
    assert settings["sampler"]["sample_mode"] == "ema"
    assert settings["sampler"]["bf16"] is True
    assert sample_calls[0][2]["atom_count_strategy"] == "empirical"
    assert set(settings["thermo"]) == set(posthoc.asdict(posthoc.ThermoSettings()))
    assert settings["symmetry_reference"] == str(context.reference)
    assert not (out / "symmetry_reference.json").exists()
    pool = json.loads((out / "pool.json").read_text())
    assert pool["status"] == "complete" and pool["samples_sha256"] == posthoc._sha256(out / "samples.pt")
    assert pool["inode"] == context.snapshot.stat().st_ino


def test_subprocess_uses_core_sampler_ema_and_snapshot_settings(context, monkeypatch):
    args = posthoc.parse_args(cli(context))
    settings = posthoc._sampler_settings(context.model_args, args)
    out = context.root / "child"
    out.mkdir()
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        torch.save(context.items, out / "samples.pt")

    monkeypatch.setattr(posthoc.subprocess, "run", run)
    items = posthoc.sample_crystalite(context.snapshot, out, settings)
    assert len(items) == 2
    command, kwargs = commands[0]
    assert "import orbitol.core" in command[2]
    assert "src.sample_crystalite_ckpt" in command[2]
    assert command[command.index("--sample_mode") + 1] == "ema"
    assert command[command.index("--sample_num_steps") + 1] == "150"
    assert command[command.index("--sample_chunk_size") + 1] == "2048"
    assert "--bf16" in command and "--no-save_extxyz" in command
    assert kwargs["check"] is True and kwargs["cwd"] == out
    assert kwargs["env"]["PYTHONPATH"] == str(Path(posthoc.__file__).resolve().parents[1])


def test_missing_input_never_starts_sampler(context, monkeypatch):
    args = posthoc.parse_args(cli(context))
    settings = posthoc._sampler_settings(context.model_args, args)
    (context.data / "raw/train.csv").unlink()
    monkeypatch.setattr(posthoc.subprocess, "run", lambda *args, **kw: pytest.fail("must not invoke sampler"))
    with pytest.raises(FileNotFoundError, match="read-only"):
        posthoc.sample_crystalite(context.snapshot, context.root, settings)


def test_rejects_missing_ema_or_mismatched_paper_settings(context):
    torch.save(dict(step=20000, model_args=context.model_args), context.snapshot)
    with pytest.raises(ValueError, match="no EMA"):
        posthoc._snapshot_info(context.snapshot)
    args = posthoc.parse_args(cli(context))
    with pytest.raises(ValueError, match="S_churn"):
        posthoc._sampler_settings({**context.model_args, "S_churn": 20}, args)


def test_wandb_resumes_same_name_across_output_dirs(context, monkeypatch):
    monkeypatch.setenv("ORBITOL_WANDB_ENTITY", "fixture-entity")
    monkeypatch.setenv("WANDB_ENTITY", "generic-entity")
    initializations, definitions, logs, finishes = [], [], [], []

    def init(**kwargs):
        initializations.append(kwargs)
        return SimpleNamespace(finish=lambda: finishes.append(True))

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(
        init=init, define_metric=lambda *args, **kw: definitions.append((args, kw)),
        log=lambda values: logs.append(values),
    ))
    pool = {"kind": "orbitol", "sampler": {"cfg": {"dataset": "mp20"}}}
    for step in (20000, 40000):
        args = posthoc.parse_args(cli(context, out=str(step)) + ["--wandb-name", "baseline-eval"])
        posthoc._log_wandb(args, {"sample_ema/MSUN": .2}, step, {"thermo": {}}, pool)
    assert initializations[0]["id"] == initializations[1]["id"]
    assert initializations[0]["project"] == "orbitol"
    assert initializations[0]["entity"] == "fixture-entity"
    assert all(kwargs["resume"] == "allow" for kwargs in initializations)
    assert (("*",), {"step_metric": "train_step"}) in definitions
    assert logs == [{"eval/MSUN": .2, "train_step": step} for step in (20000, 40000)]
    assert finishes == [True, True]


def test_wandb_project_follows_the_pool_dataset():
    default, explicit = SimpleNamespace(wandb_project=None), SimpleNamespace(wandb_project="other")
    alex = {"kind": "orbitol", "sampler": {"cfg": {"dataset": "alex_mp20"}}}
    assert posthoc.wandb_project(default, alex) == "orbitol_alex_mp20"
    assert posthoc.wandb_project(default, {"kind": "orbitol", "sampler": {"cfg": {}}}) == "orbitol"
    assert posthoc.wandb_project(default, {"kind": "crystalite", "sampler": {}}) == "orbitol"
    assert posthoc.wandb_project(explicit, alex) == "other"


def diagnostic_row(index=0, **kwargs):
    return dict(dict(pool_index=index, is_un=True, raw_struct_valid=True,
                     raw_comp_valid=True, raw_valid=True, relaxation_outcome="success",
                     failure_reason=None, e_above_hull=-0.1, energy_per_atom=-1.,
                     initial_volume_per_atom=10., relaxed_volume_per_atom=10.,
                     relaxed_min_distance=2., relaxation_steps=200, final_max_force=None,
                     stable=True, metastable=True), **kwargs)


@pytest.mark.parametrize("criterion,key,value,boundary", [
    ("e_above_hull", "e_above_hull", -0.51, -0.5),
    ("min_distance", "relaxed_min_distance", .69, .7),
    ("volume_per_atom", "relaxed_volume_per_atom", 2.99, 3.),
])
def test_diagnostic_suspect_criteria(criterion, key, value, boundary):
    for measurement, suspect in ((value, 1), (boundary, 0), (None, 0)):
        rows = [diagnostic_row(**{key: measurement})]
        metrics = variant_metrics(rows, checked_count=2)
        assert metrics["diag/stable_suspect_count"] == suspect
        assert metrics["diag/metastable_suspect_count"] == suspect
        assert metrics[f"diag/suspect_{criterion}_count"] == suspect
        assert metrics["diag/SUN_plausible"] == (1 - suspect) / 2
        assert metrics["diag/MSUN_plausible"] == (1 - suspect) / 2


def test_diagnostic_numerators_denominators_and_robust_statistics():
    rows = [
        diagnostic_row(0),
        diagnostic_row(1, e_above_hull=-1., relaxed_min_distance=.1, relaxed_volume_per_atom=1.),
        diagnostic_row(2, raw_struct_valid=False, raw_valid=False),
        diagnostic_row(3, stable=False, e_above_hull=.05),
        diagnostic_row(4, is_un=False),
        diagnostic_row(5, stable=False, metastable=False, e_above_hull=None,
                       relaxation_outcome="relaxation_failed", failure_reason="relax_divergence"),
        # An unstable endpoint satisfying a geometry criterion is not a
        # counted-stable/metastable suspect; it still contributes hull stats.
        diagnostic_row(6, stable=False, metastable=False, relaxed_min_distance=.1, e_above_hull=1.),
    ]
    metrics = variant_metrics(rows, checked_count=7)
    assert metrics["diag/stable_suspect_count"] == 1
    assert metrics["diag/metastable_suspect_count"] == 1
    for name in ("e_above_hull", "min_distance", "volume_per_atom"):
        assert metrics[f"diag/suspect_{name}_count"] == 1
    assert metrics["diag/SUN_plausible"] == 2 / 7
    assert metrics["diag/MSUN_plausible"] == 3 / 7
    assert metrics["diag/SUN_valid"] == 2 / 7
    assert metrics["diag/MSUN_valid"] == 3 / 7
    assert metrics["diag/valid_checked_count"] == 6  # includes failed and non-UN attempts
    assert metrics["diag/e_above_hull_median"] == -.1
    assert metrics["diag/e_above_hull_p05"] == pytest.approx(-.775)
    empty = variant_metrics([], checked_count=0)
    assert math.isnan(empty["diag/SUN_plausible"])
    assert math.isnan(empty["diag/e_above_hull_median"])
    meta_only = variant_metrics([
        diagnostic_row(stable=False, e_above_hull=.05, relaxed_min_distance=.6),
        diagnostic_row(1, is_un=False, e_above_hull=-2.),
    ], checked_count=2)
    assert meta_only["diag/stable_suspect_count"] == 1
    assert meta_only["diag/metastable_suspect_count"] == 2
    assert meta_only["diag/MSUN_plausible"] == 0.
    assert meta_only["diag/MSUN_valid"] == .5


def test_pbc_minimum_distance_includes_images_and_coincident_sites():
    single = Structure(Lattice.cubic(.6), ["Li"], [[0, 0, 0]])
    assert minimum_distance(single) == pytest.approx(.6)
    across_boundary = Structure(Lattice.cubic(4), ["Li", "Li"], [[.01, 0, 0], [.99, 0, 0]])
    assert minimum_distance(across_boundary) == pytest.approx(.08)
    coincident = Structure(Lattice.cubic(4), ["Li", "Na"], [[0, 0, 0], [0, 0, 0]])
    assert minimum_distance(coincident) == 0.


def test_minimum_distance_on_exploded_and_skewed_cells():
    # An exploded relaxed cell (one 1000 A vector) made pymatgen's neighbor list corrupt the heap.
    exploded = Structure(Lattice([[1000, 0, 0], [0, 1.5, 0], [0, 0, 2]]), ["Li", "Na"], [[0, 0, 0], [.001, .5, 0]])
    assert minimum_distance(exploded) == pytest.approx(1.25)
    rng = np.random.default_rng(3)
    shifts = np.array(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij")).reshape(3, -1).T
    for _ in range(20):
        matrix = np.diag(rng.uniform(2, 6, 3)) + np.triu(rng.uniform(-3, 3, (3, 3)), 1)
        frac = rng.uniform(0, 1, (4, 3))
        structure = Structure(Lattice(matrix), ["Li"] * 4, frac)
        brute = min(np.linalg.norm((frac[j] - frac[i] + shift) @ matrix)
                    for i in range(4) for j in range(4) for shift in shifts if i != j or shift.any())
        assert minimum_distance(structure) == pytest.approx(brute)


def test_diagnostic_record_telemetry_and_missing_values():
    from orbitol.eval_adapter import RelaxedRecord
    final = Structure(Lattice.cubic(4), ["Li", "Li"], [[0, 0, 0], [.5, .5, .5]])
    record = RelaxedRecord(4, final, -.1, True, True, True)
    logger = SimpleNamespace(
        endpoints={4: (True, dict(final_structure=final, nsteps=200,
                                 trajectory=SimpleNamespace(energies=[-2.], forces=[[[3., 4., 0.], [0., 0., 0.]]])))},
        failures={9: ("relax_non_finite", "non-finite relaxation (wrapped ValueError: bad)")},
        initial_volume_per_atom={4: 30., 9: 10.},
    )
    validity = {4: dict(raw_struct_valid=False, raw_comp_valid=True, raw_valid=False)}
    rows = relaxation_records([record], {9: False, 4: True}, validity, logger)
    assert [row["pool_index"] for row in rows] == [4, 9]
    assert rows[0]["energy_per_atom"] == -1.
    assert rows[0]["relaxed_volume_per_atom"] == 32.
    assert rows[0]["relaxation_steps"] == 200
    assert rows[0]["final_max_force"] == 5.
    assert not rows[0]["raw_valid"] and rows[0]["raw_comp_valid"]
    assert rows[1]["relaxation_outcome"] == "relaxation_failed"
    assert rows[1]["failure_reason"] == "relax_non_finite"
    assert "ValueError" in rows[1]["failure_message"]
    assert rows[1]["final_max_force"] is rows[1]["relaxed_min_distance"] is None
    assert not rows[1]["stable"] and not rows[1]["metastable"]
    json.dumps(rows, allow_nan=False)


def test_diagnostics_atomic_overwrite_and_headline_bytes(context, monkeypatch):
    from orbitol import posthoc_io
    monkeypatch.setattr(posthoc, "sample_crystalite", lambda *args: context.items)
    posthoc.main(cli(context) + ["--stage", "sample"])
    out = context.root / "eval"
    headlines = {"joint/checked_count": 2., "joint/SUN": 1., "joint/MSUN": 1.,
                 "sample_ema/SUN": .5, "sample_ema/MSUN": .5,
                 "sample_ema/e_above_hull_mean": -55.}
    class Evaluator:
        def evaluate(self, *args, **kwargs):
            return dict(headlines)
    evaluator = Evaluator()
    baseline = posthoc.evaluate_stage(out, evaluator=evaluator)
    rows = [diagnostic_row(0, e_above_hull=-110.), diagnostic_row(1)]
    evaluator.relaxation_diagnostics = lambda: rows
    result = posthoc.evaluate_stage(out, evaluator=evaluator)
    encoded = lambda metrics: json.dumps({key: metrics[key] for key in headlines}).encode()
    assert encoded(baseline) == encoded(result) == encoded(headlines)
    assert result["diag/SUN_plausible"] == .5
    path = out / "relaxed_records.jsonl"
    original = path.read_bytes()
    assert [json.loads(line) for line in original.splitlines()] == rows
    replace = posthoc_io.os.replace
    replacements = []
    def interrupted(src, dest):
        if Path(dest) == path:
            assert path.read_bytes() == original
            # Whole new JSONL exists under a temporary name, old target intact.
            assert [json.loads(line) for line in Path(src).read_bytes().splitlines()] == rows
            replacements.append((src, dest))
            raise KeyboardInterrupt
        return replace(src, dest)
    rows[:] = [diagnostic_row(1)]
    monkeypatch.setattr(posthoc_io.os, "replace", interrupted)
    with pytest.raises(KeyboardInterrupt):
        posthoc.evaluate_stage(out, evaluator=evaluator)
    assert replacements and path.read_bytes() == original
    assert not list(out.glob(".relaxed_records.jsonl.*.tmp"))
    monkeypatch.setattr(posthoc_io.os, "replace", replace)
    posthoc.evaluate_stage(out, evaluator=evaluator)
    assert [json.loads(line) for line in path.read_bytes().splitlines()] == rows
    rewritten = path.read_bytes()
    posthoc.evaluate_stage(out, evaluator=evaluator)
    assert path.read_bytes() == rewritten  # replace, never append


def test_evaluate_after_snapshot_deleted_and_reject_corrupt_samples(context, monkeypatch):
    from orbitol.posthoc_queue import Queue, parse_args
    monkeypatch.setattr(posthoc, "sample_crystalite", lambda *args: context.items)
    out = context.root / "pools/baseline/step_0020000"
    posthoc.main(cli(context, out="pools/baseline/step_0020000") + ["--stage", "sample"])
    (context.snapshot.parent / "step_0040000.pt").write_bytes(b"newest fixture snapshot")
    args = parse_args(["--run", f"crystalite:{context.root / 'baseline'}",
                       "--out-root", str(context.root / "pools")])
    Queue(args)._prune(args.run[0])
    assert not context.snapshot.exists()  # only the queue's verified prune deletes it
    calls = []
    class Evaluator:
        def evaluate(self, rows, **kwargs):
            calls.append(kwargs)
            return {"sample_ema/SUN": .5}
    metrics = posthoc.evaluate_stage(out, evaluator=Evaluator())
    assert metrics["sample_ema/SUN"] == .5
    assert calls == [dict(step=20000, requested_G=None)]
    # The evaluate-only CLI accepts no snapshot and honors the reference flag.
    with pytest.raises(FileNotFoundError, match="Prebuilt"):
        posthoc.main(["--kind", "crystalite", "--stage", "evaluate", "--out", str(out),
                      "--symmetry-reference", str(context.root / "absent-reference.json")])
    with (out / "samples.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        posthoc.evaluate_stage(out, evaluator=Evaluator())
    assert len(calls) == 1


def test_missing_or_stale_reference_never_rebuilt(context, monkeypatch):
    missing = context.root / "missing.json"
    with pytest.raises(FileNotFoundError, match="Prebuilt"):
        posthoc.main(cli(context) + ["--symmetry-reference", str(missing)])
    assert not missing.exists()
    args = posthoc.parse_args(cli(context))
    settings = posthoc.evaluation_settings(args)
    saved = json.loads(context.reference.read_text())
    saved["metadata"]["spglib"] = "stale"
    context.reference.write_text(json.dumps(saved))
    before = context.reference.read_bytes()
    with pytest.raises(ValueError, match="Stale symmetry"):
        posthoc.build_evaluator(settings)
    assert context.reference.read_bytes() == before


def test_step_mismatch_and_atomic_commit_failure(context, monkeypatch):
    checkpoint = torch.load(context.snapshot, weights_only=False)
    checkpoint["step"] = 40000
    torch.save(checkpoint, context.snapshot)
    with pytest.raises(ValueError, match="filename"):
        posthoc._snapshot_info(context.snapshot)
    checkpoint["step"] = 20000
    torch.save(checkpoint, context.snapshot)
    monkeypatch.setattr(posthoc, "sample_crystalite", lambda *args: context.items)
    original = posthoc._write_json
    def interrupt(path, payload):
        if path.name == "pool.json":
            raise KeyboardInterrupt
        original(path, payload)
    monkeypatch.setattr(posthoc, "_write_json", interrupt)
    with pytest.raises(KeyboardInterrupt):
        posthoc.main(cli(context) + ["--stage", "sample"])
    assert not (context.root / "eval/pool.json").exists()
    monkeypatch.setattr(posthoc, "_write_json", original)
    posthoc.main(cli(context) + ["--stage", "sample"])
    assert (context.root / "eval/pool.json").is_file()


# Reuse the small table/cache fixture; no training or optimizer steps are run.
from test_train import fixture_data as training_data  # noqa: E402,F401


def test_orbitol_sample_stage_tiny_cpu_snapshot(context, training_data, monkeypatch):
    from orbitol.contracts import TrainConfig
    from orbitol.train import DECODE_FAILURES, Trainer
    extra, root = training_data
    cfg = TrainConfig(d_model=32, n_layers=2, n_heads=4, batch_size=8, num_workers=0,
                      max_steps=4, warmup_steps=1, precision="fp32", objective="P",
                      compile_trunk="none", sample_token_budget=0, eval_mode="sync",
                      nequip_backend="torchscript", thermo_relax_steps=200, sun_k=512,
                      sample_count=0, sample_num_steps=5, val_every=0, extra=dict(extra))
    trainer = Trainer(cfg, run_name="posthoc-tiny", out_root=root / "runs", device="cpu", no_wandb=True)
    snapshot = context.root / "step_0000000.pt"
    torch.save(trainer.snapshot(), snapshot)
    argv = cli(context) + ["--kind", "orbitol", "--snapshot", str(snapshot), "--sample-count", "8",
                           "--seed", "123", "--data-root", extra["data_root"],
                           "--symmetry-reference", extra["symmetry_reference"], "--stage", "sample"]
    posthoc.main(argv)
    out = context.root / "eval"
    result = torch.load(out / "samples.pt", weights_only=False)
    assert result["rows"]["A0"].shape == (8, 20)
    assert result["metadata"]["weights"] == "ema"
    assert result["metadata"]["seed"] == 123
    assert {"skeletons", "raw", "contains_a", "failures", "failure_counts"} <= result.keys()
    class Evaluator:
        def evaluate(self, rows, *, step, requested_G):
            assert step == 0 and torch.equal(requested_G, result["requested_G"])
            # Rows reach the evaluator as Niggli cells, inside the core's angle window.
            for i in range(len(rows["A0"])):
                if not bool(rows["pad_mask"][i].all()):
                    angles = decode_Y1(rows["Y1"][i])[1]
                    assert ((50 < angles) & (angles < 130)).all()
            return {"sample_ema/valid_rate": 0.}
    metrics = posthoc.evaluate_stage(out, evaluator=Evaluator())
    assert metrics["orbit/contains_a_rate"] == float(result["contains_a"].float().mean())
    for reason in set(DECODE_FAILURES) | result["failure_counts"].keys():
        assert metrics[f"orbit/failures/{reason}"] == result["failure_counts"].get(reason, 0)


def test_historical_sequential_config_is_normalized_before_validation(context, monkeypatch):
    original = dict(nequip_relax_mode="sequential", sample_count=2, sun_k=2)
    torch.save(dict(step=20000, cfg=original, ema={"weight": torch.zeros(1)}), context.snapshot)
    calls = []
    normalize = C.normalize_historical_config
    def normalized(cfg):
        calls.append(dict(cfg))
        return normalize(cfg)
    monkeypatch.setattr(C, "normalize_historical_config", normalized)
    monkeypatch.setattr(posthoc, "sample_orbitol", lambda *a, **k: dict(rows=posthoc._batch_samples(context.items)))
    posthoc.sample_stage(posthoc.parse_args(cli(context) + ["--kind", "orbitol", "--relax-mode", "sequential",
                                                           "--sun-k", "2"]))
    settings = json.loads((context.root / "eval/evaluation_settings.json").read_text())
    assert calls == [original]
    assert settings["sampler"]["cfg"]["nequip_backend"] == "torchscript"
    assert settings["sampler"]["cfg"]["eval_mode"] == "sync"
    assert settings["thermo"]["nequip_relax_mode"] == "sequential"
    assert settings["thermo"]["thermo_relax_steps"] == 200


WATCHDOG = 120


@pytest.fixture
def scoring_watchdog():
    def expired(signum, frame):
        raise TimeoutError("Scoring test watchdog expired")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.alarm(WATCHDOG)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture
def scoring_pool(context, monkeypatch, scoring_watchdog):
    monkeypatch.setattr(posthoc, "sample_crystalite", lambda *a: context.items)
    args = posthoc.parse_args(cli(context) + ["--sun-k", "0", "--relax-steps", "500"])
    posthoc.sample_stage(args)
    context.out = args.out
    context.saved = posthoc.read_json(args.out / "evaluation_settings.json")
    context.pool = posthoc.verify_pool(args.out)
    context.relaxed = [dict(index=i, constructed=True, relax_status="failed", reason="fixture",
                            final_structure=None, energy_eV=None, nsteps=None) for i in range(2)]
    return context


def _observe_scoring_worker(connection, out, settings, step, digest, observations, started):
    from orbitol import eval_protocol as protocol

    report = dict(prepare_count=0, complete_count=0, cuda_initialized=False)

    def observe(frame, event, arg):
        if event == "call" and frame.f_code in (protocol.score_raw.__code__, protocol.complete_score.__code__):
            report["cuda_initialized"] |= torch.cuda.is_initialized()
            if frame.f_code is protocol.score_raw.__code__:
                report["prepare_count"] += 1
                report["prepare_affinity"] = sorted(os.sched_getaffinity(0))
                started.set()
            else:
                report["complete_count"] += 1
                report["complete_affinity"] = sorted(os.sched_getaffinity(0))
                report["step"], report["tag"] = frame.f_locals["step"], frame.f_locals["tag"]
                report["sample_seed"] = frame.f_locals["settings"]["sample_seed"]
                report["has_ppd"] = frame.f_locals["ppd"] is not None

    sys.setprofile(observe)
    try:
        posthoc._score_pool_worker(connection, out, settings, step, digest)
    finally:
        sys.setprofile(None)
        report["cuda_initialized"] |= torch.cuda.is_initialized()
        observations.send(report)
        observations.close()


def observe_scoring_process(monkeypatch):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    started = context.Event()
    processes = []

    def process(*, target, args):
        assert target is posthoc._score_pool_worker and len(args) == 5
        assert isinstance(args[1], str) and isinstance(args[2], dict)
        assert isinstance(args[3], int) and isinstance(args[4], str)
        child = context.Process(target=_observe_scoring_worker, args=(*args, writer, started))
        processes.append(child)
        return child

    def get_context(method):
        assert method == "spawn"
        return SimpleNamespace(Pipe=context.Pipe, Process=process)

    monkeypatch.setattr(posthoc, "multiprocessing", SimpleNamespace(get_context=get_context))
    return SimpleNamespace(reader=reader, writer=writer, started=started, processes=processes)


@pytest.mark.parametrize("persistent", [False, True])
def test_full_pool_uses_shared_relaxation_and_scoring(scoring_pool, monkeypatch, persistent):
    import gzip
    import csv
    import orbitol.eval_protocol as protocol

    context = scoring_pool
    calls = []
    parent_pid = os.getpid()
    relaxer = object()
    observer = None

    def make_relaxer(self):
        if observer is not None:
            assert observer.started.wait(WATCHDOG)
        assert os.getpid() == parent_pid
        return relaxer

    monkeypatch.setattr(posthoc.ThermoSettings, "make_relaxer", make_relaxer)

    def relax_all(items, backend, steps):
        assert len(items) == 2 and backend is relaxer and steps == 500
        calls.append("relax")
        return context.relaxed

    complete = protocol.complete_score

    def counted_complete(prepared, records, **kwargs):
        assert records is context.relaxed and kwargs["step"] == 20000 and kwargs["tag"] == "sample_ema"
        assert kwargs["ppd"] is not None and kwargs["settings"]["sample_seed"] == 777
        calls.append("score")
        return complete(prepared, records, **kwargs)

    monkeypatch.setattr(protocol, "relax_all", relax_all)
    evaluator = posthoc.build_evaluator(context.saved) if persistent else None
    if persistent:
        monkeypatch.setattr(protocol, "complete_score", counted_complete)
    affinity = sorted(os.sched_getaffinity(0))
    monkeypatch.setattr(posthoc.os, "killpg", lambda *args: pytest.fail("Normal scoring must not signal a group"))
    for i in range(2):
        if not persistent:
            observer = observe_scoring_process(monkeypatch)
        try:
            metrics = posthoc.evaluate_stage(context.out, evaluator=evaluator)
            if observer is not None:
                assert observer.reader.poll(WATCHDOG)
                report = observer.reader.recv()
                assert report == dict(prepare_count=1, complete_count=1, cuda_initialized=False,
                                      prepare_affinity=affinity[1:] if len(affinity) > 1 else affinity,
                                      complete_affinity=affinity, step=20000, tag="sample_ema",
                                      sample_seed=777, has_ppd=True)
                assert observer.processes[0].exitcode == 0
                calls.append("score")
            assert calls == ["relax", "score"] * (i + 1)
        finally:
            if observer is not None:
                observer.reader.close()
                observer.writer.close()
    assert metrics["sample_ema/SUN"] == metrics["sample_ema/SUN_raw_UN"] == 0.
    with gzip.open(context.root / "eval/per_sample.csv.gz", "rt") as stream:
        assert [row["index"] for row in csv.DictReader(stream)] == ["0", "1"]
    assert context.saved["protocol"] == "relaxed_endpoints_v1"


def test_injected_evaluator_reloads_replaced_phase_diagram(scoring_pool, monkeypatch):
    from orbitol import eval_protocol as protocol

    context = scoring_pool
    evaluator = posthoc.build_evaluator(context.saved)
    cached = evaluator.ppd
    evaluator.relaxer = object()
    relaxed = [dict(index=i, constructed=True, relax_status="ok", reason=None,
                    final_structure=Structure(Lattice.cubic(4), [element], [[.1, .2, .3]]),
                    energy_eV=-.01, nsteps=500) for i, element in enumerate(("Li", "Na"))]
    monkeypatch.setattr(protocol, "relax_all", lambda *args: relaxed)
    with context.phase.open("wb") as stream:
        pickle.dump(PhaseDiagram([ComputedEntry(el, -1.) for el in ("Li", "Na", "Cu")]), stream)
    context.saved["phase_diagram_sha256"] = posthoc._sha256(context.phase)
    posthoc._write_json(context.out / "evaluation_settings.json", context.saved)
    metrics = posthoc.evaluate_stage(context.out, evaluator=evaluator)
    assert metrics["sample_ema/SUN"] == 0.
    assert evaluator.ppd is cached


def test_spawn_scoring_ignores_parent_tqdm_lock(scoring_pool, monkeypatch):
    from tqdm import tqdm
    from orbitol import eval_protocol as protocol

    acquired, release = threading.Event(), threading.Event()

    def background():
        with tqdm.get_lock():
            acquired.set()
            assert release.wait(WATCHDOG)

    thread = threading.Thread(target=background)
    monkeypatch.setattr(posthoc.ThermoSettings, "make_relaxer", lambda self: object())
    monkeypatch.setattr(protocol, "relax_all", lambda *args: scoring_pool.relaxed)
    thread.start()
    try:
        assert acquired.wait(WATCHDOG)
        metrics = posthoc.evaluate_stage(scoring_pool.out)
        assert metrics["joint/checked_count"] == 2
    finally:
        release.set()
        thread.join(WATCHDOG)
    assert not thread.is_alive()


@pytest.mark.parametrize("phase", ["prepare", "complete"])
def test_spawn_scoring_exceptions_include_traceback(scoring_pool, phase):
    pool = scoring_pool
    digest = "incorrect" if phase == "prepare" else pool.pool["samples_sha256"]
    with pytest.raises(RuntimeError, match="Traceback") as error:
        posthoc._overlap_score(pool.out, pool.saved, pool.pool["step"], digest, lambda: [])
    assert ("Sample hash mismatch" if phase == "prepare" else "exactly one record") in str(error.value)


def test_killed_spawn_scoring_reports_exit_code(scoring_pool, monkeypatch):
    observer = observe_scoring_process(monkeypatch)

    def relax():
        assert observer.started.wait(WATCHDOG)
        os.kill(observer.processes[0].pid, signal.SIGKILL)
        return scoring_pool.relaxed

    try:
        with pytest.raises(RuntimeError, match="died with exit code -9"):
            posthoc._overlap_score(scoring_pool.out, scoring_pool.saved, scoring_pool.pool["step"],
                                   scoring_pool.pool["samples_sha256"], relax)
        assert observer.processes[0].exitcode == -signal.SIGKILL
    finally:
        observer.reader.close()
        observer.writer.close()


def _pause_with_symmetry_worker(connection, out, settings, step, digest, observations, release):
    start = multiprocessing.process.BaseProcess.start.__code__

    def observe(frame, event, arg):
        if event == "return" and frame.f_code is start:
            observations.send(frame.f_locals["self"].pid)
            assert release.wait(WATCHDOG)

    sys.setprofile(observe)
    try:
        posthoc._score_pool_worker(connection, out, settings, step, digest)
    finally:
        sys.setprofile(None)
        observations.close()


def test_relaxation_failure_kills_group_before_reaping_leader(scoring_pool, monkeypatch):
    import ctypes

    libc = ctypes.CDLL(None)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    release = context.Event()
    processes, signals = [], []
    worker = None

    def process(*, target, args):
        child = context.Process(target=_pause_with_symmetry_worker, args=(*args, writer, release))
        processes.append(child)
        return child

    monkeypatch.setattr(posthoc, "multiprocessing", SimpleNamespace(
        get_context=lambda method: SimpleNamespace(Pipe=context.Pipe, Process=process)))
    killpg = os.killpg

    def signal_group(pid, sig):
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        signals.append((pid, sig))
        killpg(pid, sig)

    monkeypatch.setattr(posthoc.os, "killpg", signal_group)

    def relax():
        nonlocal worker
        assert reader.poll(WATCHDOG)
        worker = reader.recv()
        raise ValueError("relaxation failed")

    try:
        with pytest.raises(ValueError, match="relaxation failed"):
            posthoc._overlap_score(scoring_pool.out, scoring_pool.saved, scoring_pool.pool["step"],
                                   scoring_pool.pool["samples_sha256"], relax)
        assert signals == [(processes[0].pid, signal.SIGTERM)]
        assert processes[0].exitcode == -signal.SIGTERM
        deadline = time.monotonic() + WATCHDOG
        while time.monotonic() < deadline:
            pid, status = os.waitpid(worker, os.WNOHANG)
            if pid:
                assert os.waitstatus_to_exitcode(status) == -signal.SIGTERM
                break
            time.sleep(.01)
        else:
            pytest.fail("Symmetry worker did not exit")
    finally:
        reader.close()
        writer.close()
        assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def test_spawn_scoring_exits_when_parent_connection_closes(scoring_pool):
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe()
    process = context.Process(target=posthoc._score_pool_worker,
                              args=(sender, str(scoring_pool.out), scoring_pool.saved,
                                    scoring_pool.pool["step"], scoring_pool.pool["samples_sha256"]))
    process.start()
    sender.close()
    try:
        assert receiver.poll(WATCHDOG)
        assert receiver.recv() == ("ready", None)
        receiver.close()
        process.join(WATCHDOG)
        assert process.exitcode == 0
    finally:
        receiver.close()
        if process.is_alive():
            process.kill()
        process.join(WATCHDOG)
