"""One-shot/geometry scoring, denominators and portable persistence."""
from dataclasses import replace
import json

import numpy as np
import pytest

from orbitol import eval_protocol as protocol_module
from orbitol.eval_adapter import _sample_items
from orbitol.eval_diagnostics import RELAXATION_FIELDS
from orbitol.eval_protocol import _relax_metrics, relax_all, score
from orbitol.eval_score import RELAXED_FIELDS, read_relaxed, write_csv
from test_eval_protocol import Relaxer, evaluator, protocol  # noqa: F401 -- shared real references


class TelemetryRelaxer(Relaxer):
    initial = dict(Li=0., Na=.1, K=.3, Al=0., Cu=.05)
    forces = dict(Li=.05, Na=.06, K=None, Al=.04, Cu=.1)

    def __init__(self):
        super().__init__()
        self.inputs = []

    def relax_many(self, structures, **kwargs):
        results = super().relax_many(structures, **kwargs)
        self.inputs.extend(structures)
        for initial, result in zip(structures, results, strict=True):
            symbol = initial[0].specie.symbol
            result.update(initial_energy=self.initial[symbol] * len(initial), initial_fmax=.4,
                          final_fmax=self.forces[symbol], final_max_force=.01)
        return results


def test_oneshot_denominators_old_metrics_and_sync_diagnostics(protocol):
    backend = TelemetryRelaxer()
    runner = evaluator(protocol, backend)
    metrics = runner.evaluate(protocol.rows, step=0)
    rows = runner.per_sample
    assert metrics["sample_ema/SUN_oneshot"] == 1 / 8
    assert metrics["sample_ema/MSUN_oneshot"] == 2 / 8
    assert metrics["sample_ema/relax/telemetry_count"] == 6
    assert metrics["sample_ema/relax/converged_rate"] == 3 / 5
    assert metrics["sample_ema/relax/sg_kept_rate"] == 1
    assert rows[1]["converged"] and not rows[3]["converged"]
    assert rows[4]["converged"] is None
    assert rows[1]["energy_drop_meV"] == 10
    assert rows[1]["e_above_hull_oneshot"] == 0
    assert rows[3]["e_above_hull_oneshot"] == .1
    assert rows[6]["e_above_hull_oneshot"] is None  # Al absent from the hull
    assert all(rows[i]["initial_energy_eV"] is None and not rows[i]["MSUN_oneshot"] for i in (0, 5))
    for row in rows:
        assert "SUN_raw" not in row and "MSUN_raw" not in row
        assert row["SUN_raw_UN"] == bool(row["stable"] and row["raw_is_un"])
        assert row["MSUN_raw_UN"] == bool(row["metastable"] and row["raw_is_un"])
    for diagnostic in runner.relaxation_diagnostics():
        if diagnostic["relaxation_outcome"] != "relaxation_failed":
            row = rows[diagnostic["pool_index"]]
            assert {key: diagnostic[key] for key in RELAXATION_FIELDS} == {key: row[key] for key in RELAXATION_FIELDS}
            assert diagnostic["final_max_force"] == .01
    json.dumps(runner.relaxation_diagnostics(), allow_nan=False)
    baseline = evaluator(protocol).evaluate(protocol.rows, step=0)
    retained = {k: v for k, v in metrics.items() if k in baseline}
    assert retained == pytest.approx(baseline, nan_ok=True)
    assert not any("oneshot" in k or "/relax/" in k for k in baseline)


@pytest.mark.parametrize("method", ["uncorrected", "mp2020_like"])
def test_score_reuses_exact_evaluator_relaxation_inputs(protocol, monkeypatch, method):
    protocol.settings = replace(protocol.settings, thermo_ehull_method=method)
    runner = evaluator(protocol)
    backend = TelemetryRelaxer()
    items = _sample_items(protocol.rows)
    records = relax_all(items, backend, 500)
    constructed = [s for s in backend.inputs]
    evaluator_calls, geometry_inputs, hull_calls = [], [], []
    original_evaluator = protocol_module.compute_evaluator_metrics
    original_geometry = protocol_module.relaxation_geometry
    name = f"compute_e_above_hull_{method}"
    original_hull = getattr(protocol_module, name)

    def evaluate(*args, **kwargs):
        result = original_evaluator(*args, **kwargs)
        evaluator_calls.append(result)
        return result

    def geometry(initial, final, *args):
        geometry_inputs.append(initial)
        return original_geometry(initial, final, *args)

    def hull(ppd, structure, energy, **kwargs):
        hull_calls.append((structure, energy, kwargs.get("mp2020_compat")))
        return original_hull(ppd, structure, energy, **kwargs)

    monkeypatch.setattr(protocol_module, "compute_evaluator_metrics", evaluate)
    monkeypatch.setattr(protocol_module, "relaxation_geometry", geometry)
    monkeypatch.setattr(protocol_module, name, hull)
    score(items, records, references=runner.references, ppd=runner.ppd,
          settings=runner.settings, tag=runner.tag, step=0)
    assert len(evaluator_calls) == 1
    assert len(constructed) == len(geometry_inputs) == 6
    for initial, scored in zip(constructed, geometry_inputs, strict=True):
        assert initial.species == scored.species
        np.testing.assert_array_equal(initial.frac_coords, scored.frac_coords)
        np.testing.assert_array_equal(initial.lattice.matrix, scored.lattice.matrix)
        assert any(scored is c.structure for c in evaluator_calls[0].pred_crys_list if c.constructed)
    assert len(hull_calls) == 12
    for i, scored in enumerate(geometry_inputs):
        assert hull_calls[2 * i + 1][0] is scored
        assert hull_calls[2 * i + 1][1] == backend.initial[scored[0].specie.symbol] * len(scored)
        if method == "mp2020_like":
            assert hull_calls[2 * i][2] is hull_calls[2 * i + 1][2] is hull_calls[0][2]


def test_csv_roundtrip_and_old_records_score_without_new_aggregates(protocol, tmp_path):
    runner = evaluator(protocol)
    items = _sample_items(protocol.rows)
    records = relax_all(items, TelemetryRelaxer(), 500)
    path = tmp_path / "relaxed.csv.gz"
    write_csv(path, records, fields=RELAXED_FIELDS)
    restored = read_relaxed(path)
    kwargs = dict(references=runner.references, ppd=runner.ppd, settings=runner.settings, tag=runner.tag, step=0)
    expected, expected_rows = score(items, records, **kwargs)
    actual, actual_rows = score(items, restored, **kwargs)
    assert actual == pytest.approx(expected, nan_ok=True)
    assert actual_rows == expected_rows
    write_csv(path, records, fields=RELAXED_FIELDS[:7])
    old = read_relaxed(path)
    for record in old:
        assert all(record[key] is None for key in RELAXED_FIELDS[7:])
    for record in old:  # Also accept old in-memory/JSON records with absent fields.
        for key in RELAXED_FIELDS[7:]:
            del record[key]
    metrics, rows = score(items, old, **kwargs)
    assert not any("oneshot" in k or "/relax/" in k for k in metrics)
    assert all(row["initial_energy_eV"] is None for row in rows)
    write_csv(tmp_path / "per_sample.csv.gz", actual_rows)


def test_aggregate_subsets_finite_values_and_complete_key_set():
    quantities = ("rms_disp", "max_disp", "strain", "volume_change_pct", "energy_drop_meV")
    rows = []
    for i, value in enumerate((1., 3., 9., 99.)):
        rows.append(dict(relax_status="ok", initial_energy_eV=0., initial_fmax=value,
                         final_fmax=.01 if i % 2 else .1, converged=bool(i % 2),
                         sg_kept=i == 0, MSUN=i < 2, SUN_oneshot=i == 0, MSUN_oneshot=i < 2,
                         e_above_hull_oneshot=value if i < 3 else None,
                         **{key: value for key in quantities},
                         **{"raw_space_group_0.1": 221, "relaxed_space_group_0.1": (221, 1, 1, 0)[i]}))
    rows[2]["max_disp"] = None
    rows[3]["energy_drop_meV"] = np.inf
    rows.append(dict(relax_status="failed", SUN_oneshot=False, MSUN_oneshot=False))
    metrics = _relax_metrics(rows, "sample_ema")
    expected_keys = {"sample_ema/SUN_oneshot", "sample_ema/MSUN_oneshot"}
    expected_keys.update(f"sample_ema/stability/e_above_hull_oneshot_{q}" for q in ("median", "p10", "p90"))
    for key in (*quantities, "initial_fmax", "final_fmax"):
        expected_keys.update(f"sample_ema/relax/{key}_{q}" for q in ("median", "mean", "p90"))
    expected_keys.update(f"sample_ema/relax/{key}_median_msun" for key in quantities)
    expected_keys.update(f"sample_ema/relax/{key}_median_sg_{split}"
                         for key in ("rms_disp", "strain", "energy_drop_meV") for split in ("kept", "changed"))
    expected_keys.update(f"sample_ema/relax/{key}" for key in
                         ("converged_rate", "sg_kept_rate", "sg_kept_rate_msun", "telemetry_count"))
    assert set(metrics) == expected_keys
    assert len(expected_keys) == 41
    assert metrics["sample_ema/SUN_oneshot"] == 1 / 5
    assert metrics["sample_ema/MSUN_oneshot"] == 2 / 5
    assert metrics["sample_ema/relax/telemetry_count"] == 4
    assert metrics["sample_ema/relax/converged_rate"] == .5
    assert metrics["sample_ema/relax/sg_kept_rate"] == 1 / 3
    assert metrics["sample_ema/relax/sg_kept_rate_msun"] == .5
    assert metrics["sample_ema/relax/rms_disp_median_sg_changed"] == 6
    assert metrics["sample_ema/relax/rms_disp_median_sg_kept"] == 1
    assert metrics["sample_ema/relax/rms_disp_median_msun"] == 2
    assert metrics["sample_ema/relax/energy_drop_meV_mean"] == pytest.approx(13 / 3)
    assert metrics["sample_ema/relax/max_disp_mean"] == pytest.approx(103 / 3)
    assert metrics["sample_ema/relax/rms_disp_median"] == 6
    assert metrics["sample_ema/relax/rms_disp_p90"] == pytest.approx(np.percentile([1, 3, 9, 99], 90))
    assert metrics["sample_ema/stability/e_above_hull_oneshot_p10"] == pytest.approx(1.4)
    assert all(np.isfinite(v) for v in metrics.values())


def test_empty_subsets_omit_keys_and_no_telemetry_omits_all():
    row = dict(relax_status="ok", initial_energy_eV=1., final_fmax=None,
               SUN_oneshot=False, MSUN_oneshot=False, MSUN=False,
               **{"raw_space_group_0.1": 0, "relaxed_space_group_0.1": 0})
    assert _relax_metrics([row], "sample_ema") == {
        "sample_ema/SUN_oneshot": 0., "sample_ema/MSUN_oneshot": 0., "sample_ema/relax/telemetry_count": 1,
    }
    assert _relax_metrics([{**row, "initial_energy_eV": None}], "sample_ema") == {}
    assert _relax_metrics([], "sample_ema") == {}
