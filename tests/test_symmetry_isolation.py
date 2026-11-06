"""Isolated spglib detection agrees with in-process detection and survives a crashing worker."""
import os
import json
from types import SimpleNamespace

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from orbitol.spglib_worker import detect_cells
from orbitol.symmetry_panel import SYMPRECS, SymmetryPanel, detect_group, detect_many, group_histogram


def rutile():
    return Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(4.59, 2.96), ["Ti", "O"],
                                     [[0, 0, 0], [0.305, 0.305, 0]])


def structures():
    base = rutile()
    perturbed = base.copy()
    for i, offset in enumerate(np.random.default_rng(17).normal(0, 0.06, (len(base), 3))):
        perturbed.translate_sites(i, offset, frac_coords=True)
    rocksalt = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    return [base, perturbed, None, rocksalt] * 5


def abort_on_chlorine(cells, symprecs, angle_tolerance):
    # Stands in for spglib corrupting the heap: the worker process dies outright.
    if any(cell is not None and 17 in cell[2] for cell in cells):
        os.abort()
    return detect_cells(cells, symprecs, angle_tolerance)


@pytest.mark.parametrize("workers", [1, 2])
def test_isolated_detection_matches_in_process(workers):
    items = structures()
    expected = np.asarray([[detect_group(s, p) for p in SYMPRECS] for s in items], dtype=np.int16)
    assert expected[0, 0] == 136 and expected[3, 0] == 225 and expected[2, 0] == 0
    np.testing.assert_array_equal(detect_many(items, workers=workers, chunk=3), expected)
    assert detect_many([], workers=2).shape == (0, len(SYMPRECS))


def test_crashing_worker_only_fails_its_structures():
    items = structures()
    expected = np.asarray([[detect_group(s, p) for p in SYMPRECS] for s in items], dtype=np.int16)
    expected[3::4] = 0  # every rocksalt cell aborts its worker
    np.testing.assert_array_equal(detect_many(items, workers=2, chunk=4, function=abort_on_chlorine), expected)


@pytest.mark.parametrize("items", [[], [None, None]])
def test_empty_detection_never_starts_workers(monkeypatch, items):
    def unexpected(*args, **kwargs):
        pytest.fail("No cells require detection")

    monkeypatch.setattr("orbitol.symmetry_panel.ProcessPoolExecutor", unexpected)
    groups = detect_many(items)
    assert groups.dtype == np.int16
    np.testing.assert_array_equal(groups, np.zeros((len(items), len(SYMPRECS)), dtype=np.int16))


@pytest.mark.parametrize("endpoints", [[3, 0, 1], []])
def test_panel_precomputed_groups_preserve_metrics_and_requested_retention(monkeypatch, endpoints):
    raw = structures()[:4]
    records = [SimpleNamespace(sample_index=i, final_structure=raw[i]) for i in endpoints]
    raw_groups = detect_many(raw, workers=2)
    panel = SymmetryPanel.__new__(SymmetryPanel)
    panel._reference = {str(p): group_histogram(raw_groups[:, j]).tolist() for j, p in enumerate(SYMPRECS)}
    requested = [136, 136, 1, 225]
    expected = panel.evaluate(raw, records, requested_G=requested)

    def unexpected(*args, **kwargs):
        pytest.fail("Precomputed groups must avoid detection")

    monkeypatch.setattr("orbitol.symmetry_panel.detect_many", unexpected)
    actual = panel.evaluate(raw, records, requested_G=requested,
                            precomputed_groups=dict(raw=raw_groups, relaxed=raw_groups[endpoints]))
    assert json.dumps(actual) == json.dumps(expected)
    assert actual["sym/relaxed/0.1/retention_total"] == len(endpoints)
    if endpoints:
        assert actual["sym/relaxed/0.1/retention_count"] == 2
