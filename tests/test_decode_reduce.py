"""Bitwise differential checks and bounded-work tests for reduction.

The full fixture replay, 2,160-case fuzz, and warmed timing are marked slow. The fixture checks are
maintainer-only: they need a captured reduction fixture that is not released, and skip without it.
Use pytest -s to see fallback counts and timings, including successful tests.
"""
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from orbitol import contracts as C
from orbitol.decode import _niggli_safe, reduced_cell
from orbitol.decode_reduce import _lll, _sphere_images, reduce_lattices
from orbitol.jit import ENABLED


FIXTURE = Path(C.local_path("ORBITOL_DATA_HOME"), "decode_fixtures", "reduce_lattices.npz")


def _bits_equal(actual, expected):
    """Include signed zero in the equality contract."""
    np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))


@pytest.fixture(scope="module")
def lattice_fixture():
    if not FIXTURE.is_file():
        pytest.skip(f"Maintainer-only parity check: captured reduction fixture {FIXTURE} is not released")
    with np.load(FIXTURE, allow_pickle=False) as data:
        return {key: data[key] for key in ("L", "reduced", "path", "source")}


def _check_fixture(data, indices):
    """Every certified row must match both the path and all 64 bits."""
    lattices = data["L"][indices]
    expected = data["reduced"][indices]
    paths = data["path"][indices]
    sources = data["source"][indices].astype(str)
    result, codes = reduce_lattices(lattices)
    certified = codes != 2
    _bits_equal(result[certified], expected[certified])
    np.testing.assert_array_equal(codes[certified], paths[certified])
    for source in np.unique(sources):
        selected = sources == source
        fallback = int(np.sum(codes[selected] == 2))
        print(f"{source}: code2={fallback}/{int(selected.sum())}")
        if "s150" in source or "150-step" in source:
            assert fallback == 0, source


@pytest.mark.data
def test_fixture_smoke(lattice_fixture):
    sources = lattice_fixture["source"]
    indices = np.concatenate([np.flatnonzero(sources == source)[:24] for source in np.unique(sources)])
    _check_fixture(lattice_fixture, indices)


@pytest.mark.slow
@pytest.mark.data
def test_fixture_parity(lattice_fixture):
    _check_fixture(lattice_fixture, np.arange(len(lattice_fixture["L"])))


def _fuzz_cases():
    """Reproducible cells, including both determinant signs and ties."""
    rng = np.random.default_rng(7319)
    for unused in range(600):
        yield "random", rng.normal(size=(3, 3)) * rng.uniform(0.7, 8)
    for unused in range(500):
        nice = Lattice.from_parameters(*rng.uniform(2, 7, 3), *rng.uniform(70, 110, 3)).matrix
        transform = np.eye(3)
        for step in range(4):
            i, j = rng.choice(3, 2, replace=False)
            transform[i] += rng.integers(-3, 4) * transform[j]
        yield "unimodular", transform @ nice
    for angle in (60, 90, 120):
        for offset in np.concatenate(([0.], np.logspace(-12, -3, 9), -np.logspace(-12, -3, 9))):
            for variant in range(10):
                lengths = [3.0, 3.0 + offset, 4.0 + variant / 10]
                angles = [90., 90., 90.]
                angles[variant % 3] = angle + offset
                yield "ties", Lattice.from_parameters(*lengths, *angles).matrix
    for unused in range(180):
        shear = rng.uniform(10, 1000)
        yield "shear", np.array([[3., 0., 0.], [shear, 4., 0.], [shear / 3, -shear, 5.]])
    for scale in np.logspace(-100, 100, 100):
        yield "scale", np.diag([2., 3., 4.]) * scale
    for small in np.logspace(-16, -3, 100):
        yield "singular", np.array([[3., 0., 0.], [3., small, 0.], [0., 0., 3.]])
    for i in range(110):
        matrix = np.eye(3) * 3
        matrix.flat[i % 9] = (np.nan, np.inf, -np.inf)[i % 3]
        yield "nonfinite", matrix


def _check_fuzz(cases):
    labels, matrices = zip(*cases)
    lattices = np.asarray(matrices)
    result, codes = reduce_lattices(lattices)
    counts = Counter()
    for i, (label, matrix, code) in enumerate(zip(labels, lattices, codes)):
        counts[(label, int(code))] += 1
        if code == 2:
            # The live reference can hang/crash on degenerate cells. It has no
            # equality obligation when the fast path explicitly declines.
            continue
        structure = Structure(matrix, ["H"], [[0., 0., 0.]])
        expected_path = 0 if _niggli_safe(structure.lattice) else 1
        assert code == expected_path, (i, label)
        expected = reduced_cell(structure).lattice.matrix
        try:
            _bits_equal(result[i], expected)
        except AssertionError as error:
            raise AssertionError(f"fuzz {i} ({label}), code {code}, lattice={matrix.tolist()}") from error
    print(f"fuzz generated={len(cases)}, live reference={int(np.sum(codes != 2))}, counts={dict(sorted(counts.items()))}")
    assert np.sum(codes != 2) > len(cases) // 2


def test_fuzz_smoke():
    cases = list(_fuzz_cases())
    _check_fuzz(cases[::20])


@pytest.mark.slow
def test_differential_fuzz():
    cases = list(_fuzz_cases())
    assert len(cases) >= 2000
    _check_fuzz(cases)


def test_contract_and_input_unchanged():
    storage = np.tile(np.diag([3., 4., 5.]), (4, 1, 1))
    lattices = storage[::2]
    before = storage.copy()
    result, codes = reduce_lattices(lattices)
    assert result.shape == lattices.shape and result.dtype == np.float64
    assert codes.shape == (2,) and codes.dtype == np.int8
    assert np.all(codes == 0)
    _bits_equal(storage, before)
    empty, empty_codes = reduce_lattices(np.empty((0, 3, 3)))
    assert empty.shape == (0, 3, 3) and empty_codes.shape == (0,)
    with pytest.raises(ValueError):
        reduce_lattices(np.eye(3))
    with pytest.raises(TypeError):
        reduce_lattices(lattices.astype(np.float32))


def test_degenerate_and_scale_limits():
    cases = [np.zeros((3, 3)), np.ones((3, 3)), np.eye(3) * 1e200, np.eye(3) * 1e-200]
    for value in (np.inf, -np.inf, np.nan):
        matrix = np.eye(3)
        matrix[0, 1] = value
        cases.append(matrix)
    unused, codes = reduce_lattices(np.stack(cases))
    np.testing.assert_array_equal(codes, np.full(len(cases), 2))


def test_lll_only_and_keep_original():
    matrices = np.array([
        [[.2, 0., 0.], [0., .3, 0.], [0., 0., .4]],
        [[.2, -0., 0.], [.6, .3, 0.], [0., 0., .4]],
        [[-3., 0., 0.], [0., -4., 0.], [0., 0., 5.]],
    ])
    result, codes = reduce_lattices(matrices)
    np.testing.assert_array_equal(codes, [1, 1, 0])
    for i, matrix in enumerate(matrices):
        expected = reduced_cell(Structure(matrix, ["H"], [[0., 0., 0.]])).lattice.matrix
        _bits_equal(result[i], expected)
    _bits_equal(result[0], matrices[0])


def test_origin_sphere_order():
    rng = np.random.default_rng(211)
    for radius in (0.8, 1.0, 4.3, 8.0):
        for unused in range(4):
            matrix = np.diag([2., 3., 4.]) + rng.normal(size=(3, 3)) * 0.25
            images, distances, valid = _sphere_images(matrix, radius)
            assert valid
            expected, expected_distances, unused_indices, unused_images = Lattice(matrix).get_points_in_sphere(
                [[0., 0., 0.]], [0., 0., 0.], radius, zip_results=False
            )
            _bits_equal(images, expected)
            _bits_equal(distances, expected_distances)


def test_lll_rounding_and_least_squares():
    rng = np.random.default_rng(91)
    matrices = list(rng.normal(size=(32, 3, 3)) * 3)
    # Centred bases exercise projections either side of .5 after the actual
    # least-squares solve, where replacing the solver with division is unsafe.
    for a, b in rng.uniform(1, 5, (32, 2)):
        matrices.append(np.array([[-a, a, b], [a, -a, b], [a, a, -b]]))
    for matrix in matrices:
        result, valid, unused_tie = _lll(matrix)
        assert valid
        _bits_equal(result, Lattice(matrix).lll_matrix)


def test_work_and_tolerance_fallbacks():
    # Image budget before allocation, then candidate budget during traversal.
    for matrix in (np.diag([1e-6, 1., 1.]), np.eye(3) * .1):
        unused_images, unused_distances, valid = _sphere_images(matrix, 1.)
        assert not valid
    # The guard's .5-A boundary must not be guessed across rounding errors.
    matrices = np.array([np.diag([length, 2., 3.]) for length in
                         (.5, np.nextafter(.5, 0.), np.nextafter(.5, 1.))])
    unused_result, codes = reduce_lattices(matrices)
    np.testing.assert_array_equal(codes, [2, 2, 2])
    # Regression: 100 two-step fixture rows disagree with the live reference
    # at this symmetric LLL tie. Neither basis can be certified for both.
    tiny_bcc = 0.0033689734995427335 * (np.ones((3, 3)) - 2 * np.eye(3))
    unused_result, codes = reduce_lattices(tiny_bcc[None])
    assert codes[0] == 2


@pytest.mark.slow
@pytest.mark.data
def test_compiled_timing(lattice_fixture):
    if not ENABLED:
        pytest.skip("Timing requires Numba (ORBITOL_NUMBA not 0)")
    source = lattice_fixture["source"].astype(str)
    selected = np.array(["s150" in value or "150-step" in value for value in source])
    assert selected.any()
    lattices = np.ascontiguousarray(lattice_fixture["L"][selected])
    reduce_lattices(lattices[:1])
    seconds = []
    for unused in range(5):
        start = time.perf_counter()
        unused_result, codes = reduce_lattices(lattices)
        seconds.append(time.perf_counter() - start)
        assert np.all(codes == 0)
    print(f"compiled median={np.median(seconds) * 1e6 / len(lattices):.3f} us/cell, n={len(lattices)}")
