"""Strict decoder parity, exceptional paths and compiled fixture timing."""
from collections import Counter
from functools import lru_cache
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol import decode as reference
from orbitol import decode_fast as fast
from orbitol import contracts as C
from orbitol.contracts import Skeleton
from orbitol.jit import ENABLED
from orbitol.sample import load_encoding


FIXTURE_ROOT = Path(C.local_path("ORBITOL_DATA_HOME"), "decode_fixtures")
FIXTURES = (
    "d256_s150_c60", "d256_s10_c60", "d256_s10_c0", "d256_s5_c60", "d256_s3_c60",
    "d256_s2_c60", "d512_s150_c60", "d512_s10_c0", "d512_s3_c60", "d512_s2_c60",
)


@pytest.fixture(scope="module")
def encoding():
    return load_encoding()


@pytest.fixture(scope="module")
def tables(frozen_tables):
    return frozen_tables


@lru_cache(maxsize=len(FIXTURES))
def _fixture(name):
    """Load a captured decoder fixture, or skip: maintainer-only parity checks (fixtures not released)."""
    path = FIXTURE_ROOT / name
    required = ("decode_inputs.pt", "expected_outputs.pt", "tables.npz", "allowed_mask.pt")
    if not all((path / part).is_file() for part in required):
        pytest.skip(f"Maintainer-only parity check: captured decoder fixture {path} is not released")
    with np.load(path / "tables.npz", allow_pickle=False) as data:
        arrays = dict(data)
    meta_path = path / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    capture_cpu = meta.get("cpu")
    return SimpleNamespace(
        path=path, tables=arrays, live=None, capture_cpu=capture_cpu,
        compare_stored=capture_cpu is not None and capture_cpu == _cpu_model(),
        chunks=torch.load(path / "decode_inputs.pt", map_location="cpu", weights_only=False),
        expected=torch.load(path / "expected_outputs.pt", map_location="cpu", weights_only=False),
        allowed=torch.load(path / "allowed_mask.pt", map_location="cpu", weights_only=False),
    )


@lru_cache(maxsize=1)
def _cpu_model():
    """Match the capture's CPU model; unknown hardware uses live rows only."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "model name":
                return value.strip()
    except OSError:
        pass
    return None


def _indices(chunk, expected, limit=6):
    """Cover each chunk and every observed failure reason in small tests."""
    count = len(chunk["skeletons"])
    selected = set(np.linspace(0, count - 1, min(count, limit), dtype=int).tolist())
    reasons = set()
    for i, reason in expected["failures"].items():
        if reason not in reasons:
            selected.add(i)
            reasons.add(reason)
    return np.array(sorted(selected), dtype=np.int64)


def _inputs(chunk, indices=None):
    if indices is None:
        indices = np.arange(len(chunk["skeletons"]))
    return (
        {key: value[indices] for key, value in chunk["sampler_out"].items()},
        [Skeleton(**chunk["skeletons"][i]) for i in indices],
    )


def _live_full(fixture, encoding):
    """Run the live oracle once per fixture in this process, outside timing.

    Retain rows only, so the timing test can reuse a prior full replay without
    retaining thousands of primitive objects or running reference work timed.
    """
    if fixture.live is None:
        fixture.live = []
        for chunk in fixture.chunks:
            output, skeletons = _inputs(chunk)
            decoded = reference.decode_batch(output, skeletons, fixture.tables, encoding,
                                             fixture.allowed, n_max=chunk["n_max"])
            fixture.live.append(dict(rows=decoded.rows, failures=decoded.failures,
                                     contains_a=decoded.contains_a, requested_G=decoded.requested_G))
    return fixture.live


def _expected(expected, indices):
    return dict(
        rows={key: value[indices] for key, value in expected["rows"].items()},
        failures={j: expected["failures"][i] for j, i in enumerate(indices) if i in expected["failures"]},
        contains_a=np.asarray(expected["contains_a"])[indices],
        requested_G=np.asarray(expected["requested_G"])[indices],
    )


def _assert_parity(got, expected):
    if isinstance(expected, reference.DecodedBatch):
        expected = dict(rows=expected.rows, failures=expected.failures,
                        contains_a=expected.contains_a, requested_G=expected.requested_G)
    assert isinstance(got, reference.DecodedBatch)
    assert isinstance(got.rows, reference.CrystaliteRows)
    assert set(got.rows) == {"A0", "F1", "Y1", "pad_mask"}
    for key, dtype in (("A0", torch.int64), ("F1", torch.float32),
                       ("Y1", torch.float32), ("pad_mask", torch.bool)):
        assert got.rows[key].dtype == dtype
        assert got.rows[key].shape == expected["rows"][key].shape
    for key in ("A0", "pad_mask"):
        assert torch.equal(got.rows[key], expected["rows"][key]), key
    assert torch.equal(got.rows["Y1"].view(torch.int32), expected["rows"]["Y1"].view(torch.int32)), "Y1 bits"
    fractional = got.rows["F1"]
    assert torch.isfinite(fractional).all()
    assert ((fractional >= 0) & (fractional < 1)).all()
    delta = ((fractional.double() - expected["rows"]["F1"].double() + 0.5) % 1.0 - 0.5).abs()
    assert not delta.numel() or delta.max().item() <= 1e-7
    assert got.failures == expected["failures"]
    assert got.rows.failures == got.failures
    np.testing.assert_array_equal(got.contains_a, expected["contains_a"])
    np.testing.assert_array_equal(got.requested_G, expected["requested_G"])
    for i in got.failures:
        assert got.rows["pad_mask"][i].all()
        for key in ("A0", "F1", "Y1"):
            assert not got.rows[key][i].any()


def _assert_structures(got, expected):
    for actual, wanted in zip(got.structures, expected.structures, strict=True):
        if wanted is None:
            assert actual is None
            continue
        assert actual.species == wanted.species
        np.testing.assert_array_equal(actual.lattice.matrix, wanted.lattice.matrix)
        delta = (actual.frac_coords - wanted.frac_coords + 0.5) % 1.0 - 0.5
        np.testing.assert_allclose(delta, 0, rtol=0, atol=1e-12)


def _report(name, mode, fixture, failures, record_property):
    result = dict(fixture=name, mode=mode, path=str(fixture.path),
                  failures=dict(failures), stats=dict(fast.STATS), cpu=_cpu_model(),
                  capture_cpu=fixture.capture_cpu, compared_stored=fixture.compare_stored)
    record_property("decode_fast", json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True))


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.data
def test_fixture_slice(name, encoding, record_property):
    fixture = _fixture(name)
    fast.STATS.clear()
    failures = Counter()
    for chunk, expected in zip(fixture.chunks, fixture.expected, strict=True):
        indices = _indices(chunk, expected)
        output, skeletons = _inputs(chunk, indices)
        live = reference.decode_batch(output, skeletons, fixture.tables, encoding,
                                      fixture.allowed, n_max=chunk["n_max"])
        actual = fast.decode_batch(output, skeletons, fixture.tables, encoding,
                                   fixture.allowed, n_max=chunk["n_max"])
        _assert_parity(actual, live)
        if fixture.compare_stored:
            _assert_parity(actual, _expected(expected, indices))
        assert actual.structures is None
        failures.update(actual.failures.values())
    _report(name, "slice", fixture, failures, record_property)


@pytest.mark.slow
@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.data
def test_fixture_full(name, encoding, record_property):
    fixture = _fixture(name)
    live = _live_full(fixture, encoding)
    fast.STATS.clear()
    failures = Counter()
    results = []
    for chunk, expected, wanted in zip(fixture.chunks, fixture.expected, live, strict=True):
        output, skeletons = _inputs(chunk)
        actual = fast.decode_batch(output, skeletons, fixture.tables, encoding,
                                   fixture.allowed, n_max=chunk["n_max"])
        results.append((actual, wanted, expected))
        assert actual.structures is None
        failures.update(actual.failures.values())
    _report(name, "full", fixture, failures, record_property)
    for actual, wanted, expected in results:
        _assert_parity(actual, wanted)
        if fixture.compare_stored:
            _assert_parity(actual, expected)


@pytest.mark.parametrize("name", FIXTURES)
@pytest.mark.data
def test_fixture_primitive_structures(name, encoding):
    fixture = _fixture(name)
    for chunk, expected in zip(fixture.chunks, fixture.expected, strict=True):
        indices = _indices(chunk, expected, limit=2)
        output, skeletons = _inputs(chunk, indices)
        kwargs = dict(n_max=chunk["n_max"])
        wanted = reference.decode_batch(output, skeletons, fixture.tables, encoding, fixture.allowed, **kwargs)
        actual = fast.decode_batch(output, skeletons, fixture.tables, encoding, fixture.allowed, structures=True, **kwargs)
        _assert_parity(actual, wanted)
        _assert_structures(actual, wanted)
        if fixture.compare_stored:
            _assert_parity(actual, _expected(expected, indices))


def _rutile(tables, encoding, count=1):
    rows = (tables.row_id(136, "a"), tables.row_id(136, "f"))
    output = dict(
        type=encoding.z_to_enc[torch.tensor([22, 8])].repeat(count, 1, 1),
        u=torch.tensor([[[-0.5, -0.5, -0.5], [-0.1951, -0.5, -0.5]]], dtype=torch.float64).repeat(count, 1, 1),
        z=torch.tensor([[np.log(4.594), 0, 0, 0, 0, np.log(2.959)]], dtype=torch.float64).repeat(count, 1),
    )
    return output, Skeleton(136, rows)


@pytest.mark.parametrize("objects", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_reduction_fallback_keeps_nonleading_padding(tables, encoding, monkeypatch, objects, fixed):
    output, skeleton = _rutile(tables, encoding, 2)
    output["type"] = torch.cat([output["type"], torch.full((2, 1, 16), float("nan"))], dim=1)[:, [0, 2, 1]]
    output["u"] = torch.cat([output["u"], torch.full((2, 1, 3), float("nan"))], dim=1)[:, [0, 2, 1]]
    skel = dict(G=torch.tensor([136, 136]), row=torch.tensor([[skeleton.rows[0], 0, skeleton.rows[1]]] * 2),
                pad_mask=torch.tensor([[False, True, False]] * 2),
                A0=torch.tensor([[14, 0, 14], [14, 0, 14]]))
    if fixed:
        skel["fixed_A0"] = torch.tensor([[14, 0, 14], [22, 0, 8]])
        output["type"][0] = float("nan")
    expected = reference.decode_batch(output, skel, tables, encoding, None, n_max=20)
    monkeypatch.setattr(fast, "_reduce", lambda L: (L.copy(), np.full(len(L), 2, dtype=np.int8)))
    fast.STATS.clear()
    got = fast.decode_batch(output, skel, tables, encoding, None, n_max=20, structures=objects)
    _assert_parity(got, expected)
    if objects:
        _assert_structures(got, expected)
    else:
        assert got.structures is None
    assert fast.STATS == Counter(structures=2, fallback=2, fallback_reduction=2)


@pytest.mark.parametrize("length, fallback", [(1e-3, True), (0.1, False)])
def test_uncertified_cell_and_lll_only_reducer(tables, encoding, length, fallback):
    output = dict(type=encoding.z_to_enc[torch.tensor([14])][None],
                  u=torch.zeros(1, 1, 3, dtype=torch.float64),
                  z=torch.tensor([[np.log(length), 0, np.log(4), 0, 0, np.log(5)]], dtype=torch.float64))
    skeleton = Skeleton(1, (1,))
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    fast.STATS.clear()
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    _assert_parity(got, expected)
    assert fast.STATS["structures"] == 1
    assert fast.STATS["fallback"] == int(fallback)
    assert fast.STATS["fallback_radius_certificate"] == int(fallback)
    _, codes = fast._reduce(np.diag([length, 4, 5])[None])
    np.testing.assert_array_equal(codes, [1])


@pytest.mark.parametrize("kind", [
    "invalid_group", "invalid_shape", "empty_structure", "non_finite", "invalid_rows",
    "reused_fixed_row", "invalid_species", "finite_flag", "type_nan", "z_nan",
])
def test_validation_precedence(tables, encoding, kind):
    output, skeleton = _rutile(tables, encoding)
    if kind == "invalid_group":
        skeleton = Skeleton(0, (), ())
    elif kind == "invalid_shape":
        skeleton = Skeleton(136, skeleton.rows, (22,))
    elif kind == "empty_structure":
        skeleton = Skeleton(136, (), ())
    elif kind == "non_finite":
        skeleton = Skeleton(136, (-1, skeleton.rows[1]), (np.nan, 8))
    elif kind == "invalid_rows":
        skeleton = Skeleton(136, (-1, skeleton.rows[1]), (0, 8))
    elif kind == "reused_fixed_row":
        skeleton = Skeleton(136, (skeleton.rows[0],) * 2, (0, 0))
    elif kind == "invalid_species":
        skeleton = Skeleton(136, skeleton.rows, (0, 8))
        output["z"][0, 0] = float("nan")
    elif kind == "finite_flag":
        output["finite"] = torch.tensor([False])
    elif kind == "type_nan":
        output["type"][0, 0, 0] = float("nan")
    elif kind == "z_nan":
        output["z"][0, 0] = float("nan")
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20, structures=True)
    _assert_parity(got, expected)
    assert got.failures == {0: "non_finite" if kind in ("finite_flag", "type_nan", "z_nan") else kind}
    assert got.structures == [None]


@pytest.mark.parametrize("kind", ["coinciding_orbits", "collapsed_orbit", "too_many_atoms"])
def test_geometry_failure_precedence(tables, encoding, kind):
    output, skeleton = _rutile(tables, encoding)
    n_max = 20
    if kind == "coinciding_orbits":
        output["u"][0, 1, 0] = -0.5  # also collapses the free orbit
    elif kind == "collapsed_orbit":
        output["u"][0, 1, 0] = 0
    else:
        n_max = 5
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=n_max)
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=n_max, structures=True)
    _assert_parity(got, expected)
    assert got.failures == {0: kind}
    assert got.contains_a[0] == (kind == "too_many_atoms")
    assert got.structures == [None]


@pytest.mark.parametrize("distance", [0.000999, 0.001, 0.001001])
def test_collision_cutoff(tables, encoding, distance):
    output = dict(type=encoding.z_to_enc[torch.tensor([14, 8])][None],
                  u=torch.tensor([[[-0.5, -0.5, -0.5], [-0.5 + distance, -0.5, -0.5]]], dtype=torch.float64),
                  z=torch.zeros(1, 6, dtype=torch.float64))
    skeleton = Skeleton(1, (1, 1))
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    fast.STATS.clear()
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    _assert_parity(got, expected)
    assert fast.STATS["fallback_collision_boundary"] == int(distance == 0.001)


@pytest.mark.parametrize("threshold", ["coincidence", "containment"])
def test_large_lattice_arithmetic_guard(tables, encoding, threshold):
    frac = np.array([[0.3127452273270544, 0.125, 0.25],
                     [0.3127452273270554, 0.125, 0.25]])
    if threshold == "coincidence":
        lattice_length = 1e12
        species = [14, 8]
        arrays = tables.arrays
        cause = "fallback_collision_boundary"
    else:
        lattice_length = 1e9
        species = [14]
        arrays = dict(tables.arrays)
        arrays["group_ops_t"] = arrays["group_ops_t"].copy()
        arrays["group_ops_t"][1, 0, 0] = frac[1, 0] - frac[0, 0]
        frac = frac[:1]
        cause = "fallback_containment_boundary"
    output = dict(type=encoding.z_to_enc[torch.tensor(species)][None],
                  u=torch.tensor(frac - 0.5)[None],
                  z=torch.tensor([[np.log(lattice_length), 0, np.log(3), 0, 0, np.log(4)]],
                                 dtype=torch.float64))
    skeleton = Skeleton(1, (1,) * len(species), tuple(species))
    expected = reference.decode_batch(output, skeleton, arrays, encoding, None, n_max=20)
    if threshold == "coincidence":
        assert expected.failures == {}
    fast.STATS.clear()
    got = fast.decode_batch(output, skeleton, arrays, encoding, None, n_max=20, structures=True)
    _assert_parity(got, expected)
    _assert_structures(got, expected)
    assert fast.STATS == Counter({"structures": 1, "fallback": 1, cause: 1})


@pytest.mark.parametrize("dtype", [np.float16, np.longdouble])
def test_unsupported_validator_dtypes(tables, encoding, dtype):
    output = dict(type=encoding.z_to_enc[torch.tensor([14])][None],
                  u=torch.zeros(1, 1, 3, dtype=torch.float64),
                  z=torch.zeros(1, 6, dtype=torch.float64))
    skeleton = Skeleton(1, (dtype(1),), (dtype(14),))
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    assert expected.failures == {}
    _assert_parity(got, expected)


@pytest.mark.parametrize("scale", [0.999, 1.0, 1.001])
def test_containment_cutoff(tables, encoding, scale):
    arrays = dict(tables.arrays)
    arrays["group_ops_t"] = arrays["group_ops_t"].copy()
    arrays["group_ops_t"][1, 0, 0] = scale * 1e-6
    output = dict(type=encoding.z_to_enc[torch.tensor([14])][None],
                  u=torch.full((1, 1, 3), -0.5, dtype=torch.float64),
                  z=torch.zeros(1, 6, dtype=torch.float64))
    skeleton = Skeleton(1, (1,))
    expected = reference.decode_batch(output, skeleton, arrays, encoding, None, n_max=20)
    fast.STATS.clear()
    got = fast.decode_batch(output, skeleton, arrays, encoding, None, n_max=20)
    _assert_parity(got, expected)
    assert fast.STATS["fallback_containment_boundary"] == int(scale == 1)


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_metric_predicate_and_boundary(tables, encoding, scale):
    arrays = dict(tables.arrays)
    arrays["group_ops_R"] = arrays["group_ops_R"].astype(np.float64)
    arrays["group_ops_R"][1, 0, 0, 0] += scale * np.sqrt(3) * 0.5e-10
    output = dict(type=encoding.z_to_enc[torch.tensor([14])][None],
                  u=torch.full((1, 1, 3), -0.5, dtype=torch.float64),
                  z=torch.zeros(1, 6, dtype=torch.float64))
    skeleton = Skeleton(1, (1,))
    expected = reference.decode_batch(output, skeleton, arrays, encoding, None, n_max=20)
    fast.STATS.clear()
    got = fast.decode_batch(output, skeleton, arrays, encoding, None, n_max=20)
    _assert_parity(got, expected)
    assert fast.STATS["fallback_metric_boundary"] == int(scale == 1)
    if scale > 1:
        assert got.failures == {0: "containment_failed"}


def test_ambiguous_matching_uses_reference_bijection(tables, encoding, monkeypatch):
    monkeypatch.setattr(reference, "CHECK_A", dict(cart_tol=0.03, metric_rtol=1e-10))
    row = tables.row_id(2, "i")
    skeleton = Skeleton(2, (row, row), (6, 6))
    output = dict(type=encoding.z_to_enc[torch.tensor([6, 6])][None],
                  u=torch.tensor([[[-0.4, -0.5, -0.5], [-0.39, -0.5, -0.5]]], dtype=torch.float64),
                  z=torch.zeros(1, 6, dtype=torch.float64))
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    fast.STATS.clear()
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    _assert_parity(got, expected)
    assert got.failures == {}
    assert fast.STATS == Counter(structures=1, fallback=1, fallback_ambiguous_matching=1)


def test_nonfinite_y1_leaves_zero_rows(tables, encoding, monkeypatch):
    def nonfinite(gram):
        return torch.full((len(gram), 6), float("nan"), dtype=gram.dtype)

    monkeypatch.setattr(reference, "gram_to_y1", nonfinite)
    monkeypatch.setattr(fast, "gram_to_y1", nonfinite)
    output, skeleton = _rutile(tables, encoding)
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20, structures=True)
    _assert_parity(got, expected)
    assert got.failures == {0: "non_finite"}
    assert got.contains_a[0]
    assert got.structures == [None]


def test_content_cache_tracks_arrays_and_sha256(tables):
    fast._TABLE_CACHE.clear()
    first = fast._prepared(tables)
    same_hash = SimpleNamespace(arrays=tables.arrays, sha256=tables.sha256)
    assert fast._prepared(same_hash) is first
    arrays = dict(tables.arrays)
    original = fast._prepared(arrays)
    assert fast._prepared(dict(arrays)) is original
    arrays["group_ops_t"] = arrays["group_ops_t"].copy()
    arrays["group_ops_t"][1, 0, 0] += 0.125
    changed = fast._prepared(arrays)
    assert changed is not original
    start = changed.op_offsets[1]
    assert changed.translations[start, 0] != original.translations[start, 0]
    arrays["group_Pc"] = arrays["group_Pc"].copy()
    previous = fast._prepared(arrays)
    arrays["group_Pc"][1] *= 2
    fast._prepared(arrays)
    np.testing.assert_array_equal(previous.arrays["group_Pc"][1], np.eye(3))
    assert len(fast._TABLE_CACHE) <= 4


@pytest.mark.data
def test_chunk_split_determinism(encoding):
    fixture = _fixture("d256_s150_c60")
    chunk, expected = fixture.chunks[0], fixture.expected[0]
    indices = _indices(chunk, expected, limit=12)
    output, skeletons = _inputs(chunk, indices)
    whole = fast.decode_batch(output, skeletons, fixture.tables, encoding, fixture.allowed, n_max=chunk["n_max"])
    live = reference.decode_batch(output, skeletons, fixture.tables, encoding,
                                  fixture.allowed, n_max=chunk["n_max"])
    _assert_parity(whole, live)
    for start, end in ((0, 1), (1, 5), (5, len(indices))):
        part = fast.decode_batch({k: v[start:end] for k, v in output.items()}, skeletons[start:end],
                                 fixture.tables, encoding, fixture.allowed, n_max=chunk["n_max"])
        _assert_parity(part, _expected(dict(rows=whole.rows, failures=whole.failures,
                                          contains_a=whole.contains_a, requested_G=whole.requested_G),
                                     np.arange(start, end)))
        for key in whole.rows:
            assert torch.equal(part.rows[key], whole.rows[key][start:end])


def test_empty_batch_and_u_alias(tables, encoding):
    output, skeleton = _rutile(tables, encoding)
    output["u_c"] = output.pop("u")
    got = fast.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    expected = reference.decode_batch(output, skeleton, tables, encoding, None, n_max=20)
    _assert_parity(got, expected)
    output = {key: value[:0] for key, value in output.items()}
    _assert_parity(fast.decode_batch(output, [], tables, encoding, None, n_max=20),
                   reference.decode_batch(output, [], tables, encoding, None, n_max=20))


@pytest.mark.slow
@pytest.mark.parametrize("name", (
    "d256_s150_c60", "d512_s150_c60", "d256_s10_c60", "d256_s10_c0", "d512_s10_c0",
))
@pytest.mark.data
def test_compiled_fixture_timing(name, encoding, record_property):
    if not ENABLED:
        pytest.skip("Compiled timing requires Numba (ORBITOL_NUMBA not 0)")
    fixture = _fixture(name)
    live = _live_full(fixture, encoding)
    inputs = [_inputs(chunk) for chunk in fixture.chunks]

    def run():
        return [fast.decode_batch(output, skeletons, fixture.tables, encoding, fixture.allowed,
                                  n_max=chunk["n_max"])
                for (output, skeletons), chunk in zip(inputs, fixture.chunks, strict=True)]

    run()  # Compilation and table preparation are outside warm timings.
    count = sum(len(skeletons) for _, skeletons in inputs)
    total = []
    fast.STATS.clear()
    for _ in range(3):
        start = time.perf_counter()
        results = run()
        elapsed = time.perf_counter() - start
        total.append(elapsed * 1e6 / count)
        for got, wanted, expected in zip(results, live, fixture.expected, strict=True):
            _assert_parity(got, wanted)
            if fixture.compare_stored:
                _assert_parity(got, expected)
    result = dict(fixture=name, path=str(fixture.path), structures=count,
                  compiled_us_per_structure=total,
                  median_us_per_structure=float(np.median(total)), stats=dict(fast.STATS),
                  cpu=_cpu_model())
    record_property("decode_fast_timing", json.dumps(result, sort_keys=True))
    print(json.dumps(result, sort_keys=True))



def test_unexpected_fast_path_error_falls_back_to_reference(tables, encoding, monkeypatch):
    """A Python error inside the array pipeline decodes the whole batch with the reference."""
    output, skeleton = _rutile(tables, encoding, 3)
    expected = reference.decode_batch(output, [skeleton] * 3, tables, encoding, None, n_max=20)

    def broken(*args, **kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(fast, "_decode_arrays", broken)
    fast.STATS.clear()
    got = fast.decode_batch(output, [skeleton] * 3, tables, encoding, None, n_max=20)
    _assert_parity(got, expected)
    assert got.structures is None
    assert fast.STATS == Counter(structures=3, fallback=3, fallback_exception=3)
