"""Frozen acceptance tests for orbitol.tables."""
import itertools

import numpy as np
import pytest

from orbitol import contracts as C
from orbitol.tables import build_tables, load_tables, wrap_exact_np

GROUPS = [1, 2, 27, 47, 136, 166, 194, 221, 225, 227, 229]


@pytest.fixture(scope="module")
def ts():
    return build_tables(groups=GROUPS)


def _rows(a, g):
    s, c = int(a["group_row_start"][g]), int(a["group_row_count"][g])
    return range(s, s + c)


def test_layout_and_row_pins(ts):
    a = ts.arrays
    for name, (shape, _dtype, _doc) in C.TABLE_ARRAYS.items():
        assert name in a, name
        want = tuple(C.R_TOTAL if s == "R" else C.G_TOTAL if s == "G" else getattr(C, s) if isinstance(s, str) else s
                     for s in shape)
        assert a[name].shape == want, (name, a[name].shape, want)
    assert ts.row_id(1, "a") == 1
    assert ts.row_id(2, "a") == 2
    assert ts.row_id(47, "a") == 226
    assert ts.row_id(47, "A") == 252
    assert int(a["group_row_start"][1]) == 1
    assert int(a["group_row_start"][230]) + int(a["group_row_count"][230]) - 1 == C.N_ROWS


def test_rutile_rows(ts):
    a = ts.arrays
    r2a, r4f = ts.row_id(136, "a"), ts.row_id(136, "f")
    assert (int(a["row_mult"][r2a]), int(a["row_dof"][r2a])) == (2, 0)
    assert (int(a["row_mult"][r4f]), int(a["row_dof"][r4f])) == (4, 1)
    np.testing.assert_allclose(a["row_A"][r4f][:, 0], [1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(a["row_A"][r4f][:, 1:], 0, atol=0)
    np.testing.assert_allclose(a["row_H"][r4f][0, 0], 2.0, atol=1e-12)
    assert bool(a["row_wrap_fast"][r4f])


def test_hexagonal_metric_and_projector(ts):
    a = ts.arrays
    mg = np.array([[4 / 3, -2 / 3, 0], [-2 / 3, 4 / 3, 0], [0, 0, 1]])
    for g in (166, 194):
        np.testing.assert_allclose(a["group_MG"][g], mg, atol=1e-12)
    for g in (1, 47, 136, 221):
        np.testing.assert_allclose(a["group_MG"][g], np.eye(3), atol=1e-12)
    r = ts.row_id(194, "h")  # 6h (x, 2x, 1/4)
    np.testing.assert_allclose(a["row_A"][r][:, 0], [1, 2, 0], atol=1e-12)
    np.testing.assert_allclose(a["row_b"][r], [0, 0, 0.25], atol=1e-12)
    np.testing.assert_allclose(a["row_H"][r][0, 0], 4.0, atol=1e-12)
    np.testing.assert_allclose(a["row_Pi"][r][0], [0, 0.5, 0], atol=1e-12)


def _stabiliser_average(a, g, x):
    n = int(a["group_ops_count"][g])
    acc, cnt = np.zeros((3, 3)), 0
    for i in range(n):
        rot = a["group_ops_R"][g, i].astype(float)
        dlt = rot @ x + a["group_ops_t"][g, i] - x
        if np.allclose(dlt - np.round(dlt), 0, atol=1e-8):
            acc += rot
            cnt += 1
    return acc / cnt, cnt


def test_projector_equals_stabiliser_average(ts):
    a, rng = ts.arrays, np.random.default_rng(0)
    for g in GROUPS:
        for r in _rows(a, g):
            d = int(a["row_dof"][r])
            u = np.zeros(3)
            u[:d] = rng.uniform(0.1, 0.9, size=d)
            x = a["row_A"][r] @ u + a["row_b"][r]
            avg, cnt = _stabiliser_average(a, g, x)
            assert cnt * int(a["row_mult"][r]) == int(a["group_ops_count"][g]), (g, r)
            if d:
                np.testing.assert_allclose(a["row_A"][r] @ a["row_Pi"][r], avg, atol=1e-9, err_msg=f"group {g} row {r}")


def _key(v):
    w = v - np.floor(v + 1e-9)
    return {tuple(k) for k in np.round(w * 1e6).astype(np.int64)}


def test_members_are_the_orbit(ts):
    a, rng = ts.arrays, np.random.default_rng(1)
    for g in GROUPS:
        n = int(a["group_ops_count"][g])
        for r in _rows(a, g):
            d, m = int(a["row_dof"][r]), int(a["row_mult"][r])
            u = np.zeros(3)
            u[:d] = rng.uniform(0.1, 0.9, size=d)
            mem = np.einsum("jab,b->ja", a["row_W"][r, :m], u) + a["row_v"][r, :m]
            x0 = a["row_A"][r] @ u + a["row_b"][r]
            orbit = np.einsum("iab,b->ia", a["group_ops_R"][g, :n].astype(float), x0) + a["group_ops_t"][g, :n]
            assert len(_key(mem)) == m, (g, r)
            assert _key(mem) == _key(orbit), (g, r)
            dd = mem[0] - x0
            np.testing.assert_allclose(dd - np.round(dd), 0, atol=1e-9)
            assert bool(a["row_member_mask"][r, :m].all()) and not bool(a["row_member_mask"][r, m:].any())
            assert int(a["row_prim_mask"][r, :m].sum()) == m // int(a["group_ncent"][g])


def test_noise_map_and_chart_metric(ts):
    a = ts.arrays
    for g in GROUPS:
        for r in _rows(a, g):
            d = int(a["row_dof"][r])
            if d == 0:
                assert not np.any(a["row_N"][r]) and not np.any(a["row_Pi"][r])
                continue
            A = a["row_A"][r][:, :d]
            H = A.T @ a["group_MG"][g] @ A
            np.testing.assert_allclose(a["row_H"][r][:d, :d], H, atol=1e-12)
            N = a["row_N"][r][:d]
            np.testing.assert_allclose(N @ N.T, np.linalg.inv(H), atol=1e-12)


def _brute(delta, basis, H, radius=6):
    # Enumerate coefficient offsets around the Babai point round(B^-1 delta), not around the origin:
    # skewed bases (R-3m 18h) need coefficients beyond any fixed origin-centred box.
    d = basis.shape[0]
    c0 = np.round(np.linalg.solve(basis, delta))
    best = None
    for c in itertools.product(range(-radius, radius + 1), repeat=d):
        e = delta - basis @ (c0 + np.array(c, float))
        v = e @ H @ e
        if best is None or v < best[0] - 1e-12:
            best = (v, e)
    return best


def test_exact_wrap(ts):
    a, rng = ts.arrays, np.random.default_rng(2)
    rows = [ts.row_id(136, "f"), ts.row_id(27, "a"), ts.row_id(229, "f"), ts.row_id(166, "h"), ts.row_id(194, "l")]
    for r in rows:
        d = int(a["row_dof"][r])
        basis, H = a["row_lambda_basis"][r][:d, :d], a["row_H"][r][:d, :d]
        delta = np.zeros((24, 3))
        delta[:, :d] = rng.uniform(-3, 3, size=(24, d))
        got = wrap_exact_np(delta, r, ts)
        for i in range(24):
            best = _brute(delta[i, :d], basis, H, radius=6 if d < 3 else 5)
            e = got[i, :d]
            assert abs(e @ H @ e - best[0]) < 1e-9, (r, i)
            coeff = np.linalg.solve(basis, delta[i, :d] - e)
            np.testing.assert_allclose(coeff, np.round(coeff), atol=1e-9)
    # pure-translation chart images are periods
    for (g, letter, t) in ((27, "a", [0.5]), (229, "f", [0.5]), (166, "h", [2 / 3, 1 / 3])):
        r = ts.row_id(g, letter)
        dlt = np.zeros((1, 3))
        dlt[0, : len(t)] = t
        np.testing.assert_allclose(wrap_exact_np(dlt, r, ts), 0, atol=1e-9)
        assert not bool(a["row_wrap_fast"][r])


def test_centring_families_and_origin_shift(ts):
    a = ts.arrays
    assert [int(a["group_ncent"][g]) for g in (136, 166, 225, 229)] == [1, 3, 4, 2]
    fam = {g: C.FAMILIES[int(a["group_family"][g])] for g in GROUPS}
    assert fam[1] == fam[2] == "triclinic"
    assert fam[27] == fam[47] == "orthorhombic"
    assert fam[136] == "tetragonal"
    assert fam[166] == fam[194] == "hexagonal"
    assert fam[221] == fam[225] == fam[227] == fam[229] == "cubic"
    np.testing.assert_allclose(a["group_q"][227], [1 / 8, 1 / 8, 1 / 8], atol=1e-12)
    np.testing.assert_allclose(a["group_q"][225], 0, atol=1e-12)


def test_descriptors_finite(ts):
    a = ts.arrays
    for k in ("row_desc", "group_desc"):
        assert a[k].dtype == np.float32
        assert np.isfinite(a[k]).all()


def test_save_load_roundtrip(ts, tmp_path):
    ts.save(tmp_path)
    back = load_tables(tmp_path)
    for k in C.TABLE_ARRAYS:
        np.testing.assert_array_equal(back.arrays[k], ts.arrays[k])
    assert back.sha256 == ts.sha256
