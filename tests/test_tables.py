"""Fast regression checks for frozen symmetry-table construction."""
import itertools
import json

import numpy as np
import pytest
import torch

from orbitol import contracts as C
from orbitol.descriptors import descriptor_tables, raw_row_descriptor, groups_metadata, point_metric
from orbitol.tables import (
    build_tables, load_tables, wrap_exact_np, verify_tables,
    _verify_wrap_certificate, _op_codes, _coset_key, _point_keys,
)

GROUPS = [1, 19, 27, 44, 45, 80, 143, 166, 194, 225, 227, 229]


@pytest.fixture(scope="module")
def tables():
    return build_tables(GROUPS)


def test_contract_dtypes_padding_and_full_images(tables):
    a = tables.arrays
    assert set(a) == set(C.TABLE_ARRAYS)
    for name, (_, dtype, _) in C.TABLE_ARRAYS.items():
        assert a[name].dtype == np.dtype(dtype)
    r = tables.row_id(225, "l")
    assert a["row_img_count"][r] == 192
    assert len({_op_codes(S[None], s[None])[0]
                for S, s in zip(a["row_img_S"][r], a["row_img_s"][r])}) == 192
    np.testing.assert_array_equal(a["row_A"][0], np.eye(3))
    np.testing.assert_array_equal(a["row_N"][0], np.eye(3))
    assert a["row_mult"][0] == 1 and a["row_dof"][0] == 3
    assert not a["row_desc"][0].any() and not a["group_desc"][0].any()
    assert a["row_letter"][0] == "" and a["row_group"][0] == 0
    assert a["group_ops_count"][0] == 1
    r2 = int(a["group_row_start"][2])
    assert a["row_group"][r2] == 0 and a["row_letter"][r2] == ""
    np.testing.assert_array_equal(a["row_H"][r2], np.eye(3))
    assert a["group_hall"][2] == 0
    assert a["group_row_start"][230] + a["group_row_count"][230] == C.R_TOTAL
    with pytest.raises(KeyError):
        tables.row_id(2, "a")
    with pytest.raises(ValueError):
        tables.row_id(0, "a")


def test_subset_order_hash_and_descriptors(tables):
    one = build_tables([225, 1, 225])
    two = build_tables(iter([1, 225]))
    assert one.sha256 == two.sha256
    for g in (1, 225):
        start, count = one.arrays["group_row_start"][g], one.arrays["group_row_count"][g]
        for name in C.TABLE_ARRAYS:
            if name.startswith("row_"):
                np.testing.assert_array_equal(one.arrays[name][start:start + count], tables.arrays[name][start:start + count])
            else:
                np.testing.assert_array_equal(one.arrays[name][g], tables.arrays[name][g])
    desc, sg_desc, names, mean, scale = descriptor_tables()
    assert len(names) == 62 and len(set(names)) == 62
    np.testing.assert_allclose(desc[1:].mean(0, dtype=float), 0, atol=1e-7)
    np.testing.assert_allclose(sg_desc[1:].mean(0, dtype=float), 0, atol=1e-7)
    std = sg_desc[1:].std(0, dtype=float)
    np.testing.assert_allclose(std[std > 1e-7], 1, atol=1e-7)
    g = groups_metadata()[193]
    wp = next(wp for wp in g if wp.letter == "h")
    raw = raw_row_descriptor(wp, point_metric(g))
    r = tables.row_id(194, "h")
    np.testing.assert_allclose(tables.arrays["row_desc"][r], (raw - mean) / scale, atol=2e-7)


@pytest.mark.parametrize("groups", [[0], [231], [1.5], [True]])
def test_invalid_groups(groups):
    with pytest.raises(ValueError):
        build_tables(groups)


def test_empty_build():
    ts = build_tables([])
    assert not ts.arrays["group_hall"].any()
    assert not ts.arrays["row_group"].any()
    assert len(ts.cosets.group) == 0 and ts.cosets.map_start.tolist() == [0]


def test_torch_conversion(tables):
    sha256 = tables.sha256
    tensors = tables.torch("cpu", dtype=torch.float64)
    assert tensors.row_desc.dtype == torch.float64
    assert tensors.row_group.dtype == torch.int16
    assert tensors.row_dof.dtype == torch.int8
    assert tensors.row_member_mask.dtype == torch.bool
    assert not hasattr(tensors, "row_letter") and not hasattr(tensors, "group_symbol")
    assert tables.torch("cpu").row_H.dtype == torch.float32
    assert tensors.row_Hinv_sqrt.dtype == torch.float64
    diagonal = torch.diag_embed(tensors.row_H.diagonal(dim1=-2, dim2=-1))
    is_diagonal = (tensors.row_H == diagonal).all(dim=-1).all(dim=-1)
    expected = torch.diag_embed(1 / tensors.row_H[is_diagonal].diagonal(dim1=-2, dim2=-1).sqrt())
    assert torch.equal(tensors.row_Hinv_sqrt[is_diagonal], expected)
    identity = torch.eye(3, dtype=torch.float64).expand_as(tensors.row_H)
    torch.testing.assert_close(tensors.row_Hinv_sqrt @ tensors.row_H @ tensors.row_Hinv_sqrt,
                               identity, atol=1e-14, rtol=0)
    assert "row_Hinv_sqrt" not in tables.arrays
    assert tables.sha256 == sha256


def _certified_brute(delta, B, H):
    """Search an ellipsoid-bounded coefficient box around a feasible point."""
    center = np.rint(np.linalg.solve(B, delta))
    initial = delta - B @ center
    upper = initial @ H @ initial
    Kinv = np.linalg.inv(B.T @ H @ B)
    # In coefficient coordinates, every solution improving the feasible point
    # lies this far from B^-1 delta. Rounding outwards includes boundary ties.
    real_center = np.linalg.solve(B, delta)
    bounds = np.sqrt(max(upper, 0) * np.diag(Kinv)) + 1e-10
    ranges = [range(int(np.ceil(c - b)), int(np.floor(c + b)) + 1) for c, b in zip(real_center, bounds)]
    coeff = np.array(list(itertools.product(*ranges)))
    errors = delta - coeff @ B.T
    return np.einsum("ni,ij,nj->n", errors, H, errors).min()


def test_wrap_certificates_and_independent_closest_vectors(tables):
    a = tables.arrays
    rng = np.random.default_rng(11)
    rows = [tables.row_id(g, letter) for g, letter in
            [(1, "a"), (27, "a"), (80, "b"), (166, "h"), (194, "l"), (225, "l"), (229, "f")]]
    for r in rows:
        d = int(a["row_dof"][r])
        B, H = a["row_lambda_basis"][r, :d, :d], a["row_H"][r, :d, :d]
        v = a["row_vrv"][r, :a["row_vrv_count"][r], :d]
        _verify_wrap_certificate(B, H, v)
        delta = rng.uniform(-20, 20, (20, 3))
        wrapped = wrap_exact_np(delta, r, tables)
        assert not wrapped[:, d:].any()
        for x, e in zip(delta[:, :d], wrapped[:, :d]):
            assert abs(e @ H @ e - _certified_brute(x, B, H)) < 1e-9
        lattice = (B @ rng.integers(-20, 20, (d, 20))).T
        shifted = delta.copy()
        shifted[:, :d] += lattice
        np.testing.assert_allclose(wrap_exact_np(shifted, r, tables), wrapped, atol=1e-12)
    # Exact boundary ties can choose either minimiser; both norms must be correct.
    delta = np.array([[.5, -.5, 1.5]])
    np.testing.assert_array_equal(wrap_exact_np(delta, 1, tables), delta - np.rint(delta))
    fixed = tables.row_id(225, "a")
    np.testing.assert_array_equal(wrap_exact_np(delta, fixed, tables), np.zeros((1, 3)))
    assert wrap_exact_np(np.empty((0, 3)), 1, tables).shape == (0, 3)


def test_rhombohedral_wrap_requires_search_centered_at_input(tables):
    a = tables.arrays
    r = tables.row_id(166, "h")
    rng = np.random.default_rng(2)
    for _ in range(3):
        rng.uniform(-3, 3, (24, 1))
    delta = np.zeros((24, 3))
    delta[:, :2] = rng.uniform(-3, 3, (24, 2))
    B, H = a["row_lambda_basis"][r, :2, :2], a["row_H"][r, :2, :2]
    e = wrap_exact_np(delta, r, tables)[2, :2]
    coeff = np.linalg.solve(B, delta[2, :2] - e)
    assert np.abs(coeff).max() > 6
    assert abs(e @ H @ e - _certified_brute(delta[2, :2], B, H)) < 1e-12


def test_normaliser_repairs_and_maps(tables):
    a, cs = tables.arrays, tables.cosets
    assert cs.group_count[1] == 0
    assert cs.group_count[19] == 96
    rng = np.random.default_rng(32)
    for g in (19, 44, 45, 80, 143):
        k = int(a["group_ops_count"][g])
        R, t = a["group_ops_R"][g, :k], a["group_ops_t"][g, :k]
        polar = np.flatnonzero(a["group_polar"][g]).tolist()
        start, count = int(cs.group_start[g]), int(cs.group_count[g])
        inds = range(start, start + count)
        keys = {_coset_key(cs.P[j], cs.p[j], R, t, polar) for j in inds}
        assert len(keys) == count
        for j in inds:
            for l in inds:
                assert _coset_key(cs.P[j] @ cs.P[l], cs.P[j] @ cs.p[l] + cs.p[j], R, t, polar) in keys
            for h in range(cs.map_start[j], cs.map_start[j + 1]):
                src, dst = cs.source_row[h], cs.target_row[h]
                u = rng.uniform(.1, .9, 3)
                u[int(a["row_dof"][src]):] = 0
                uu = cs.S[h] @ u + cs.s[h]
                m = int(a["row_mult"][src])
                source = a["row_W"][src, :m] @ u + a["row_v"][src, :m]
                target = a["row_W"][dst, :m] @ uu + a["row_v"][dst, :m]
                assert _point_keys(source @ cs.P[j].T + cs.p[j]) == _point_keys(target)
    # These two non-Sohncke groups have proper representatives in every coset.
    for g in (44, 45):
        assert cs.proper[cs.group == g].all()
    assert cs.reexpress_lattice[cs.group == 19].any()


def test_persistence_hash_covers_cosets_and_tables(tables, tmp_path):
    tables.save(tmp_path)
    before = json.loads((tmp_path / "manifest.json").read_text())
    back = load_tables(tmp_path)
    assert back.sha256 == tables.sha256
    for name, arr in tables.cosets.arrays.items():
        np.testing.assert_array_equal(getattr(back.cosets, name), arr)
    assert back.build_date == before["build_date"]
    back.cosets.p[0, 0] += .125
    np.savez_compressed(tmp_path / "cosets.npz", **back.cosets.arrays)
    with pytest.raises(ValueError, match="sha256"):
        load_tables(tmp_path)
    np.savez_compressed(tmp_path / "cosets.npz", **tables.cosets.arrays)
    back.arrays["row_b"][1, 0] += .125
    np.savez_compressed(tmp_path / "tables.npz", **back.arrays)
    with pytest.raises(ValueError, match="sha256"):
        load_tables(tmp_path)


def test_verifier_on_small_subset():
    subset = build_tables([1, 27, 80, 143])
    with pytest.raises(ValueError, match="requires all 230"):
        verify_tables(subset)
    result = verify_tables(subset, require_all=False)
    assert result["status"] == "passed" and result["groups"] == 4
    subset.arrays["row_Pi"][1, 0, 0] = 2
    with pytest.raises(ValueError, match="Projection"):
        verify_tables(subset, require_all=False)
