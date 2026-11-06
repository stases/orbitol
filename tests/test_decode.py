"""Primitive emission, containment controls and Crystalite row parity."""
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import orbitol.core  # noqa: F401
from orbitol.contracts import Skeleton, TABLES_DIR, TYPE_ENCODING_STATE, TrainConfig
from orbitol.decode import (
    CHECK_A, CHECK_B, Conventional, _expand_members, contains_ops, decode_batch,
    decode_types, emit_primitive, expand_conventional, reduce_rows, to_crystalite_rows,
)
from orbitol.tables import load_tables
from src.crystalite.sampler import clamp_lattice_latent
from pymatgen.analysis.structure_matcher import StructureMatcher
from src.data.mp20_tokens import tokens_to_eval_dict, tokens_to_structure
from src.eval.crystal import array_dict_to_crystal
from src.models.lattice_repr import gram_to_y1, y1_to_lattice_latent
from src.models.type_encoding_state import restore_type_encoding


@pytest.fixture(scope="module")
def tables():
    # Read the frozen artifact; never build or write shared tables in this test.
    if not Path(TABLES_DIR, "tables.npz").is_file():
        pytest.skip(f"Frozen tables unavailable: {TABLES_DIR}")
    return load_tables(TABLES_DIR)


@pytest.fixture(scope="module")
def encoding():
    path = Path(__file__).resolve().parents[1] / TYPE_ENCODING_STATE
    return restore_type_encoding(torch.load(path, map_location="cpu", weights_only=False))


def case(tables, name):
    """Hand-built chart tokens plus an independent physical lattice."""
    if name == "rutile":
        G, letters, species, U = 136, "af", [22, 8], [[0, 0, 0], [.3049, 0, 0]]
        lattice = Lattice.tetragonal(4.594, 2.959)
        z = [np.log(4.594), 0, 0, 0, 0, np.log(2.959)]
    elif name == "spinel":
        G, letters, species, U = 227, "ade", [12, 13, 8], [[0, 0, 0], [0, 0, 0], [.26, 0, 0]]
        lattice, z = Lattice.cubic(8.08), [np.log(8.08), 0, 0, 0, 0, 0]
    elif name == "body_centred":
        G, letters, species, U = 139, "ae", [38, 8], [[0, 0, 0], [.23, 0, 0]]
        lattice = Lattice.tetragonal(3.9, 12.4)
        z = [np.log(3.9), 0, 0, 0, 0, np.log(12.4)]
    elif name == "long_body_centred":
        # The primitive basis has ltri off-diagonals above the core's 10 Angstrom clamp.
        G, letters, species, U = 139, "ae", [38, 8], [[0, 0, 0], [.23, 0, 0]]
        lattice = Lattice.tetragonal(3.9, 30.0)
        z = [np.log(3.9), 0, 0, 0, 0, np.log(30.0)]
    elif name == "rhombohedral":
        G, letters, species, U = 166, "acc", [52, 83, 52], [[0, 0, 0], [.4, 0, 0], [.21, 0, 0]]
        lattice = Lattice.hexagonal(4.386, 30.497)
        z = [np.log(4.386), 0, 0, 0, 0, np.log(30.497)]
    else:
        raise AssertionError(name)
    rows = np.array([tables.row_id(G, letter) for letter in letters])
    return G, rows, np.array(species), np.array(U) - .5, np.array(z), lattice


CASES = ["rutile", "spinel", "body_centred", "long_body_centred", "rhombohedral"]


def emitted(tables, name):
    # Exercise the actual expansion/emission implementation independently of
    # the lattice charts, using an explicit physical test lattice.
    G, rows, A0, u, _, lattice = case(tables, name)
    conv = _expand_members(G, rows, A0, u, lattice.matrix, tables)
    assert conv.failure is None
    return conv, emit_primitive(conv, G, tables)


@pytest.mark.parametrize("name", CASES)
def test_handbuilt_tokens(tables, name):
    G, rows, A0, u, z, lattice = case(tables, name)
    conv = expand_conventional(G, rows, A0, u, z, tables)
    assert conv.failure is None
    # The chart uses a different rigid orientation from pymatgen's constructor.
    np.testing.assert_allclose(conv.lattice @ conv.lattice.T, lattice.metric_tensor, atol=1e-11)
    structure = emit_primitive(conv, G, tables)
    assert contains_ops(structure, G, tables, transform=tables.arrays["group_Pc"][G], **CHECK_A)


@pytest.mark.parametrize("name,count", list(zip(CASES, [6, 14, 3, 3, 5])))
def test_exact_members_primitive_and_containment(tables, name, count):
    G, rows, A0, _, _, _ = case(tables, name)
    conv, structure = emitted(tables, name)
    assert len(structure) == count == tables.arrays["row_mprim"][rows].sum()
    assert len(conv.frac) == tables.arrays["row_mult"][rows].sum()
    assert conv.frac.dtype == conv.lattice.dtype == structure.frac_coords.dtype == np.float64
    assert ((conv.frac >= 0) & (conv.frac < 1)).all()
    P = tables.arrays["group_Pc"][G]
    np.testing.assert_allclose(structure.lattice.matrix, P.T @ conv.lattice, atol=1e-14)
    np.testing.assert_allclose(structure.frac_coords, conv.frac[conv.prim_mask] @ np.linalg.inv(P).T % 1, atol=1e-14)
    np.testing.assert_array_equal(structure.atomic_numbers, conv.species[conv.prim_mask])
    assert contains_ops(structure, G, tables, transform=P, **CHECK_A)
    if name == "spinel":
        assert tables.arrays["group_hall"][G] == 526  # frozen origin choice 2
        assert structure.composition.as_dict() == {"Mg": 2, "Al": 4, "O": 8}


def test_rutile_known_positions_and_negative_controls(tables):
    conv, structure = emitted(tables, "rutile")
    x = .3049
    known = {22: [[0, 0, 0], [.5, .5, .5]],
             8: [[x, x, 0], [1-x, 1-x, 0], [.5+x, .5-x, .5], [.5-x, .5+x, .5]]}
    for species, coords in known.items():
        actual = conv.frac[conv.species == species]
        assert len(actual) == len(coords)
        distances = structure.lattice.get_all_distances(actual, coords)
        assert (distances.min(axis=0) < 1e-12).all()
    assert contains_ops(structure, 136, tables, **CHECK_A)
    assert not contains_ops(structure, 123, tables, **CHECK_A)
    displaced = structure.copy()
    displaced.translate_sites([2], [.01, 0, 0], frac_coords=False)
    assert not contains_ops(displaced, 136, tables, **CHECK_A)
    wrong_species = structure.copy()
    wrong_species.replace(2, "N")
    assert not contains_ops(wrong_species, 136, tables, **CHECK_A)
    strained_L = structure.lattice.matrix.copy()
    strained_L[0] *= 1.001
    strained = Structure(strained_L, structure.species, structure.frac_coords)
    assert not contains_ops(strained, 136, tables, **CHECK_A)
    # Disabling only the metric threshold makes this sites-only check pass.
    assert contains_ops(strained, 136, tables, cart_tol=1e-6, metric_rtol=1)


def test_bijection_is_required_and_ambiguous_matches_can_pass(tables):
    # Every inverted site finds a neighbour, but two compete for a single site.
    failed = Structure(np.eye(3), [6]*3, [[.10, 0, 0], [.11, 0, 0], [.90, 0, 0]])
    assert not contains_ops(failed, 2, tables, cart_tol=.03, metric_rtol=1e-10)
    valid = Structure(np.eye(3), [6]*4, [[.10, 0, 0], [.11, 0, 0], [.90, 0, 0], [.89, 0, 0]])
    assert contains_ops(valid, 2, tables, cart_tol=.03, metric_rtol=1e-10)


def test_periodic_matching_in_a_skew_cell(tables):
    lattice = Lattice([[1, 0, 0], [10, .1, 0], [0, 0, 1]])
    # Inversion's nearest image needs a shift larger than one in the original
    # basis. Componentwise wrapping would give a distance above 4 Angstrom.
    structure = Structure(lattice, [6], [[.2, .23, 0]])
    assert contains_ops(structure, 2, tables, cart_tol=.05, metric_rtol=1e-10)


@pytest.mark.parametrize("name", CASES)
def test_crystalite_rows_round_trip(tables, name):
    G = case(tables, name)[0]
    _, structure = emitted(tables, name)
    rows = to_crystalite_rows([structure], n_max=20)
    assert set(rows) == {"A0", "F1", "Y1", "pad_mask"}
    assert rows.failures == {}
    assert rows["A0"].dtype == torch.int64
    assert rows["F1"].dtype == rows["Y1"].dtype == torch.float32
    assert rows["pad_mask"].dtype == torch.bool
    assert ((rows["F1"] >= 0) & (rows["F1"] < 1)).all()
    assert rows["Y1"][:, :3].abs().max() <= 5
    assert rows["Y1"][:, 3:].abs().max() <= .999
    assert (~rows["pad_mask"]).sum() == len(structure)
    assert torch.count_nonzero(rows["A0"][rows["pad_mask"]]) == 0
    assert torch.count_nonzero(rows["F1"][rows["pad_mask"]]) == 0
    rebuilt = tokens_to_structure({key: value[0] for key, value in rows.items()})
    # Rows carry the Niggli cell of the emitted crystal; M maps the emitted basis to it.
    reduced = structure.get_reduced_structure()
    M = np.rint(reduced.lattice.matrix @ np.linalg.inv(structure.lattice.matrix))
    assert abs(abs(np.linalg.det(M)) - 1) < 1e-9
    np.testing.assert_allclose(rebuilt.lattice.metric_tensor, reduced.lattice.metric_tensor, rtol=1e-6, atol=2e-5)
    assert contains_ops(rebuilt, G, tables, transform=tables.arrays["group_Pc"][G] @ M.T, **CHECK_B)
    assert all(50 < angle < 130 for angle in rebuilt.lattice.angles)


def test_rows_use_niggli_cells_that_the_core_evaluator_accepts(tables):
    # Regressions: the core's ltri latent clamp once distorted such emitted cells, and
    # the core's array_dict_to_crystal dummies any cell with an angle outside (50, 130).
    _, structure = emitted(tables, "long_body_centred")
    L = torch.as_tensor(structure.lattice.matrix[None])
    latent = y1_to_lattice_latent(gram_to_y1(L @ L.transpose(-1, -2)), "ltri")
    assert latent[0, [1, 3, 4]].abs().max() > 10
    assert not all(50 < angle < 130 for angle in structure.lattice.angles)
    rows = to_crystalite_rows([structure], n_max=20)
    item = {key: value[0] for key, value in rows.items()}
    assert array_dict_to_crystal(tokens_to_eval_dict(item)).constructed
    rebuilt = tokens_to_structure(item)
    assert StructureMatcher(scale=False, primitive_cell=False).fit(rebuilt, structure)


def test_rows_carry_the_core_y1_of_the_niggli_cell():
    lattice = [[4, 0, 0], [1.1, 5, 0], [-.7, 2, 6]]
    structure = Structure(lattice, [14], [[-.1, 1 + 1e-8, 1 - 1e-10]])
    rows = to_crystalite_rows([structure], 3)
    L = torch.as_tensor(structure.get_reduced_structure().lattice.matrix[None])
    expected = clamp_lattice_latent(gram_to_y1(L @ L.transpose(-1, -2)), "y1").float()
    torch.testing.assert_close(rows["Y1"], expected, atol=0, rtol=0)
    assert ((rows["F1"] >= 0) & (rows["F1"] < 1)).all()


def test_reduce_rows_rewrites_old_pools_without_changing_crystals(tables):
    # Pools committed before Niggli emission stored the emitted primitive basis.
    _, structure = emitted(tables, "long_body_centred")
    n, L = len(structure), torch.as_tensor(structure.lattice.matrix[None])
    old = dict(A0=torch.zeros(2, 20, dtype=torch.int64), F1=torch.zeros(2, 20, 3),
               Y1=torch.zeros(2, 6), pad_mask=torch.ones(2, 20, dtype=torch.bool))
    old["A0"][0, :n] = torch.as_tensor(structure.atomic_numbers)
    old["F1"][0, :n] = torch.as_tensor(structure.frac_coords % 1.0, dtype=torch.float32)
    old["Y1"][0] = clamp_lattice_latent(gram_to_y1(L @ L.transpose(-1, -2)), "y1").float()[0]
    old["pad_mask"][0, :n] = False  # row 1 stays a failed decode
    new = reduce_rows(old)
    assert bool(new["pad_mask"][1].all())
    matcher = StructureMatcher(scale=False, primitive_cell=False)
    for rows in (new, reduce_rows(new)):
        item = {key: value[0] for key, value in rows.items()}
        assert array_dict_to_crystal(tokens_to_eval_dict(item)).constructed
        assert matcher.fit(tokens_to_structure(item), structure)


def test_decode_types_matches_frozen_encoding(encoding):
    gen = torch.Generator().manual_seed(27)
    feats = torch.randn(2, 5, 16, generator=gen)
    pad = torch.tensor([[False]*4 + [True], [False]*2 + [True]*3])
    allowed = torch.zeros(encoding.vz, dtype=torch.bool)
    allowed[[7, 21]] = True
    actual = decode_types(feats, pad, encoding, allowed)
    expected = encoding.decode_logits_to_A0(feats, pad, allowed)
    assert torch.equal(actual, expected)
    assert ((actual[~pad] == 8) | (actual[~pad] == 22)).all()
    assert not actual[pad].any()
    # Freeze the core's surprising all-false-mask policy too: it allows all.
    assert torch.equal(decode_types(feats, pad, encoding, torch.zeros_like(allowed)),
                       encoding.decode_logits_to_A0(feats, pad, None))


@pytest.mark.parametrize("kind", ["non_finite", "reused_fixed_row", "invalid_rows", "invalid_species"])
def test_bad_tokens_return_reasons_before_lattice(tables, kind):
    G, rows, A0, u, z, _ = case(tables, "rutile")
    if kind == "non_finite":
        z[0] = np.nan
    elif kind == "reused_fixed_row":
        rows[1] = rows[0]
    elif kind == "invalid_rows":
        rows[0] = -1
    else:
        A0[0] = 0
    conv = expand_conventional(G, rows, A0, u, z, tables)
    assert conv.failure == kind
    assert emit_primitive(conv, G, tables) is None
    out = to_crystalite_rows([conv], 20)
    assert out.failures == {0: kind}
    assert out["pad_mask"].all()


def test_coinciding_orbits_including_periodic_boundary(tables):
    G, rows, A0, u, _, lattice = case(tables, "rutile")
    u[1, 0] = -.5 + 1e-5
    assert _expand_members(G, rows, A0, u, lattice.matrix, tables).failure == "coinciding_orbits"
    u[1, 0] = .5 - 1e-5
    assert _expand_members(G, rows, A0, u, lattice.matrix, tables).failure == "coinciding_orbits"
    u[1, 0] = -.5 + 1e-3
    assert _expand_members(G, rows, A0, u, lattice.matrix, tables).failure is None
    bad_lattice = lattice.matrix.copy()
    bad_lattice[0, 0] = np.inf
    assert _expand_members(G, rows, A0, u, bad_lattice, tables).failure == "non_finite"


def test_failed_and_oversize_rows(tables):
    _, structure = emitted(tables, "rutile")
    rows = to_crystalite_rows([structure, None, Conventional(failure="coinciding_orbits")], 5)
    assert rows.failures == {0: "too_many_atoms", 1: "failed_decode", 2: "coinciding_orbits"}
    assert rows["pad_mask"].all()
    assert not rows["A0"].any() and not rows["F1"].any() and not rows["Y1"].any()
    assert to_crystalite_rows([], 20)["Y1"].shape == (0, 6)
    assert not contains_ops(None, 136, tables, **CHECK_A)


def sampler_batch(tables, encoding, n=3):
    G, rows, A0, u, z, _ = case(tables, "rutile")
    feats = encoding.z_to_enc[torch.tensor(A0)].repeat(n, 1, 1)
    out = {"type": feats, "u_c": torch.tensor(u).repeat(n, 1, 1), "z": torch.tensor(z).repeat(n, 1)}
    return out, Skeleton(G, tuple(rows))


@pytest.mark.parametrize("key,index", [("D_type", (1, 0, 0)), ("D_u", (1, 1, 0)), ("D_z", (1, 0))])
def test_non_finite_sampler_outputs_keep_pool_slots(tables, encoding, key, index):
    from orbitol import sample
    from orbitol.sampler import orbit_edm_sampler
    from src.utils.ema import EMA

    targets, skeleton = sampler_batch(tables, encoding)
    poison = False

    def oracle(state, sigma, skel, tb, *, objective, sigma_data):
        den = {"D_type": targets["type"].to(state["type_noisy"]).clone(),
               "D_u": targets["u_c"].to(state["u_noisy"]).clone(),
               "D_z": targets["z"].to(state["z_noisy"]).clone()}
        if poison:
            den[key][index] = float("nan")
        return den

    def sampler(*args, **kwargs):
        return orbit_edm_sampler(*args, **kwargs, denoise_fn=oracle)

    model = torch.nn.Linear(1, 1)
    ema = EMA(model, .9)
    cfg = TrainConfig(K=2, sample_num_steps=2, S_churn=0., sample_token_budget=0)
    bank = SimpleNamespace(sample=lambda n, **kwargs: [skeleton] * n)
    tb = tables.torch("cpu")

    def pool():
        return sample.sample_pool(model, ema, cfg, tables, tb, encoding, None, bank,
                                  n=3, seed=7, device="cpu", chunk=3, sampler=sampler)

    clean = pool()
    assert clean["failures"] == {}
    # Old pools without the flag still decode exactly as before.
    legacy = decode_batch({k: v for k, v in clean["raw"].items() if k != "finite"},
                          skeleton, tables, encoding, None, n_max=20)
    assert legacy.failures == {}
    for name in clean["rows"]:
        torch.testing.assert_close(legacy.rows[name], clean["rows"][name], atol=0, rtol=0)

    poison = True
    out = pool()
    assert out["raw"]["finite"].tolist() == [True, False, True]
    assert all(torch.isfinite(out["raw"][name]).all() for name in ("type", "u", "z"))
    assert out["failures"] == {1: "non_finite"}
    assert out["failure_counts"] == {"non_finite": 1}
    assert out["contains_a"].tolist() == [True, False, True]
    assert out["rows"]["pad_mask"][1].all()
    assert len(out["skeletons"]) == len(out["requested_G"]) == out["metadata"]["n"] == 3
    valid_count = (~out["rows"]["pad_mask"].all(dim=1)).sum().item()
    assert valid_count + sum(out["failure_counts"].values()) == 3
    for name in ("type", "u", "z"):
        torch.testing.assert_close(out["raw"][name][[0, 2]], clean["raw"][name][[0, 2]], atol=0, rtol=0)
    for name in out["rows"]:
        assert len(out["rows"][name]) == 3
        torch.testing.assert_close(out["rows"][name][[0, 2]], clean["rows"][name][[0, 2]], atol=0, rtol=0)
    fixed = Skeleton(skeleton.G, skeleton.rows, (22, 8))
    assert decode_batch(out["raw"], fixed, tables, encoding, None, n_max=20).failures == {1: "non_finite"}


@pytest.mark.parametrize("single_orbit", [True, False])
def test_collapsed_orbit_and_nearby_generic_chart(tables, encoding, single_orbit):
    out, skeleton = sampler_batch(tables, encoding, 2)
    # SG 136, 4f (row 1004): x=.5 collapses pairs; x=.501 keeps all four members.
    assert tables.arrays["row_dof"][skeleton.rows[1]] == 1
    out["u_c"][:, 1, 0] = torch.tensor([0., .001], dtype=torch.float64)
    if single_orbit:
        out["type"], out["u_c"] = out["type"][:, 1:], out["u_c"][:, 1:]
        skeleton = Skeleton(skeleton.G, (skeleton.rows[1],))
    result = decode_batch(out, skeleton, tables, encoding, None, n_max=20)
    assert result.failures == {0: "collapsed_orbit"}
    assert result.structures[0] is None
    assert result.rows["pad_mask"][0].all()
    assert result.contains_a.tolist() == [False, True]
    assert result.structures[1].composition.as_dict() == ({"O": 4} if single_orbit else {"Ti": 2, "O": 4})


def test_batch_invalid_samples_do_not_require_lattice(tables, encoding):
    out, skeleton = sampler_batch(tables, encoding)
    out["type"][0, 0, 0] = float("nan")
    out["u_c"][1, 0, 0] = float("inf")
    out["z"][2, 5] = float("nan")
    result = decode_batch(out, skeleton, tables, encoding, None, n_max=20)
    assert result.failures == {0: "non_finite", 1: "non_finite", 2: "non_finite"}
    assert result.structures == [None]*3
    assert not result.contains_a.any()
    assert result.rows["pad_mask"].all()


def test_batch_mixed_failures_and_padding(tables, encoding):
    out, skel = sampler_batch(tables, encoding, 5)
    skeletons = [skel, skel, Skeleton(136, (skel.rows[0],)*2), skel, Skeleton(136, (skel.rows[0],), (14,))]
    out["u_c"][1, 1, 0] = -.5
    out["z"][3, 0] = float("nan")
    out["type"][4] = float("nan")  # fixed species overrides type features
    out["u_c"][4, 1] = float("nan")  # padding is ignored
    result = decode_batch(out, skeletons, tables, encoding, None, n_max=20)
    assert result.failures == {1: "coinciding_orbits", 2: "reused_fixed_row", 3: "non_finite"}
    assert result.contains_a.tolist() == [True, False, False, False, True]
    assert result.requested_G.tolist() == [136]*5
    assert result.structures[4].atomic_numbers == (14, 14)
    assert result.rows["pad_mask"][[1, 2, 3]].all()
    small = decode_batch({k: v[:1] for k, v in out.items()}, skel, tables, encoding, None, n_max=5)
    assert small.failures == {0: "too_many_atoms"}
    assert small.rows["pad_mask"].all()


def test_batch_sampler_mapping_with_nonleading_padding(tables, encoding):
    out, skel = sampler_batch(tables, encoding, 1)
    out["u"] = torch.cat([out.pop("u_c"), torch.full((1, 1, 3), float("nan"))], dim=1)[:, [0, 2, 1]]
    out["type"] = torch.cat([out["type"], torch.full((1, 1, 16), float("nan"))], dim=1)[:, [0, 2, 1]]
    batch_skel = {"G": torch.tensor([136]), "row": torch.tensor([[skel.rows[0], 0, skel.rows[1]]]),
                  "pad_mask": torch.tensor([[False, True, False]]),
                  "A0": torch.tensor([[14, 0, 14]])}  # targets must not leak into generation
    result = decode_batch(out, batch_skel, tables, encoding, None, n_max=20)
    assert result.failures == {}
    assert result.structures[0].composition.as_dict() == {"Ti": 2, "O": 4}
    batch_skel["fixed_A0"] = batch_skel["A0"]
    fixed = decode_batch(out, batch_skel, tables, encoding, None, n_max=20)
    assert fixed.failures == {}
    assert fixed.structures[0].composition.as_dict() == {"Si": 6}


def test_decode_batch_2048_timing(tables, encoding):
    out, skel = sampler_batch(tables, encoding, 2048)
    gen = torch.Generator().manual_seed(104)
    out["type"] += .01 * torch.randn(out["type"].shape, generator=gen)
    out["u_c"][:, 1, 0] = torch.rand(2048, generator=gen) * .15 - .3
    out["z"][:, 0] = torch.log(4 + torch.rand(2048, generator=gen, dtype=torch.float64))
    out["z"][:, 5] = torch.log(2.7 + .6 * torch.rand(2048, generator=gen, dtype=torch.float64))
    cpu_start, wall_start = time.process_time(), time.perf_counter()
    result = decode_batch(out, skel, tables, encoding, None, n_max=20)
    cpu, wall = time.process_time() - cpu_start, time.perf_counter() - wall_start
    print(f"decode_batch: 2048 random rutile samples, CPU={cpu:.3f}s, wall={wall:.3f}s, torch threads={torch.get_num_threads()}")
    assert result.failures == {}
    assert result.contains_a.all()
    assert (~result.rows["pad_mask"]).sum().item() == 2048 * 6


def test_reduced_cell_skips_niggli_on_degenerate_cells(monkeypatch):
    """Niggli segfaulted or hung on 2-step samples (NaN lengths, huge image box); LLL is used instead."""
    from orbitol.decode import reduced_cell

    calls = []
    original = Structure.get_reduced_structure

    def recording(self, reduction_algo="niggli"):
        calls.append(reduction_algo)
        return original(self, reduction_algo=reduction_algo)

    monkeypatch.setattr(Structure, "get_reduced_structure", recording)
    normal = Structure(Lattice.from_parameters(4, 5, 6, 80, 95, 110), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])
    assert reduced_cell(normal).lattice.abc == normal.get_reduced_structure("niggli").lattice.abc
    assert calls[0] == "niggli"
    calls.clear()
    for matrix in ([[50, 0, 0], [0, 50, 0], [25, 25, 1e-3]],  # skewed: ~1e9 images in the mapping search
                   [[1e-4, 0, 0], [0, 5, 0], [0, 0, 5]]):    # near-zero translation: NaN Niggli lengths
        calls.clear()
        reduced_cell(Structure(Lattice(matrix), ["Na"], [[0, 0, 0]]))
        assert calls == ["LLL"]
