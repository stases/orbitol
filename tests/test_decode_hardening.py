"""Per-structure exception isolation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Species, Structure

from orbitol import decode
from test_decode import encoding, sampler_batch  # noqa: F401


@pytest.fixture(scope="module")
def tables(frozen_tables):
    return frozen_tables


@pytest.mark.parametrize("stage", ["expand", "emit", "contain", "reduce", "reduce_once"])
def test_exception_is_local(monkeypatch, tables, encoding, stage):
    out, skel = sampler_batch(tables, encoding)
    skels = [replace(skel, A0=(a, 8)) for a in (22, 14, 24)]
    out["z"][1, 0] = np.log(4.8)
    expected = decode.decode_batch(out, skels, tables, encoding, None, n_max=20)
    assert not expected.failures
    name = dict(expand="_expand_members", emit="emit_primitive", contain="contains_ops",
                reduce="reduced_cell", reduce_once="reduced_cell")[stage]
    original = getattr(decode, name)
    raised = False

    def broken(*args, **kwargs):
        nonlocal raised
        if stage == "expand":
            bad = args[2][0] == 14
        elif stage == "emit":
            bad = args[0].species[0] == 14
        else:
            bad = args[0].atomic_numbers[0] == 14
        if bad and (stage != "reduce_once" or not raised):
            raised = True
            raise RuntimeError("injected bad structure")
        return original(*args, **kwargs)

    monkeypatch.setattr(decode, name, broken)
    got = decode.decode_batch(out, skels, tables, encoding, None, n_max=20)
    assert got.failures == got.rows.failures == {1: "decode_error"}
    assert got.structures[1] is None
    assert got.rows["pad_mask"][1].all()
    for key in ("A0", "F1", "Y1"):
        assert not got.rows[key][1].any()
    for key in got.rows:
        a, b = got.rows[key][[0, 2]], expected.rows[key][[0, 2]]
        if key == "Y1":
            a, b = a.view(torch.int32), b.view(torch.int32)
        assert torch.equal(a, b)
    for i in (0, 2):
        assert got.structures[i] == expected.structures[i]
    assert np.array_equal(got.requested_G, expected.requested_G)
    assert got.contains_a.tolist() == [True, stage in ("reduce", "reduce_once"), True]


def test_row_validation_exception_is_local(tables, encoding):
    out, skel = sampler_batch(tables, encoding)
    structures = decode.decode_batch(out, skel, tables, encoding, None, n_max=20).structures
    expected = decode.to_crystalite_rows(structures, 20)

    class BadValidation:
        def __len__(self):
            return len(structures[1])

        @property
        def is_ordered(self):
            raise RuntimeError("injected validation failure")

    got = decode.to_crystalite_rows([structures[0], BadValidation(), structures[2]], 20)
    assert got.failures == {1: "decode_error"}
    for key in got:
        a, b = got[key][[0, 2]], expected[key][[0, 2]]
        if key == "Y1":
            a, b = a.view(torch.int32), b.view(torch.int32)
        assert torch.equal(a, b)
    assert got["pad_mask"][1].all()
    assert all(not got[key][1].any() for key in ("A0", "F1", "Y1"))


@pytest.mark.parametrize("failure", ["lll", "inverse"])
def test_niggli_guard_exception_uses_lll(monkeypatch, failure):
    cell = Lattice.cubic(4)

    def broken(*args, **kwargs):
        raise RuntimeError("guard failed")

    if failure == "lll":
        monkeypatch.setattr(Lattice, "get_lll_reduced_lattice", broken)
    else:
        monkeypatch.setattr(np.linalg, "inv", broken)
    assert not decode._niggli_safe(cell)
    calls, reduced = [], object()
    structure = SimpleNamespace(lattice=cell, get_reduced_structure=lambda *, reduction_algo:
                                calls.append(reduction_algo) or reduced)
    assert decode.reduced_cell(structure) is reduced
    assert calls == ["LLL"]


@pytest.mark.parametrize("kind", ["type", "u", "z", "skeleton", "n_max"])
def test_integration_errors_still_raise(tables, encoding, kind):
    out, skel = sampler_batch(tables, encoding)
    if kind == "type":
        out["type"] = out["type"][..., :15]
    elif kind == "u":
        out["u_c"] = out["u_c"][:1]
    elif kind == "z":
        out["z"] = out["z"][:, :5]
    elif kind == "skeleton":
        skel = [skel]
    with pytest.raises(ValueError):
        decode.decode_batch(out, skel, tables, encoding, None, n_max=0 if kind == "n_max" else 20)


@pytest.mark.parametrize("same", [True, False])
def test_contains_reads_species_once_and_keeps_species_equality(tables, same):
    species = [Species("Fe", 2), Species("Fe", 2 if same else 3)]
    structure = Structure(Lattice.cubic(4), species, [[.1, 0, 0], [.9, 0, 0]])

    class CountedSpecies:
        reads = 0

        def __len__(self):
            return len(structure)

        def __getattr__(self, name):
            return getattr(structure, name)

        @property
        def species(self):
            self.reads += 1
            return structure.species

    counted = CountedSpecies()
    assert decode.contains_ops(counted, 2, tables, **decode.CHECK_A) == same
    assert counted.reads == 1
