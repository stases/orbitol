"""Read-only FIRE observations, order preservation and GPU regression gate."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
from pymatgen.core import Structure
import pytest
import torch
import torch_sim as ts
from torch_sim.optimizers.cell_filters import deform_grad
from torch_sim.state import _pop_states

from orbitol import contracts as C, nequip_relax as relax
from test_fast_relax import factory, model, structure  # noqa: F401 -- shared CPU LJ fixtures


def historical(calculator, batch_size=256):
    path = Path(__file__).parent / "fixtures/fixed_fire_before_telemetry.py"
    spec = importlib.util.spec_from_file_location("fixed_fire_before_telemetry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._FixedFIRE(calculator=calculator, cell_filter="frechet",
                             max_force_abort=1e6, batch_size=batch_size)


def assert_bitwise(actual, expected):
    assert actual["trajectory"].energies == expected["trajectory"].energies
    assert actual["nsteps"] == expected["nsteps"]
    for attr in ("cart_coords", "frac_coords"):
        np.testing.assert_array_equal(getattr(actual["final_structure"], attr),
                                       getattr(expected["final_structure"], attr))
    np.testing.assert_array_equal(actual["final_structure"].lattice.matrix,
                                   expected["final_structure"].lattice.matrix)


def test_cpu_trajectory_is_bitwise_unchanged(factory):
    fast = factory()
    structures = [structure(2), structure(3), structure(4)]
    expected = historical(fast.backend.calculator).relax_many(structures, steps=8)
    actual = fast.relax_many(structures, steps=8)
    for a, b in zip(actual, expected, strict=True):
        assert_bitwise(a, b)


def test_telemetry_values_host_transfers_and_read_only(monkeypatch, model):
    inputs = [structure(2), structure(3)]
    state = ts.io.structures_to_state(inputs, device="cpu", dtype=torch.float64)
    single_point = model(state)
    state = ts.fire_init(state, model, cell_filter=ts.CellFilter.frechet, fire_flavor="ase_fire")
    initial = relax._initial_telemetry(state)
    assert [v[0] for v in initial] == single_point["energy"].tolist()
    for i in range(2):
        assert initial[i][1] == torch.linalg.vector_norm(single_point["forces"][state.system_idx == i], dim=1).max()
    for _ in range(5):
        state = ts.fire_step(state, model, fire_flavor="ase_fire")
    before = {k: v.clone() for k, v in state.attributes.items() if isinstance(v, torch.Tensor)}
    deformation = deform_grad(state.reference_cell.mT, state.row_vector_cell)
    atoms = torch.bmm(state.forces.unsqueeze(1), deformation[state.system_idx]).squeeze(1)
    expected = []
    for i in range(2):
        raw = torch.linalg.vector_norm(state.forces[state.system_idx == i], dim=1).max().item()
        atom_filter = torch.linalg.vector_norm(atoms[state.system_idx == i], dim=1).max().item()
        cell_filter = torch.linalg.vector_norm(state.cell_forces[i], dim=-1).max().item()
        expected.append([raw, max(atom_filter, cell_filter)])
    original, transfers = torch.Tensor.cpu, []

    def cpu(value, *args, **kwargs):
        transfers.append(value.shape)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("scalar host transfer"))
    relax._initial_telemetry(state)
    assert relax._final_telemetry(state) == expected
    assert transfers == [torch.Size([2, 2]), torch.Size([2, 2])]
    for key, value in before.items():
        assert torch.equal(state.attributes[key], value)


def test_failed_init_captures_only_successful_halves(monkeypatch, factory):
    inputs = [structure(2), structure(3)]
    fast = factory()
    expected = fast.relax_many(inputs, steps=0)
    original, inits = ts.fire_init, []

    def fail_batch(**kwargs):
        inits.append(len(kwargs["state"].cell))
        if len(kwargs["state"].cell) > 1:
            raise RuntimeError("out of memory")
        return original(**kwargs)

    monkeypatch.setattr(ts, "fire_init", fail_batch)
    actual = fast.relax_many(inputs, steps=1)
    assert inits == [2, 1, 1]
    for a, b in zip(actual, expected, strict=True):
        assert a["initial_energy"] == b["initial_energy"]
        assert a["initial_fmax"] == b["initial_fmax"]


def test_filter_maximum_uses_deformed_atomic_forces(model):
    state = ts.io.structures_to_state([structure(2)], device="cpu", dtype=torch.float64)
    state = ts.fire_init(state, model, cell_filter=ts.CellFilter.frechet, fire_flavor="ase_fire")
    state.cell *= .5
    state.cell_forces.zero_()
    state.forces[:] = torch.tensor([1., 0., 0.], dtype=state.dtype)
    # The reference filter maximum need not bound raw atomic fmax under contraction.
    assert relax._final_telemetry(state) == [[1., .5]]


def test_torchsim_site_order_survives_fire_pop_and_concatenate(model):
    inputs = [Structure(s.lattice, ["Ar", "Ne", "Ar", "He"][:len(s)], s.frac_coords)
              for s in (structure(3), structure(4), structure(2))]
    state = ts.io.structures_to_state(inputs, device="cpu", dtype=torch.float64)
    for initial, final in zip(inputs, ts.io.state_to_structures(state), strict=True):
        assert initial.species == final.species
        np.testing.assert_allclose(initial.cart_coords, final.cart_coords, atol=1e-14)
    state = ts.fire_init(state, model, cell_filter=ts.CellFilter.frechet, fire_flavor="ase_fire")
    state = ts.fire_step(state, model, fire_flavor="ase_fire")
    keep, popped = _pop_states(state, [1])
    reordered = ts.concatenate_states([keep, *popped])
    for i, output in zip((0, 2, 1), ts.io.state_to_structures(reordered), strict=True):
        assert output.species == inputs[i].species
        np.testing.assert_array_equal(output.cart_coords, ts.io.state_to_structures(state[i])[0].cart_coords)


@pytest.mark.gpu
@pytest.mark.data
def test_fast_nequip_telemetry_matches_historical_gpu():
    """Sixteen real MP20 structures against the frozen pre-change driver.

    OEQ convolutions use atomics and have no deterministic mode. Over 500 FIRE steps
    two historical runs already diverge by up to ~0.3 A on a sensitive structure, so
    parity is checked over 20 steps, before last-bit differences are amplified; the
    500-step run checks the telemetry itself.
    """
    data = Path(C.dataset_spec("mp20").data_root, "raw", "val.csv")
    if not data.is_file() or not Path(relax.DEFAULT_OEQ_PATH).is_file():
        pytest.skip("requires MP20 validation CSV and OEQ artifact; set ORBITOL_DATA_HOME or ORBITOL_MP20_DATA_ROOT and ORBITOL_NEQUIP_OEQ")
    torch.use_deterministic_algorithms(False)
    with data.open(newline="") as stream:
        from itertools import islice
        structures = [Structure.from_str(row["cif"], fmt="cif") for row in islice(csv.DictReader(stream), 16)]
    assert len(structures) == 16
    fast = relax.make_fast_batch_relaxer(device="cuda", batch_size=16)
    calculator = fast.backend.calculator
    state = ts.io.structures_to_state(structures, device=calculator.device, dtype=calculator.dtype)
    energies = calculator(state)["energy"].detach().cpu().tolist()  # forces need autograd

    def gaps(a, b):
        return (max(abs(x["trajectory"].energies[-1] - y["trajectory"].energies[-1]) for x, y in zip(a, b)),
                max(float(np.abs(x["final_structure"].cart_coords - y["final_structure"].cart_coords).max())
                    for x, y in zip(a, b)))

    expected = historical(calculator, batch_size=16).relax_many(structures, steps=20)
    repeat = historical(calculator, batch_size=16).relax_many(structures, steps=20)
    actual = fast.relax_many(structures, steps=20)
    noise, change = gaps(repeat, expected), gaps(actual, expected)
    print("20 steps: historical repeat |dE|, |dx|:", noise, " telemetry vs historical:", change)
    assert change[0] <= 1e-4 and change[1] <= 1e-5
    actual = fast.relax_many(structures, steps=500)
    for initial, energy, a in zip(structures, energies, actual, strict=True):
        assert a["final_structure"].species == initial.species
        assert a["nsteps"] == 500
        assert abs(a["initial_energy"] - energy) <= 1e-6 * max(1.0, abs(energy))  # OEQ atomics
        assert np.isfinite([a["initial_fmax"], a["final_fmax"], a["final_max_force"]]).all()
