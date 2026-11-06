"""CPU contract/parity checks; real FIRE + Frechet with Lennard-Jones."""
import inspect
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import torch
import torch_sim as ts
from torch_sim.models.lennard_jones import LennardJonesModel

import orbitol.nequip_relax as relax


def structure(n=2, separation=1.35):
    return Structure(Lattice.cubic(12), ["Ar"] * n,
                     [[1 + i * separation, 1 + (i % 2) * .1, 1] for i in range(n)],
                     coords_are_cartesian=True)


@pytest.fixture
def model():
    return LennardJonesModel(device=torch.device("cpu"), dtype=torch.float64,
                            compute_stress=True, use_neighbor_list=False, cutoff=3.)


@pytest.fixture
def factory(monkeypatch, tmp_path, model):
    artifact = tmp_path / "test.nequip.pth"
    artifact.write_bytes(b"LJ test loader")
    class Calculator(torch.nn.Module):
        device, dtype = model.device, model.dtype

        def forward(self, state):
            # NequIP accepts the cell proxy; LJ explicitly requires SimState.
            return model(state._state)

    monkeypatch.setitem(sys.modules, "nequip.integrations.torchsim", SimpleNamespace(
        NequIPTorchSimCalc=SimpleNamespace(from_compiled_model=lambda **_: Calculator()),
    ))
    return lambda **kwargs: relax.make_fast_batch_relaxer(artifact, "cpu", **kwargs)


def assert_result(actual, expected, *, atol=2e-11):
    assert {"final_structure", "trajectory", "nsteps"} <= actual.keys() & expected.keys()
    assert {"initial_energy", "initial_fmax", "final_fmax", "final_max_force"} <= actual.keys()
    assert actual["nsteps"] == expected["nsteps"]
    assert type(actual["trajectory"]) is type(expected["trajectory"])
    assert actual["trajectory"].energies == pytest.approx(expected["trajectory"].energies, rel=0, abs=atol)
    np.testing.assert_allclose(actual["final_structure"].cart_coords,
                               expected["final_structure"].cart_coords, rtol=0, atol=atol)
    np.testing.assert_allclose(actual["final_structure"].lattice.matrix,
                               expected["final_structure"].lattice.matrix, rtol=0, atol=atol)


@pytest.mark.parametrize("steps", [0, 1, 8])
def test_fixed_steps_and_historical_contract(monkeypatch, factory, model, steps):
    structures = [structure(), structure(3)]
    historical = relax.NequIPBatchRelaxer(calculator=model, cell_filter="frechet", max_force_abort=1e6)
    expected = historical.relax_many(structures, steps=steps)
    original, calls = ts.fire_step, []

    def counted(**kwargs):
        assert kwargs["fire_flavor"] == "ase_fire"
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(ts, "fire_step", counted)
    fast = factory()
    actual = fast.relax_many(structures, steps=steps)
    assert len(calls) == steps
    assert inspect.signature(fast.relax_many).parameters["steps"].default == 500
    assert inspect.signature(fast.relax).parameters["steps"].default == 500
    for a, b in zip(actual, expected, strict=True):
        assert_result(a, b)


def test_batch_bound_order_empty_and_single(monkeypatch, factory):
    original, batches = ts.fire_init, []

    def counted(**kwargs):
        batches.append(kwargs["state"].cell.shape[0])
        return original(**kwargs)

    monkeypatch.setattr(ts, "fire_init", counted)
    fast = factory(batch_size=2)
    assert fast.relax_many([]) == []
    output = fast.relax_many([structure(n) for n in (2, 3, 4, 2, 3)], steps=1)
    assert batches == [2, 2, 1]
    assert [len(r["final_structure"]) for r in output] == [2, 3, 4, 2, 3]
    assert fast.relax(structure(), steps=1)["nsteps"] == 1


def test_default_500_updates_even_at_zero_force(monkeypatch, factory):
    original, calls = ts.fire_step, []

    def counted(**kwargs):
        calls.append(1)
        return original(**kwargs)

    monkeypatch.setattr(ts, "fire_step", counted)
    result = factory().relax(structure(1))
    assert result["nsteps"] == len(calls) == 500
    assert result["trajectory"].energies == [0.]


@pytest.mark.parametrize("kind", ["force", "energy", "nan_force", "cell_force", "stress",
                                 "endpoint_value", "endpoint_arithmetic"])
def test_divergence_isolation_preserves_survivor_history(monkeypatch, factory, kind):
    # A diverging structure fails alone; the survivor's FIRE history matches a solo run.
    good, bad = structure(2), structure(3)
    original, traces = ts.fire_step, []

    def traced(**kwargs):
        state = original(**kwargs)
        sizes = state.n_atoms_per_system.tolist()
        if 2 in sizes:
            good_state = state[sizes.index(2)]
            traces.append({k: v.clone() for k, v in good_state.attributes.items()
                           if isinstance(v, torch.Tensor)})
        if 3 in sizes and len(traces) == 3 and not kind.startswith("endpoint_"):
            p = sizes.index(3)
            if kind == "force":
                state.forces[state.system_idx == p] = 1e9
            elif kind == "nan_force":
                state.forces[state.system_idx == p] = torch.nan
            elif kind == "cell_force":
                state.cell_forces[p] = torch.nan
            elif kind == "stress":
                state.stress[p] = torch.inf
            else:
                state.energy[p] = torch.nan
        return state

    monkeypatch.setattr(ts, "fire_step", traced)
    if kind.startswith("endpoint_"):
        convert = ts.io.state_to_structures

        def fail_endpoint(state):
            if 3 in state.n_atoms_per_system.tolist():
                error = ValueError if kind == "endpoint_value" else ArithmeticError
                raise error("injected endpoint failure")
            return convert(state)

        monkeypatch.setattr(ts.io, "state_to_structures", fail_endpoint)
    expected = factory().relax(good, steps=8)
    expected_trace = list(traces)
    traces.clear()
    fast = relax.RobustRelaxer(factory())  # Exact evaluator composition.
    with pytest.raises(RuntimeError, match="non-finite"):
        fast.relax_many([bad, good], steps=8)
    actual = fast.relax_many([good], steps=8)[0]
    # Changed batch shape changes float64 matrix-function roundoff.
    assert_result(actual, expected, atol=2e-9)
    assert len(traces) == len(expected_trace) == 8
    for actual_step, expected_step in zip(traces, expected_trace, strict=True):
        for name in expected_step:
            tolerance = 0 if name in ("dt", "alpha", "n_pos") else 2e-9
            # LJ derivatives amplify geometry roundoff: historical FIRE itself
            # differs by ~4e-8 in forces between singleton and batched execution.
            if name in ("forces", "cell_forces"):
                tolerance = 2e-7
            torch.testing.assert_close(actual_step[name], expected_step[name], rtol=0, atol=tolerance,
                                       equal_nan=True, msg=lambda msg: f"{name}: {msg}")
    reason = ("Endpoint conversion" if kind.startswith("endpoint_") else
              "Force divergence" if kind == "force" else "Non-finite")
    with pytest.raises(RuntimeError, match=reason):
        fast.relax_many([bad], steps=8)
    assert len(traces) == 8  # Logger subset retries use the cache.


def test_step_exception_rolls_back_without_resetting_fire(monkeypatch, factory):
    structures = [structure(), structure(3)]
    expected = factory().relax_many(structures, steps=8)
    original, calls = ts.fire_step, []

    def failing(**kwargs):
        state = kwargs["state"]
        calls.append(state.cell.shape[0])
        if len(calls) == 4:
            for value in state.attributes.values():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    value.add_(123)
            raise RuntimeError("CUDA out of memory")
        return original(**kwargs)

    monkeypatch.setattr(ts, "fire_step", failing)
    actual = factory().relax_many(structures, steps=8)
    assert calls == [2] * 4 + [1] * 10
    for a, b in zip(actual, expected, strict=True):
        assert_result(a, b, atol=2e-9)
        assert a["initial_energy"] == b["initial_energy"]
        assert a["initial_fmax"] == b["initial_fmax"]


def test_same_atomic_numbers_different_partition(monkeypatch):
    class CachedCalculator:
        def forward(self, state):
            if not torch.equal(state.atomic_numbers, getattr(self, "atomic_numbers", torch.zeros(0))):
                self.atomic_numbers = state.atomic_numbers
                self.system_idx = state.system_idx
                self.n_systems = state.system_idx.max().item() + 1
                self.total_atoms = len(state.atomic_numbers)
            assert self.n_systems == state.cell.shape[0], "stale n_systems"
            assert self.system_idx is state.system_idx
            assert self.total_atoms == state.positions.shape[0]
            assert state.row_vector_cell.is_contiguous()
            return self.n_systems

    pair = ts.io.structures_to_state([structure(), structure()], device="cpu", dtype=torch.float64)
    merged = pair.clone()
    merged.system_idx = torch.zeros_like(pair.system_idx)
    merged.cell = pair.cell[:1]
    calculator = CachedCalculator()
    calculator.forward(pair)
    with pytest.raises(AssertionError, match="stale n_systems"):
        calculator.forward(merged)
    relax._patch_fast_calculator(calculator)
    monkeypatch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("cache refresh used item()"))
    for state in (merged, pair):
        assert calculator.forward(state) == state.cell.shape[0]


def test_safety_check_has_one_host_transfer(monkeypatch, model):
    state = ts.io.structures_to_state([structure(), structure(3), structure(4)],
                                      device="cpu", dtype=torch.float64)
    state = ts.fire_init(state, model, cell_filter=ts.CellFilter.frechet, fire_flavor="ase_fire")
    state.forces[state.system_idx == 1] = 1e9
    state.energy[2] = torch.nan
    original, transfers = torch.Tensor.cpu, []

    def cpu(value, *args, **kwargs):
        transfers.append(value.shape)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    monkeypatch.setattr(torch.Tensor, "item", lambda *_: pytest.fail("per-system scalar synchronization"))
    assert relax._safety_codes(state, 1e6) == [0, 2, 1]
    assert transfers == [torch.Size([3])]


@pytest.mark.parametrize("built,compiled", [(False, False), (True, False), (False, True), (True, True)])
def test_oeq_build_check_and_dtype_restore(monkeypatch, tmp_path, model, built, compiled):
    artifact = tmp_path / "renamed.nequip.pth"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("model/code/model.py", "libtorch_tp_jit")
    extlib = SimpleNamespace(BUILT_EXTENSION=built, TORCH_COMPILE=compiled,
                             BUILT_EXTENSION_ERROR="build failed", TORCH_COMPILE_ERROR="ninja missing")
    monkeypatch.setitem(sys.modules, "openequivariance", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "openequivariance._torch", SimpleNamespace(extlib=extlib))
    loaded = []

    def load(**kwargs):
        loaded.append(kwargs)
        torch.set_default_dtype(torch.float64)
        return model

    monkeypatch.setitem(sys.modules, "nequip.integrations.torchsim", SimpleNamespace(
        NequIPTorchSimCalc=SimpleNamespace(from_compiled_model=load),
    ))
    previous = torch.get_default_dtype()
    if built and compiled:
        relax.make_fast_batch_relaxer(artifact, "cpu")
        assert loaded == [dict(compile_path=str(artifact), device="cpu", chemical_species_to_atom_type_map=True)]
    else:
        with pytest.raises(RuntimeError, match="OpenEquivariance.*build failed.*ninja missing"):
            relax.make_fast_batch_relaxer(artifact, "cpu")
        assert not loaded
    assert torch.get_default_dtype() == previous


def test_plain_artifact_skips_oeq_and_failed_loader_restores_dtype(monkeypatch, tmp_path):
    artifact = tmp_path / "plain.nequip.pth"
    artifact.write_bytes(b"plain")
    monkeypatch.setitem(sys.modules, "openequivariance", None)

    def fail(**kwargs):
        torch.set_default_dtype(torch.float64)
        raise ValueError("loader failed")

    monkeypatch.setitem(sys.modules, "nequip.integrations.torchsim", SimpleNamespace(
        NequIPTorchSimCalc=SimpleNamespace(from_compiled_model=fail),
    ))
    previous = torch.get_default_dtype()
    with pytest.raises(ValueError, match="loader failed"):
        relax.make_fast_batch_relaxer(artifact, "cpu")
    assert torch.get_default_dtype() == previous


def test_invalid_options(factory):
    with pytest.raises(ValueError, match="batch_size"):
        factory(batch_size=0)
    with pytest.raises(ValueError, match="max_force_abort"):
        factory(max_force_abort=float("nan"))
    with pytest.raises(ValueError, match="steps"):
        factory().relax_many([structure()], steps=-1)
