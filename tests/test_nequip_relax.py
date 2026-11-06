"""CPU failure isolation and numerical preservation, without NequIP."""
import sys
from types import SimpleNamespace
import weakref

from ase.calculators.emt import EMT
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import torch

import orbitol.core  # noqa: F401
from orbitol.nequip_relax import (
    RobustRelaxer, _ReuseFrechetForces, make_batch_relaxer, make_sequential_relaxer,
)
from src.utils.sample_stats import NequIPRelaxer


class Backend:
    def __init__(self, bad=(), max_batch=100):
        self.bad, self.max_batch, self.calls = bad, max_batch, []

    def relax(self, struct, **kwargs):
        return self.relax_many([struct], **kwargs)[0]

    def relax_many(self, structs, **kwargs):
        self.calls.append(list(structs))
        assert kwargs == dict(steps=7, verbose=False)
        if len(structs) > self.max_batch:
            raise MemoryError("batch too large")
        if any(s in self.bad for s in structs):
            raise ValueError("broken systems [0]")
        return [{"final_structure": s} for s in structs]


def test_bisects_ordinary_batch_exceptions_preserving_order():
    backend = Backend(max_batch=2)
    results = RobustRelaxer(backend).relax_many(list(range(5)), steps=7)
    assert [r["final_structure"] for r in results] == list(range(5))
    assert backend.calls == [[0, 1, 2, 3, 4], [0, 1], [2, 3, 4], [2], [3, 4]]


def test_failed_batch_frames_are_released_before_smaller_retries():
    live = weakref.WeakSet()

    class Allocation:
        pass

    def relax_many(structs, **kwargs):
        if len(structs) > 1:
            allocation = Allocation()
            live.add(allocation)
            raise MemoryError("batch allocation failed")
        assert not live, "Failed-batch traceback still retains its allocation"
        return [{"final_structure": structs[0]}]

    assert len(RobustRelaxer(SimpleNamespace(relax_many=relax_many)).relax_many([0, 1])) == 2


def test_isolates_all_failures_without_exposing_child_indices():
    backend = Backend(bad={1, 3})
    wrapped = RobustRelaxer(backend)
    with pytest.raises(RuntimeError, match="non-finite relaxation") as caught:
        wrapped.relax_many([0, 1, 2, 3], steps=7)
    assert "systems" not in str(caught.value)
    assert [0] in backend.calls and [2] in backend.calls
    assert [1] in backend.calls and [3] in backend.calls
    count = len(backend.calls)
    for i in range(4):
        if i in backend.bad:
            with pytest.raises(RuntimeError, match="wrapped ValueError"):
                wrapped.relax_many([i], steps=7)
        else:
            assert wrapped.relax_many([i], steps=7) == [{"final_structure": i}]
    assert len(backend.calls) == count  # logger retries reuse completed work
    assert wrapped._pending == {}
    # A later full call must execute again, never return old endpoints.
    wrapped.relax_many([0, 2], steps=7)
    assert len(backend.calls) == count + 1


@pytest.mark.parametrize("method", ["relax", "relax_many"])
@pytest.mark.parametrize("error", [ValueError("bad"), SystemExit("bad"), RuntimeError("force divergence")])
def test_single_wraps_with_original_cause(method, error):
    def fail(*args, **kwargs):
        raise error
    wrapped = RobustRelaxer(SimpleNamespace(relax=fail, relax_many=fail))
    with pytest.raises(RuntimeError) as caught:
        getattr(wrapped, method)(0 if method == "relax" else [0])
    assert str(caught.value) == f"non-finite relaxation (wrapped {type(error).__name__}: {error})"
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("method", ["relax", "relax_many"])
def test_keyboard_interrupt_is_not_swallowed(method):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt
    wrapped = RobustRelaxer(SimpleNamespace(relax=interrupt, relax_many=interrupt))
    with pytest.raises(KeyboardInterrupt):
        getattr(wrapped, method)(0 if method == "relax" else [0, 1])


def test_empty_and_malformed_batches():
    backend = SimpleNamespace(relax_many=lambda *args, **kwargs: [])
    wrapped = RobustRelaxer(backend)
    assert wrapped.relax_many([]) == []
    with pytest.raises(RuntimeError, match="non-finite.*mismatched"):
        wrapped.relax_many([0])


def test_sequential_backend_supports_many():
    backend = SimpleNamespace(relax=lambda s, **kw: {"final_structure": s})
    assert RobustRelaxer(backend).relax_many([1, 2]) == [{"final_structure": 1}, {"final_structure": 2}]


def test_frechet_reuse_preserves_actual_core_fire_endpoint():
    structure = Structure(Lattice.cubic(3.7), ["Cu"] * 4,
                          [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    outputs, derivative_counts = [], []
    for filter_cls in (FrechetCellFilter, _ReuseFrechetForces):
        counter = []

        class CountedFilter(filter_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                original = self.expm_frechet

                def counted(*a, **kw):
                    counter.append(1)
                    return original(*a, **kw)
                self.expm_frechet = counted

        relaxer = NequIPRelaxer(calculator=EMT(), optimizer_cls=FIRE, filter_cls=CountedFilter,
                               fmax=.005, max_force_abort=1e6)
        outputs.append(relaxer.relax(structure, steps=8))
        derivative_counts.append(len(counter))
    left, right = outputs
    np.testing.assert_array_equal(left["final_structure"].cart_coords, right["final_structure"].cart_coords)
    np.testing.assert_array_equal(left["final_structure"].lattice.matrix, right["final_structure"].lattice.matrix)
    assert left["trajectory"].energies == right["trajectory"].energies
    assert derivative_counts == [8 * 2 * 9, (8 + 1) * 9]


@pytest.mark.parametrize("mode", ["batch", "sequential"])
def test_factories_keep_dtype_and_backend_settings(monkeypatch, mode):
    """Use fake integration classes, including the upstream stale-batch behavior."""
    class Calculator:
        @classmethod
        def from_compiled_model(cls, **kwargs):
            torch.set_default_dtype(torch.float64)
            instance = cls()
            instance.loaded = kwargs
            return instance

        def atoms_to_data(self, atoms):
            return torch.tensor([1.123456789])

        def setup_from_system_idx(self, numbers, indices):
            self.n_systems = int(indices.max()) + 1

        def forward(self, state):
            assert state.row_vector_cell.is_contiguous()
            return self.n_systems

    module = f"nequip.integrations.{'ase' if mode == 'sequential' else 'torchsim'}"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(NequIPCalculator=Calculator, NequIPTorchSimCalc=Calculator))
    import orbitol.nequip_relax as module_under_test
    monkeypatch.setattr(module_under_test, "_model_path", lambda path: str(path))
    torch.set_default_dtype(torch.float32)
    factory = make_sequential_relaxer if mode == "sequential" else make_batch_relaxer
    relaxer = factory("test.nequip.pth", "cpu", max_force_abort=42)
    assert torch.get_default_dtype() == torch.float32
    assert relaxer.max_force_abort == 42
    assert relaxer.calculator.loaded == dict(compile_path="test.nequip.pth", device="cpu")
    if mode == "sequential":
        assert relaxer.fmax == .005 and relaxer.optimizer_cls is FIRE
        assert relaxer.calculator.atoms_to_data(None).dtype == torch.float64
        assert torch.get_default_dtype() == torch.float32
    else:
        assert relaxer.cell_filter == "frechet"
        state = SimpleNamespace(atomic_numbers=torch.tensor([1, 1]), system_idx=torch.tensor([0, 0]),
                                row_vector_cell=torch.ones(2, 3, 3).transpose(1, 2))
        assert relaxer.calculator.forward(state) == 1
        state.system_idx = torch.tensor([0, 1])
        assert relaxer.calculator.forward(state) == 2
