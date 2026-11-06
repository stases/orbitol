"""TorchScript NequIP backends for the pinned Crystalite logger.

Keep the core's fixed-step FIRE algorithms, units and Frechet cell filters.
NequIP 0.17's shared integration loader accepts .nequip.pth for both ASE and
torch-sim; AOTInductor is not needed. Calculators are loaded once and reused.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import operator
from pathlib import Path
import re
from types import MethodType
import zipfile

import torch
from ase.filters import FrechetCellFilter

import orbitol.core  # noqa: F401 -- bind the pin before src
from orbitol import contracts as C
from src.utils.sample_stats import NequIPBatchRelaxer, NequIPRelaxer, _EnergyTrajectory


def __getattr__(name):
    if name in ("DEFAULT_TS_PATH", "DEFAULT_OEQ_PATH"):
        return C.local_path("ORBITOL_NEQUIP_TS" if name == "DEFAULT_TS_PATH" else "ORBITOL_NEQUIP_OEQ")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@contextmanager
def _default_dtype(dtype=None):
    previous = torch.get_default_dtype()
    try:
        if dtype is not None:
            torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(previous)


class _ReuseFrechetForces(FrechetCellFilter):
    """Reuse the core's post-step force check at the next FIRE step.

    This private filter is used only inside NequIPRelaxer: all geometry updates
    go through set_positions, and no filter options change during relaxation.
    ASE already caches model predictions but recomputes nine matrix Frechet
    derivatives on every get_forces call. Return a copy to protect the cache.
    """

    _forces = None

    def set_positions(self, new, **kwargs):
        self._forces = None
        return super().set_positions(new, **kwargs)

    def get_forces(self, **kwargs):
        if kwargs:
            self._forces = None
            return super().get_forces(**kwargs)
        if self._forces is None:
            self._forces = super().get_forces()
        return self._forces.copy()


def _model_path(model_path):
    path = Path(model_path).expanduser().resolve()
    if not str(path).endswith(".nequip.pth"):
        raise ValueError(f"Expected a TorchScript .nequip.pth artifact: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def make_sequential_relaxer(model_path, device, *, fmax=0.005, max_force_abort=1e6):
    """Crystalite FIRE + Frechet, with float64 geometry and no early stopping."""
    from ase.optimize import FIRE
    from nequip.integrations.ase import NequIPCalculator

    class Float64Calculator(NequIPCalculator):
        def atoms_to_data(self, atoms):
            # Cast at conversion, before neighbor finding; casting float32
            # coordinates afterwards cannot recover their original precision.
            with _default_dtype(torch.float64):
                return super().atoms_to_data(atoms)

    with _default_dtype():
        calculator = Float64Calculator.from_compiled_model(
            compile_path=_model_path(model_path), device=device,
        )
    return NequIPRelaxer(
        calculator=calculator, optimizer_cls=FIRE, filter_cls=_ReuseFrechetForces,
        fmax=fmax, max_force_abort=max_force_abort,
    )


class _ContiguousCell:
    def __init__(self, state):
        self._state = state

    def __getattr__(self, name):
        if name == "row_vector_cell":
            return self._state.row_vector_cell.contiguous()
        return getattr(self._state, name)


def make_batch_relaxer(model_path, device, *, max_force_abort=1e6):
    """True torch-sim batching of the same TorchScript artifact (NequIP 0.17)."""
    from nequip.integrations.torchsim import NequIPTorchSimCalc

    class BatchCalculator(NequIPTorchSimCalc):
        def forward(self, state):
            # The upstream calculator refreshes only when atomic_numbers change.
            # Bisection can change system_idx while preserving atomic_numbers.
            self.setup_from_system_idx(state.atomic_numbers, state.system_idx)
            return super().forward(_ContiguousCell(state))

    with _default_dtype():
        calculator = BatchCalculator.from_compiled_model(
            compile_path=_model_path(model_path), device=device,
        )
    return NequIPBatchRelaxer(
        calculator=calculator, cell_filter="frechet", max_force_abort=max_force_abort,
    )


def _requires_oeq(path):
    # Inspect operators, so renaming an artifact cannot bypass registration.
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as archive:
        return any(b"libtorch_tp_jit" in archive.read(name)
                   for name in archive.namelist()
                   if "/code/" in name and name.endswith(".py"))


def _patch_fast_calculator(calculator):
    # Refresh the per-batch system metadata and pass a contiguous cell on every call.
    original_forward = calculator.forward

    def forward(self, state):
        self.atomic_numbers = state.atomic_numbers
        self.system_idx = state.system_idx
        self.n_systems = state.cell.shape[0]
        self.total_atoms = state.positions.shape[0]
        return original_forward(_ContiguousCell(state))

    calculator.forward = MethodType(forward, calculator)
    return calculator


def make_fast_batch_relaxer(path=None, device="cuda", *,
                            max_force_abort=1e6, batch_size=256):
    """OEQ FIRE + Frechet, 500 fixed updates by default, batches of 256.

    ``relax_many(structures, steps=500, verbose=False)`` returns ordered dicts
    with ``final_structure``, ``trajectory.energies`` (one total energy in eV)
    and ``nsteps``, exactly as the historical TorchScript relaxer. ``relax``
    accepts one structure. Failures raise RuntimeError; immediate subset retries
    reuse completed results, including through an outer RobustRelaxer.
    Read-only telemetry adds initial total energy/atomic fmax and final atomic
    (final_max_force) and Frechet filter (final_fmax) force maxima.

    This explicitly changes the historical 200-step default to fixed 500; no
    early stopping, symmetry constraints, perturbations or precision changes.
    Loading checks the OpenEquivariance extension before building the calculator.
    """
    batch_size = operator.index(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if math.isnan(max_force_abort) or max_force_abort <= 0:
        raise ValueError("max_force_abort must be positive")
    path = _model_path(C.local_path("ORBITOL_NEQUIP_OEQ") if path is None else path)
    with _default_dtype():
        if _requires_oeq(path):
            import openequivariance  # noqa: F401 -- registers TorchScript operators
            from openequivariance._torch import extlib

            if not extlib.BUILT_EXTENSION or not extlib.TORCH_COMPILE:
                raise RuntimeError(
                    f"OpenEquivariance extension unavailable for {path}: "
                    f"BUILT_EXTENSION_ERROR={extlib.BUILT_EXTENSION_ERROR}; "
                    f"TORCH_COMPILE_ERROR={extlib.TORCH_COMPILE_ERROR}"
                )
        from nequip.integrations.torchsim import NequIPTorchSimCalc

        calculator = NequIPTorchSimCalc.from_compiled_model(
            compile_path=path, device=device, chemical_species_to_atom_type_map=True,
        )
        calculator = _patch_fast_calculator(calculator)
    return _FastBatchRelaxer(_FixedFIRE(
        calculator=calculator, cell_filter="frechet", max_force_abort=max_force_abort,
        batch_size=batch_size,
    ))


def _safety_codes(state, max_force_abort):
    # Per structure: 0 safe, 1 non-finite state, 2 forces above max_force_abort.
    # One host transfer for the entire check; torch-sim itself also synchronizes.
    with torch.no_grad():
        n = state.cell.shape[0]
        norms = torch.linalg.vector_norm(state.forces, dim=1)
        fmax = norms.new_full((n,), -torch.inf)
        fmax.scatter_reduce_(0, state.system_idx, norms, reduce="amax")
        atom_bad = (~torch.isfinite(state.positions).all(dim=1)).to(torch.int32)
        bad_positions = atom_bad.new_zeros(n)
        bad_positions.scatter_reduce_(0, state.system_idx, atom_bad, reduce="amax")
        finite = torch.isfinite(fmax) & (bad_positions == 0)
        for value in (state.energy, state.stress, state.cell_forces, state.cell):
            finite &= torch.isfinite(value.reshape(n, -1)).all(dim=1)
        codes = torch.where(fmax > max_force_abort, 2, 0)
        return torch.where(finite, codes, 1).cpu().tolist()


def _retryable_fire_error(exc):
    # Out-of-memory and numerical failures; the caller splits the batch and retries.
    message = str(exc).lower()
    return (isinstance(exc, torch.cuda.OutOfMemoryError) or re.search(r"\boom\b", message)
            or any(word in message for word in (
                "out of memory", "cuda_error_out_of_memory",
                "must not contain infs or nans", "the input matrix is singular",
            )))


def _atomic_fmax(state, forces):
    """Per-system atomic maximum, without a host synchronization."""
    norms = torch.linalg.vector_norm(forces, dim=1)
    values = norms.new_full((len(state.cell),), -torch.inf)
    return values.scatter_reduce_(0, state.system_idx, norms, reduce="amax")


def _initial_telemetry(state):
    """One read-only transfer of initial total energy and atomic fmax."""
    with torch.no_grad():
        return torch.stack((state.energy.reshape(-1), _atomic_fmax(state, state.forces)),
                           dim=1).detach().cpu().tolist()


def _final_telemetry(state):
    """Final atomic fmax and Frechet filter fmax (deformed atomic and cell forces)."""
    from torch_sim.optimizers.cell_filters import deform_grad

    with torch.no_grad():
        deformation = deform_grad(state.reference_cell.mT, state.row_vector_cell)
        atoms = torch.bmm(state.forces.unsqueeze(1), deformation[state.system_idx]).squeeze(1)
        cells = torch.linalg.vector_norm(state.cell_forces, dim=-1).amax(dim=-1)
        return torch.stack((_atomic_fmax(state, state.forces),
                            torch.maximum(_atomic_fmax(state, atoms), cells)),
                           dim=1).detach().cpu().tolist()


class _FixedFIRE(NequIPBatchRelaxer):
    """Private result producer; RuntimeError slots are consumed by RobustRelaxer."""

    def __init__(self, *, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def relax_many(self, structs, steps=500, verbose=False):
        del verbose
        import torch_sim as ts
        from torch_sim.state import _pop_states

        structures = list(structs)
        results = [None] * len(structures)
        initial = {}

        def run(indices, state=None, completed=0):
            if state is None:
                building_state = True
                try:
                    state = ts.io.structures_to_state(
                        [structures[i] for i in indices], device=self.calculator.device,
                        dtype=self.calculator.dtype,
                    )
                    building_state = False
                    state = ts.fire_init(state=state, model=self.calculator,
                                         cell_filter=ts.CellFilter.frechet, fire_flavor="ase_fire")
                except (ValueError, RuntimeError) as exc:
                    if not _retryable_fire_error(exc) and not (building_state and isinstance(exc, ValueError)):
                        raise
                    reason = _wrapped(exc)
                else:
                    reason = None
                    for i, values in zip(indices, _initial_telemetry(state), strict=True):
                        initial.setdefault(i, values)
                if reason is not None:
                    # Release failed allocations before smaller retries.
                    state = None
                    if len(indices) == 1:
                        results[indices[0]] = reason
                    else:
                        middle = len(indices) // 2
                        run(indices[:middle])
                        run(indices[middle:])
                    return

            # Each step drops unsafe structures, then takes one FIRE step or splits on a retryable failure.
            # Fixed batches never refill, so all survivors share a step counter.
            while True:
                codes = _safety_codes(state, self.max_force_abort)
                removed = [p for p, code in enumerate(codes) if code]
                if removed:
                    for p in removed:
                        message = ("Non-finite energy, forces, cell or stress" if codes[p] == 1
                                   else f"Force divergence: fmax > {self.max_force_abort:g} eV/A")
                        results[indices[p]] = RuntimeError(message)
                    indices = [i for i, code in zip(indices, codes, strict=True) if not code]
                    if not indices:
                        return
                    state = _pop_states(state, removed)[0]
                if completed == steps:
                    break
                trial = None
                try:
                    # FIRE mutates its argument before the model can fail.
                    trial = state.clone()
                    trial = ts.fire_step(state=trial, model=self.calculator, fire_flavor="ase_fire")
                except (ValueError, RuntimeError) as exc:
                    if not _retryable_fire_error(exc):
                        raise
                    reason = _wrapped(exc)
                else:
                    state, completed = trial, completed + 1
                    continue
                trial = None
                if len(indices) == 1:
                    results[indices[0]] = reason
                else:
                    middle = len(indices) // 2
                    left, right = _pop_states(state, list(range(middle, len(indices))))
                    run(indices[:middle], left, completed)
                    run(indices[middle:], ts.concatenate_states(right), completed)
                return

            energies = state.energy.detach().cpu().tolist()
            forces = _final_telemetry(state)
            for position, (i, energy, (atomic_fmax, filter_fmax)) in enumerate(
                    zip(indices, energies, forces, strict=True)):
                try:
                    final = ts.io.state_to_structures(state[position])[0] if steps else structures[i]
                except (ValueError, ArithmeticError) as exc:
                    results[i] = RuntimeError(f"Endpoint conversion: {type(exc).__name__}: {exc}")
                    continue
                results[i] = {"final_structure": final, "trajectory": _EnergyTrajectory([float(energy)]),
                              "nsteps": steps, "initial_energy": initial[i][0],
                              "initial_fmax": initial[i][1], "final_fmax": filter_fmax,
                              "final_max_force": atomic_fmax}

        for start in range(0, len(structures), self.batch_size):
            run(list(range(start, min(start + self.batch_size, len(structures)))))
        return results


def _wrapped(exc):
    return RuntimeError(f"non-finite relaxation (wrapped {type(exc).__name__}: {exc})")


class RobustRelaxer:
    """Bisect Python failures and let Crystalite count failing structures.

    KeyboardInterrupt propagates. Native crashes (including segfaults) cannot
    be caught in-process; this wrapper does not provide process isolation.
    Batch errors deliberately omit child-batch indices. The logger bisects to
    count each singleton with its own reason. Results are retained for those
    immediate subset retries, so healthy structures are not relaxed twice.
    """

    def __init__(self, backend):
        self.backend = backend
        self._pending = {}
        self._pending_scope = set()
        self._pending_options = None

    def relax(self, struct, steps=200, verbose=False):
        self._pending.clear()
        try:
            return self.backend.relax(struct, steps=steps, verbose=verbose)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise _wrapped(exc) from exc

    def relax_many(self, structs, steps=200, verbose=False):
        structures = list(structs)
        if not structures:
            return []

        def attempt(group):
            try:
                if hasattr(self.backend, "relax_many"):
                    results = list(self.backend.relax_many(group, steps=steps, verbose=verbose))
                else:
                    results = [self.backend.relax(s, steps=steps, verbose=verbose) for s in group]
                if len(results) != len(group):
                    raise ValueError("Backend returned a mismatched number of relaxations")
                return results
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                if len(group) == 1:
                    error = _wrapped(exc)
                    # Retain diagnostics without retaining failed optimizer
                    # frames (and potentially their large GPU tensors).
                    error.__cause__ = exc.with_traceback(None)
                    return [error]
            # Leave the except block before retrying: its traceback can keep
            # the failed batch alive and make every smaller OOM retry fail too.
            middle = len(group) // 2
            return attempt(group[:middle]) + attempt(group[middle:])

        ids = {id(s) for s in structures}
        retry = (ids < self._pending_scope and ids <= self._pending.keys()
                 and self._pending_options == (steps, verbose))
        if retry:
            results = [self._pending[id(s)][1] for s in structures]
        else:
            self._pending.clear()
            results = attempt(structures)
        failures = [i for i, result in enumerate(results) if isinstance(result, RuntimeError)]
        if failures:
            error = results[failures[0]]
            if len(structures) == 1:
                self._pending.pop(id(structures[0]), None)
                raise error
            if not retry:
                # Strong references prevent id reuse; keep at most one failed
                # batch. A fresh/full batch or changed options invalidates it.
                self._pending = {id(s): (s, r) for s, r in zip(structures, results, strict=True)}
                self._pending_scope = ids
                self._pending_options = (steps, verbose)
            # Do not embed a child error here: its 'systems [0]' could make the
            # logger drop an unrelated structure, and mixed failures need their
            # own individual reason counters. The original error is the cause.
            raise RuntimeError("non-finite relaxation (batch failed; bisect to identify structures)") from error
        for s in structures:
            self._pending.pop(id(s), None)
        return results


class _FastBatchRelaxer(RobustRelaxer):
    def relax(self, struct, steps=500, verbose=False):
        return self.relax_many([struct], steps=steps, verbose=verbose)[0]

    def relax_many(self, structs, steps=500, verbose=False):
        steps = operator.index(steps)
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        return super().relax_many(structs, steps=steps, verbose=verbose)
