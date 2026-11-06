"""Frozen reference FIRE driver from before relaxation telemetry was added.

Used only to verify that observational telemetry leaves every update unchanged.
"""
from orbitol.nequip_relax import (
    NequIPBatchRelaxer, _EnergyTrajectory, _retryable_fire_error, _safety_codes, _wrapped,
)


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
            for position, (i, energy) in enumerate(zip(indices, energies, strict=True)):
                try:
                    final = ts.io.state_to_structures(state[position])[0] if steps else structures[i]
                except (ValueError, ArithmeticError) as exc:
                    results[i] = RuntimeError(f"Endpoint conversion: {type(exc).__name__}: {exc}")
                    continue
                results[i] = {"final_structure": final, "trajectory": _EnergyTrajectory([float(energy)]),
                              "nsteps": steps}

        for start in range(0, len(structures), self.batch_size):
            run(list(range(start, min(start + self.batch_size, len(structures)))))
        return results
