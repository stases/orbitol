"""Observational relaxation geometry; no neighbour-list/Cython distances."""
from itertools import product

import numpy as np


_IMAGES = np.array(list(product((-1., 0., 1.), repeat=3)))


def relaxation_geometry(initial, final, initial_energy=None, final_energy=None):
    """Same-order displacements (Å), strain, signed volume %, energy drop (meV/atom).

    Fractional displacement loses its unweighted mean translation before the
    27-image search in the LLL-reduced final lattice. LLL and all distances use
    numpy, as in eval_diagnostics.minimum_distance. Invalid geometry or a species
    sequence mismatch returns None; missing/nonfinite energies give a null drop.
    """
    try:
        n = len(initial)
        if not n or n != len(final) or initial.species != final.species:
            return None
        lin, lout = np.asarray(initial.lattice.matrix), np.asarray(final.lattice.matrix)
        fin, fout = np.asarray(initial.frac_coords), np.asarray(final.frac_coords)
        if not all(np.isfinite(a).all() for a in (lin, lout, fin, fout)):
            return None
        vin, vout = abs(np.linalg.det(lin)), abs(np.linalg.det(lout))
        if not np.isfinite([vin, vout]).all() or vin <= 0 or vout <= 0:
            return None
        delta = fout - fin
        delta -= np.round(delta)
        delta -= delta.mean(axis=0)
        delta -= np.round(delta)
        reduced = final.lattice.get_lll_reduced_lattice().matrix
        if not np.isfinite(reduced).all():
            return None
        frac = np.linalg.solve(reduced.T, (delta @ lout).T).T
        frac -= np.round(frac)
        distances = np.linalg.norm((frac[:, None, :] + _IMAGES) @ reduced, axis=-1).min(axis=1)
        ft = np.linalg.solve(lin, lout)  # Row lattices: F^T = L_in^-1 L_final.
        strain = .5 * (ft @ ft.T - np.eye(3))
        result = dict(rms_disp=float(np.sqrt(np.mean(distances ** 2))),
                      max_disp=float(distances.max()), strain=float(np.linalg.norm(strain)),
                      volume_change_pct=float(100 * (vout / vin - 1)))
        if not all(np.isfinite(v) for v in result.values()):
            return None
        drop = None
        if initial_energy is not None and final_energy is not None:
            drop = 1000 * (float(initial_energy) - float(final_energy)) / n
            if not np.isfinite(drop):
                drop = None
        return {**result, "energy_drop_meV": drop}
    except Exception:
        # A diagnostic must never discard a successful relaxation/hull result.
        return None
