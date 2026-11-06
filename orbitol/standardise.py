"""Single-pass MP-20 standardisation into the frozen fractional setting.

Displacements measure internal relaxation, after expressing input sites in the
standard basis, in the idealised cell's Cartesian metric. Homogeneous lattice
strain is reported separately as the Frobenius norm of principal stretch minus
identity (dimensionless). Neither diagnostic performs a second symmetry search.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import spglib
from pymatgen.core import Lattice, Structure
from pyxtal.symmetry import spglib_hall_numbers
from scipy.optimize import linear_sum_assignment

from orbitol import contracts as C
from orbitol.tables import TableSet, load_tables

SYMPREC = 0.1
ANGLE_TOLERANCE = 5.0
FRAC_TOL = 1e-6


@lru_cache(maxsize=1)
def default_tables() -> TableSet:
    return load_tables(C.TABLES_DIR)


def fractional_cost(x, y):
    delta = np.asarray(x, dtype=np.float64)[:, None] - np.asarray(y, dtype=np.float64)[None]
    return np.max(np.abs(delta - np.rint(delta)), axis=-1)


def species_permutation(x, types, y, target_types, *, tol=FRAC_TOL):
    """Return a species-preserving bijection, or fail at the stated tolerance."""
    x, y = np.asarray(x), np.asarray(y)
    types, target_types = np.asarray(types), np.asarray(target_types)
    if x.shape != y.shape or types.shape != target_types.shape:
        raise ValueError("Atom counts differ")
    permutation = np.empty(len(x), dtype=np.int64)
    for z in np.union1d(types, target_types):
        ii, jj = np.flatnonzero(types == z), np.flatnonzero(target_types == z)
        if len(ii) != len(jj):
            raise ValueError(f"Species counts differ for Z={z}")
        cost = fractional_cost(x[ii], y[jj])
        # A forbidden edge must cost more than any completely valid assignment.
        a, b = linear_sum_assignment(np.where(cost <= tol, cost, len(ii) + 1.0))
        if not np.isfinite(cost[a, b]).all() or np.any(cost[a, b] > tol):
            raise ValueError(f"Fractional round trip exceeds {tol:g} for Z={z}")
        permutation[ii[a]] = jj[b]
    return permutation


def assert_frozen_cell(lattice, positions, types, group, tables, *, tol=FRAC_TOL):
    """Check every frozen operation on the sites and the lattice metric."""
    a = tables.arrays
    count = int(a["group_ops_count"][group])
    if not count:
        raise ValueError(f"Space group {group} is absent from the tables")
    gram = np.asarray(lattice, dtype=np.float64) @ np.asarray(lattice, dtype=np.float64).T
    for R, t in zip(a["group_ops_R"][group, :count], a["group_ops_t"][group, :count]):
        if np.linalg.norm(R.T @ gram @ R - gram) > 1e-8 * np.linalg.norm(gram):
            raise ValueError(f"Lattice does not obey frozen group {group}")
        species_permutation(positions @ R.T + t, types, positions, types, tol=tol)


@dataclass(frozen=True)
class Standardised:
    G: int
    hall: int
    lattice: np.ndarray
    positions: np.ndarray
    types: np.ndarray
    transformation_matrix: np.ndarray
    origin_shift: np.ndarray
    rotation: np.ndarray
    frozen_shift: np.ndarray
    disp_max: float
    disp_mean: float
    strain: float

    @property
    def structure(self) -> Structure:
        return Structure(self.lattice, self.types, self.positions)


def standardise(structure: Structure, *, tables: TableSet | None = None) -> Standardised:
    """Use exactly one symmetry dataset, explicitly at 0.1 Angstrom / 5 degrees."""
    if not structure.is_ordered or not len(structure):
        raise ValueError("Standardisation requires a nonempty, ordered structure")
    tables = default_tables() if tables is None else tables
    cell = (np.asarray(structure.lattice.matrix, dtype=np.float64),
            np.asarray(structure.frac_coords, dtype=np.float64),
            np.asarray(structure.atomic_numbers, dtype=np.int64))
    if not all(np.isfinite(v).all() for v in cell):
        raise ValueError("Nonfinite input cell")
    ds = spglib.get_symmetry_dataset(cell, symprec=SYMPREC, angle_tolerance=ANGLE_TOLERANCE)
    if ds is None:
        raise ValueError("spglib.get_symmetry_dataset failed")
    group, hall = int(ds.number), int(ds.hall_number)
    frozen_hall = int(tables.arrays["group_hall"][group])
    if not frozen_hall:
        raise ValueError(f"Space group {group} is absent from the tables")
    if hall not in (frozen_hall, int(spglib_hall_numbers[group - 1])):
        raise ValueError(f"Unhandled Hall setting {hall} for group {group}")
    q = (tables.arrays["group_q"][group].copy() if hall != frozen_hall else np.zeros(3))
    lattice = np.asarray(ds.std_lattice, dtype=np.float64)
    positions = np.mod(np.asarray(ds.std_positions, dtype=np.float64) - q, 1.0)
    types = np.asarray(ds.std_types, dtype=np.int64)
    assert_frozen_cell(lattice, positions, types, group, tables)

    transform = np.asarray(ds.transformation_matrix, dtype=np.float64)
    origin = np.asarray(ds.origin_shift, dtype=np.float64)
    rotation = np.asarray(ds.std_rotation_matrix, dtype=np.float64)
    original_in_standard = cell[1] @ transform.T + origin - q
    # Primitive labels retain atom identity even for centred cells and input supercells.
    ideal_lattice = Lattice(lattice)
    distances = ideal_lattice.get_all_distances(original_in_standard, positions)
    same = ((np.asarray(ds.mapping_to_primitive)[:, None]
             == np.asarray(ds.std_mapping_to_primitive)[None])
            & (cell[2][:, None] == types[None]))
    displacements = np.where(same, distances, np.inf).min(axis=1)
    if not np.isfinite(displacements).all():
        raise ValueError("spglib primitive mappings do not identify the original sites")
    raw_conventional = np.linalg.solve(transform.T, cell[0]) @ rotation.T
    stretches = np.linalg.svd(np.linalg.solve(raw_conventional, lattice), compute_uv=False)
    return Standardised(
        group, hall, lattice, positions, types, transform, origin, rotation, q,
        float(displacements.max()), float(displacements.mean()),
        float(np.linalg.norm(stretches - 1.0)),
    )
