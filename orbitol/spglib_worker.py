"""spglib space-group detection on plain cell tuples, run inside isolated worker processes.

Kept free of torch/pymatgen imports so spawned workers start quickly.
"""
import spglib


def detect_cells(cells, symprecs, angle_tolerance):
    """Space-group numbers for each (lattice, frac_coords, numbers) cell at each symprec; 0 is a failure."""
    result = []
    for cell in cells:
        row = []
        for symprec in symprecs:
            try:
                dataset = None if cell is None else spglib.get_symmetry_dataset(
                    cell, symprec=symprec, angle_tolerance=angle_tolerance)
                row.append(int(dataset.number) if dataset is not None else 0)
            except Exception:
                row.append(0)
        result.append(row)
    return result
