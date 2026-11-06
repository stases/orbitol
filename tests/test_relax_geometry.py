"""Same-order, periodic relaxation geometry without C distance routines."""
from itertools import product

import numpy as np
from pymatgen.core import Lattice, Structure
import pytest

from orbitol.relax_geometry import relaxation_geometry


def crystal(lattice=None):
    return Structure(lattice if lattice is not None else Lattice.cubic(4),
                     ["Li", "Na"], [[.9, 0, 0], [.4, .5, .5]])


def test_rotation_has_zero_strain():
    initial = crystal(Lattice([[4, 0, 0], [1, 5, 0], [1, 2, 6]]))
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    final = Structure(initial.lattice.matrix @ rotation.T, initial.species, initial.frac_coords)
    result = relaxation_geometry(initial, final)
    assert result["strain"] == pytest.approx(0, abs=1e-14)
    assert result["rms_disp"] == pytest.approx(0, abs=1e-14)


@pytest.mark.parametrize("scale", [.8, 1., 1.2])
def test_isotropic_scale_and_signed_volume(scale):
    initial = crystal()
    final = Structure(initial.lattice.matrix * scale, initial.species, initial.frac_coords)
    result = relaxation_geometry(initial, final, -3., -4.)
    assert result["strain"] == pytest.approx(np.sqrt(3) * abs(.5 * (scale ** 2 - 1)))
    assert result["volume_change_pct"] == pytest.approx(100 * (scale ** 3 - 1))
    assert result["energy_drop_meV"] == 500
    assert result["rms_disp"] == result["max_disp"] == 0


def test_rigid_translation_across_boundary():
    initial = crystal()
    final = Structure(initial.lattice, initial.species, (initial.frac_coords + [.3, -.4, .2]) % 1)
    assert relaxation_geometry(initial, final)["rms_disp"] == pytest.approx(0, abs=1e-14)


def test_one_site_crosses_boundary():
    initial = crystal()
    final = crystal()
    final[0] = "Li", [.1, 0, 0]
    # Site 0 moves +0.2 fractionally; removing the mean gives +/-0.1 for two sites.
    result = relaxation_geometry(initial, final)
    assert result["rms_disp"] == pytest.approx(.4)
    assert result["max_disp"] == pytest.approx(.4)


def test_skew_cell_uses_reduced_lattice_images():
    initial = crystal(Lattice([[4, 0, 0], [3.8, .6, 0], [0, 0, 4]]))
    delta = np.array([[.49, .49, 0], [-.49, -.49, 0]])
    final = Structure(initial.lattice, initial.species, (initial.frac_coords + delta) % 1)
    shifts = np.array(list(product(range(-8, 9), repeat=3)))
    expected = np.linalg.norm((delta[:, None] + shifts) @ initial.lattice.matrix, axis=-1).min(axis=1)
    naive = np.linalg.norm(delta @ initial.lattice.matrix, axis=1)
    assert np.all(expected < naive / 5)
    result = relaxation_geometry(initial, final)
    assert result["rms_disp"] == pytest.approx(np.sqrt(np.mean(expected ** 2)))
    assert result["max_disp"] == pytest.approx(expected.max())


def test_species_order_and_count_must_match():
    initial = crystal()
    assert relaxation_geometry(initial, Structure(initial.lattice, ["Na", "Li"], initial.frac_coords)) is None
    assert relaxation_geometry(initial, Structure(initial.lattice, ["Li"], [[0, 0, 0]])) is None


def test_invalid_geometry_and_missing_energy():
    initial = crystal()
    assert relaxation_geometry(None, initial) is None
    assert relaxation_geometry(initial, initial)["energy_drop_meV"] is None
    assert relaxation_geometry(initial, initial, np.nan, 0.)["energy_drop_meV"] is None
    bad = Structure(initial.lattice, initial.species, [[np.nan, 0, 0], [0, 0, 0]])
    assert relaxation_geometry(initial, bad) is None
    singular = Structure(np.zeros((3, 3)), initial.species, initial.frac_coords)
    assert relaxation_geometry(initial, singular) is None
