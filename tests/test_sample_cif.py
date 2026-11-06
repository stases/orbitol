"""CIF export of a sample pool (orbitol.sample --cif-dir)."""
import math

from pymatgen.core import Structure
import torch

from orbitol import sample


def test_write_cifs_writes_decoded_structures_and_skips_failures(tmp_path):
    a = 4.12
    y1 = torch.tensor([math.log(a)] * 3 + [0.0] * 3)  # cubic: log lengths, cos angles
    rows = dict(A0=torch.tensor([[55, 17], [0, 0], [55, 0]]),
                F1=torch.tensor([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]] * 3),
                Y1=y1.expand(3, 6),
                pad_mask=torch.tensor([[False, False], [True, True], [False, False]]))
    pool = dict(rows=rows, failures={1: "decode"})

    assert sample.write_cifs(pool, tmp_path / "cifs") == 1
    assert [p.name for p in (tmp_path / "cifs").iterdir()] == ["0.cif"]
    structure = Structure.from_file(tmp_path / "cifs" / "0.cif")
    assert structure.composition.reduced_formula == "CsCl"
    assert math.isclose(structure.lattice.a, a, rel_tol=1e-5)
