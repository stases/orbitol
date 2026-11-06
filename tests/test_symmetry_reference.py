"""Relocated token caches retain valid references without rebuilding them."""
import json
import os
import shutil

from pymatgen.core import Lattice, Structure
import pytest
import torch

from orbitol import symmetry_panel as panel
from orbitol.decode import to_crystalite_rows
from orbitol.train import verify_symmetry_reference


@pytest.fixture
def reference(tmp_path):
    source = tmp_path / "original/processed/mp20_tokens_train.pt"
    source.parent.mkdir(parents=True)
    rows = to_crystalite_rows([Structure(Lattice.cubic(4), ["Si"], [[0, 0, 0]])], 20)
    items = [{key: tensor[0] for key, tensor in rows.items()}]
    torch.save(items, source)
    path = source.parent.parent / "symmetry_reference.json"
    histograms = panel.SymmetryPanel(source, items, path).reference
    saved = json.loads(path.read_text())
    assert saved["metadata"]["train_cache_sha256"] == panel.file_sha256(source)
    return source, items, path, saved, histograms


@pytest.mark.parametrize("mode", ["copied", "changed", "legacy_exact", "legacy_copied"])
def test_reference_content_and_legacy(reference, tmp_path, monkeypatch, mode):
    source, items, path, saved, histograms = reference
    if mode.startswith("legacy"):
        del saved["metadata"]["train_cache_sha256"]
        path.write_text(json.dumps(saved))
        monkeypatch.setattr(panel, "file_sha256", lambda _: pytest.fail("legacy must not hash"))
    if mode != "legacy_exact":
        copied = tmp_path / "copied/processed" / source.name
        copied.parent.mkdir(parents=True)
        shutil.copyfile(source, copied)
        stat = source.stat()
        os.utime(copied, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        copied_reference = copied.parent.parent / path.name
        shutil.copyfile(path, copied_reference)
        source, path = copied, copied_reference
    if mode == "changed":
        items[0]["F1"].add_(.125)
        torch.save(items, source)
        assert source.stat().st_size == saved["metadata"]["size"]
    expected = {k: v for k, v in saved["metadata"].items() if k != "train_cache_sha256"}
    expected.update(train_cache=str(source), mtime_ns=source.stat().st_mtime_ns)
    valid = mode in ("copied", "legacy_exact")
    assert panel.reference_valid(saved, expected, source) is valid
    before = path.read_bytes(), path.stat().st_mtime_ns
    if valid:
        monkeypatch.setattr(panel, "detect_group", lambda *_: pytest.fail("reference rebuilt"))
        verify_symmetry_reference(source.parent.parent, path)
        assert panel.SymmetryPanel(source, items, path).reference == histograms
    else:
        with pytest.raises(ValueError, match="Stale symmetry reference"):
            verify_symmetry_reference(source.parent.parent, path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
