"""Release snapshots and a complete tiny bundle; no real release artifacts are read."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from orbitol import contracts as C
from orbitol import eval_snapshot
from orbitol.data import cache_content_sha256, file_sha256
from orbitol.sample import sample_from_checkpoint
from tools.release import make_bundle as bundle
from test_train import cfg, equal, fixture_data, trainer  # noqa: F401


def payload():
    return dict(format_version=1, cfg={"dataset": "mp20", "historical_setting": 3}, step=17,
                model={"weight": torch.zeros(4), "buffer": torch.arange(3)},
                ema={"weight": torch.ones(4)}, table_sha256="table", encoding_sha256="encoding",
                core_manifest={"pin": "core"}, commit="commit", bank_sha256="bank",
                data_hashes={"train": "cache"}, allowed_mask=torch.ones(4, dtype=torch.bool),
                nequip_model_sha256="relaxer", phase_diagram_sha256="hull",
                optimizer={"large": torch.ones(30)}, scheduler={"step": 17}, rng_state="discard",
                trainer_state={"loss": 123})


def test_ema_only_preserves_metadata_and_aliases_without_mutating(tmp_path):
    original = payload()
    result = eval_snapshot.ema_only(original)
    assert set(result) == set(eval_snapshot.SNAPSHOT_KEYS) | {"model", "ema"}
    assert result["cfg"] is original["cfg"]
    assert result["ema"] is original["ema"]
    assert result["model"]["weight"] is original["ema"]["weight"]
    assert torch.equal(result["model"]["buffer"], original["model"]["buffer"])
    assert torch.equal(original["model"]["weight"], torch.zeros(4))
    # Reproduce the old writer's payload, which already had no trainer state.
    sampling = {key: original[key] for key in (*eval_snapshot.SNAPSHOT_KEYS, "model", "ema")}
    expected = dict(sampling, model={name: sampling["ema"].get(name, value.detach().cpu())
                                    for name, value in sampling["model"].items()}, step=23)
    path = eval_snapshot.write_eval_snapshot(SimpleNamespace(snapshot=lambda: sampling), tmp_path, 23)
    saved = torch.load(path, weights_only=False)
    equal(saved, expected)
    assert saved["model"]["weight"].data_ptr() == saved["ema"]["weight"].data_ptr()


def test_content_digest_ignores_manifest_order_compression_and_byte_order(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    values = np.arange(6, dtype="<i4").reshape(2, 3)
    np.savez(a, z=values, a=np.array([True]), manifest="first machine")
    np.savez_compressed(b, manifest="another path/time", a=np.array([True]), z=values.astype(">i4"))
    digest = cache_content_sha256(a)
    assert digest == cache_content_sha256(b)
    assert file_sha256(a) != file_sha256(b)
    expected = hashlib.sha256()
    for name, arr in (("a", np.array([True])), ("z", values)):
        header = json.dumps([name, list(arr.shape), arr.dtype.str], separators=(",", ":")).encode()
        expected.update(len(header).to_bytes(8, "little") + header + arr.tobytes())
    assert digest == expected.hexdigest()
    for changed in (values + 1, values.reshape(3, 2), values.astype("<i8")):
        np.savez(b, z=changed, a=np.array([True]), manifest="first machine")
        assert digest != cache_content_sha256(b)


def register_source(monkeypatch, source, state, name="tiny"):
    models = [dict(name=name, dataset=state["cfg"].get("dataset", "mp20"), step=state["step"],
                   bank_sha256=state["bank_sha256"], source_sha256=file_sha256(source),
                   source_size=source.stat().st_size,
                   snapshot=f"checkpoints/{name}/step_{state['step']:07d}.pt")]
    monkeypatch.setattr(bundle, "paper_models", lambda: [dict(model) for model in models])
    return models


def test_export_checks_identity_and_keeps_only_sampling_state(tmp_path, monkeypatch):
    state = payload()
    state["cfg"].update(eval_partition="our-gpus", eval_account="our-group", eval_gres="gpu:1")
    source = tmp_path / "full.pt"
    torch.save(state, source)
    register_source(monkeypatch, source, state)
    calls = []
    monkeypatch.setattr(bundle, "load_snapshot", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    target = bundle.export(source, tmp_path / "bundle")
    assert target.name == "step_0000017.pt"
    saved = torch.load(target, weights_only=False)
    expected = eval_snapshot.ema_only(state)
    expected["cfg"] = {**state["cfg"], "eval_partition": "", "eval_account": ""}
    equal(saved, expected)
    assert state["cfg"]["eval_partition"] == "our-gpus"
    assert calls[0][1]["device"] == "cpu"
    source.write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="Not an exact source"):
        bundle.export(source, tmp_path / "bundle")


def test_export_samples_identically_to_full_checkpoint(tmp_path, monkeypatch, cfg, fixture_data):
    run = trainer(replace(cfg, eval_mode="async"), fixture_data, "release-snapshot")
    try:
        with torch.no_grad():
            next(run.model.parameters()).add_(1.)
        state = run.snapshot()
        state["optimizer"] = {"discard": torch.ones(2)}
        source = tmp_path / "full.pt"
        torch.save(state, source)
        register_source(monkeypatch, source, state)
        monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
        snapshot = bundle.export(source, tmp_path / "bundle")
        expected = sample_from_checkpoint(source, n=2, seed=123, device="cpu", chunk=2)
        actual = sample_from_checkpoint(snapshot, n=2, seed=123, device="cpu", chunk=2)
        for key in ("rows", "requested_G", "skeletons", "raw", "contains_a", "failures"):
            equal(expected[key], actual[key])
    finally:
        run.close()


@pytest.fixture
def tiny_bundle(tmp_path, monkeypatch):
    home, out = tmp_path / "inputs", tmp_path / "release"
    monkeypatch.setenv("ORBITOL_LOCAL_ENV", str(tmp_path / "absent.env"))
    monkeypatch.setenv("ORBITOL_DATA_HOME", str(home))
    for key in C.DATA_LAYOUT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(C, "DATASETS", C._DatasetRegistry())
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    # Include an explicit per-file override to verify source resolution uses local config.
    monkeypatch.setenv("ORBITOL_MP20_SYMMETRY_REFERENCE", str(tmp_path / "custom/symmetry_reference.json"))
    models = bundle.paper_models()
    for model in models:
        path = out / model["snapshot"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny snapshot")
    for dataset in C.DATASET_NAMES:
        spec = C.dataset_spec(dataset)
        reference = Path(spec.symmetry_reference)
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_text(json.dumps(dict(metadata={"train_cache": "/original/tokens.pt", "mtime_ns": 1},
                                             histograms={"0.1": [1, 2, 3]})))
    for source, _, role, _ in bundle.inventory():
        if role in ("attribution", "paper_config", "symmetry_reference"):
            continue
        source.parent.mkdir(parents=True, exist_ok=True)
        if role == "orbit_cache" and source.suffix == ".npz":
            np.savez(source, row=np.arange(4), manifest="preserve exact bytes")
        elif role in bundle.PATH_RECORD_ROLES and source.suffix == ".json":
            source.write_text(json.dumps(dict(csv_path="/original/raw/train.csv", csv_sha256="csv",
                                              inputs={"lmdb": {"path": "/original/lmdb", "sha256": "db"}},
                                              unit="eV/atom", excluded=["/original/x.cif: failed"])))
        elif role == "relaxer":
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("root/data/0", b"weights")
                archive.writestr("root/extra/types", b"Si")
        else:
            source.write_bytes(b"fixture")
    # Files outside the allowlist must never be copied, including local relaxer build receipts.
    for dataset in C.DATASET_NAMES:
        raw = home / dataset / "raw/train.csv"
        raw.parent.mkdir(parents=True)
        raw.write_text("not for release")
    for key in ("ORBITOL_NEQUIP_TS", "ORBITOL_NEQUIP_OEQ"):
        Path(C.local_path(key) + ".build.json").write_text('{"python": "/local/bin/python"}')
    calls = []

    def load(path, **kwargs):
        model = next(m for m in models if path == out / m["snapshot"])
        assert C.local_path("ORBITOL_TABLES_DIR") == str(out / C.DATA_LAYOUT["ORBITOL_TABLES_DIR"])
        assert C.dataset_spec(model["dataset"]).data_root == str(out / model["dataset"])
        calls.append(path)
        return {"cfg": SimpleNamespace(dataset=model["dataset"])}, {"step": model["step"]}

    monkeypatch.setattr(bundle, "load_snapshot", load)
    monkeypatch.setattr(bundle, "load_tables", lambda _: SimpleNamespace(sha256="table-content"))
    return home, out, calls


def test_assemble_complete_manifest_checksums_and_unchanged_inputs(tiny_bundle):
    home, out, calls = tiny_bundle
    original = {source: source.read_bytes() for source, _, _, _ in bundle.inventory()}
    before = C.dataset_spec("mp20")
    manifest = bundle.assemble(out)
    assert C.dataset_spec("mp20") is before
    assert len(calls) == 1
    assert manifest == json.loads((out / "MANIFEST.json").read_text())
    assert manifest["hosting"] == bundle.HOSTING and manifest["licence"] == "CC-BY-4.0"
    entries = {item["path"]: item for item in manifest["files"]}
    assert not [path for path in entries if "test" in path or path.endswith("_info.txt")]
    checksums = dict(line.split("  ", 1)[::-1] for line in (out / "SHA256SUMS").read_text().splitlines())
    assert set(checksums) == set(entries) | {"MANIFEST.json"}
    assert {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} == set(checksums) | {"SHA256SUMS"}
    for relative, digest in checksums.items():
        assert file_sha256(out / relative) == digest
        if relative in entries:
            assert entries[relative]["sha256"] == digest
            assert entries[relative]["size"] == (out / relative).stat().st_size
    for source, relative, role, dataset in bundle.inventory():
        assert source.read_bytes() == original[source]
        if role == "symmetry_reference":
            expected = json.loads(original[source])
            expected["metadata"]["train_cache"] = "tokens.pt"
            expected["metadata"]["train_cache_sha256"] = file_sha256(
                out / dataset / "processed/mp20_tokens_train.pt")
            assert json.loads((out / relative).read_text()) == expected
        elif role in bundle.PATH_RECORD_ROLES and relative.endswith(".json"):
            assert json.loads((out / relative).read_text()) == dict(
                csv_path="train.csv", csv_sha256="csv", inputs={"lmdb": {"path": "lmdb", "sha256": "db"}},
                unit="eV/atom", excluded=["x.cif: failed"])
        else:
            assert (out / relative).read_bytes() == original[source]
        if role == "orbit_cache" and relative.endswith(".npz"):
            assert manifest["content_digests"][relative] == cache_content_sha256(source)
    assert not list(out.rglob("*.csv")) and not list(out.rglob("*.build.json"))
    assert not (out / C.DATA_LAYOUT["ORBITOL_MP20_PHASE_DIAGRAM"]).exists()
    for model in manifest["models"]:
        assert model["snapshot_sha256"] == checksums[model["snapshot"]]
    # Completed assembly can be repeated without changing payload bytes.
    bundle.assemble(out)
    assert all(file_sha256(out / path) == digest for path, digest in checksums.items() if path != "MANIFEST.json")


def test_assemble_space_and_missing_input_fail_before_copy(tiny_bundle, monkeypatch):
    _, out, _ = tiny_bundle
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: SimpleNamespace(free=bundle.RESERVE))
    with pytest.raises(OSError, match="20 GiB"):
        bundle.assemble(out)
    assert not (out / "tables").exists()
    monkeypatch.setattr(bundle.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))
    Path(C.dataset_spec("mp20").orbit_cache_dir, "train.npz").unlink()
    with pytest.raises(FileNotFoundError):
        bundle.assemble(out)
    assert not (out / "tables").exists()


def test_assemble_rejects_unlisted_files(tiny_bundle):
    _, out, _ = tiny_bundle
    (out / "unpublished.csv").write_text("excluded")
    with pytest.raises(ValueError, match="Unexpected files"):
        bundle.assemble(out)
    assert not (out / "MANIFEST.json").exists()
