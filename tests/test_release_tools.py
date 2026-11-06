"""Offline release transfers and verification with synthetic payloads."""
import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest
import torch

from orbitol import contracts as C
from orbitol.data import cache_content_sha256
from tools.release import fetch, verify


@pytest.fixture(autouse=True)
def local_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOL_LOCAL_ENV", str(tmp_path / "missing.env"))
    monkeypatch.setenv("ORBITOL_DATA_HOME", str(tmp_path / "home"))
    for key in C.DATA_LAYOUT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(C, "DATASETS", C._DatasetRegistry())
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda _: SimpleNamespace(free=10**12))

    def no_network(*args, **kwargs):
        pytest.fail("Unexpected network access")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", no_network)
    return tmp_path / "home"


def spec(data, **extra):
    return dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), **extra)


def response(data, status=200, headers=None):
    stream = io.BytesIO(data)
    stream.status, stream.headers = status, headers or {}
    return stream


def test_bundle_placement_skip_and_dry_run(tmp_path, local_home, monkeypatch, capsys):
    bundle = tmp_path / "bundle"
    source = bundle / "mp20/orbit_cache/train.npz"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cache")
    manifest = dict(files=[spec(b"cache", path="mp20/orbit_cache/train.npz", role="orbit_cache")])
    (bundle / "MANIFEST.json").write_text(json.dumps(manifest))
    override = tmp_path / "custom-cache"
    monkeypatch.setenv("ORBITOL_MP20_ORBIT_CACHE_DIR", str(override))
    monkeypatch.setenv("ORBITOL_MP20_DATA_ROOT", str(tmp_path / "other-mp20"))
    fetch.fetch_bundle(bundle, dry_run=True)
    assert str(override / "train.npz") in capsys.readouterr().out
    assert not override.exists() and not local_home.exists()
    fetch.fetch_bundle(bundle)
    assert source.read_bytes() == (override / "train.npz").read_bytes() == b"cache"
    assert json.loads((local_home / "MANIFEST.json").read_text()) == manifest
    assert fetch.destination("mp20/raw/train.csv") == tmp_path / "other-mp20/raw/train.csv"
    assert fetch.destination("checkpoints/model/step_0000001.pt") == local_home / "checkpoints/model/step_0000001.pt"
    # A missing source is harmless if the installed checksum already matches.
    source.unlink()
    fetch.fetch_bundle(bundle)


@pytest.mark.parametrize("range_supported", [True, False])
def test_resume_or_restart(tmp_path, monkeypatch, range_supported):
    data = b"download bytes"
    target = tmp_path / "payload"
    target.with_name("payload.part").write_bytes(data[:4])

    def urlopen(request, *, context):
        assert context.verify_mode and request.get_header("Range") == "bytes=4-"
        return (response(data[4:], 206, {"Content-Range": f"bytes 4-{len(data)-1}/{len(data)}"})
                if range_supported else response(data))

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    fetch.fetch_file("https://example.test/file", target, spec(data))
    assert target.read_bytes() == data
    assert not target.with_name("payload.part").exists()
    fetch.fetch_file("https://example.test/file", target, spec(data))


def test_mismatch_force_and_bad_range(tmp_path, monkeypatch):
    target = tmp_path / "payload"
    target.write_bytes(b"old")
    with pytest.raises(ValueError, match="Refusing"):
        fetch.fetch_file("https://example.test/file", target, spec(b"new"))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: response(b"bad"))
    with pytest.raises(ValueError, match="checksum"):
        fetch.fetch_file("https://example.test/file", target, spec(b"new"), force=True)
    assert target.read_bytes() == b"old"
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: response(b"new"))
    fetch.fetch_file("https://example.test/file", target, spec(b"new"), force=True)
    assert target.read_bytes() == b"new"
    target.unlink()
    target.with_name("payload.part").write_bytes(b"n")
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: response(b"ew", 206, {
        "Content-Range": "bytes 0-1/3"}))
    with pytest.raises(ValueError, match="Content-Range"):
        fetch.fetch_file("https://example.test/file", target, spec(b"new"))
    assert not target.exists()


def test_gunzip_checks_both_layers(local_home, monkeypatch):
    data = b"a tiny synthetic phase diagram"
    compressed = gzip.compress(data)
    item = dict(url="https://example.test/hull.gz", path="hulls/mp20/2023-02-07-ppd-mp.pkl",
                size=len(compressed), md5=hashlib.md5(compressed).hexdigest(), gunzip=spec(data))
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: response(compressed))
    fetch.fetch_public(item)
    target = fetch.destination(item["path"])
    assert target.read_bytes() == data
    item["gunzip"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Gunzipped"):
        fetch.fetch_public(item, force=True)
    assert target.read_bytes() == data
    item["md5"] = "0" * 32
    with pytest.raises(ValueError, match="checksum"):
        fetch.fetch_public(item, force=True)


def test_remote_manifest_and_traversal(local_home, monkeypatch):
    manifest = dict(files=[spec(b"x", path="tables/v1/tables.npz", role="tables")])
    urls = []

    def urlopen(request, **kwargs):
        urls.append(request.full_url)
        return response(json.dumps(manifest).encode() if "MANIFEST.json" in request.full_url else b"x")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", urlopen)
    fetch.fetch_bundle("https://example.test/release/")
    assert urls == ["https://example.test/release/MANIFEST.json", "https://example.test/release/tables/v1/tables.npz"]
    for invalid in ("../escape", "/escape", "mp20/../../escape", "a/../b", "a\\b"):
        with pytest.raises(ValueError, match="Invalid bundle path"):
            fetch.destination(invalid)


def test_nequip_source_uses_single_registry():
    from tools.build_nequip import SOURCE
    item, = fetch.public_inputs()["nequip-source"]
    assert all(item[key] == value for key, value in SOURCE.items())


def test_data_exact_content_wrong_missing(local_home, monkeypatch):
    from orbitol import tables
    local_home.mkdir()
    cache = fetch.destination("mp20/orbit_cache/train.npz")
    cache.parent.mkdir(parents=True)
    np.savez(cache, row=np.arange(3), manifest="old path")
    original = spec(cache.read_bytes(), path="mp20/orbit_cache/train.npz", role="orbit_cache")
    content = cache_content_sha256(cache)
    manifest = dict(files=[original], content_digests={original["path"]: content})
    assert verify.verify_data(manifest)["files"][0]["status"] == "exact"
    np.savez_compressed(cache, row=np.arange(3), manifest="new path")
    assert verify.verify_data(manifest)["files"][0]["status"] == "content"
    np.savez(cache, row=np.arange(4), manifest="new path")
    assert verify.verify_data(manifest)["files"][0]["status"] == "wrong"
    cache.unlink()
    assert verify.verify_data(manifest)["files"][0]["status"] == "missing"
    table = fetch.destination("tables/v1/tables.npz")
    table.parent.mkdir(parents=True)
    table.write_bytes(b"rebuilt")
    monkeypatch.setattr(tables, "load_tables", lambda path: SimpleNamespace(sha256="arrays"))
    manifest = dict(files=[spec(b"original", path="tables/v1/tables.npz", role="tables")],
                    content_digests={"tables": "arrays"})
    assert verify.verify_data(manifest)["files"][0]["status"] == "content"
    manifest["content_digests"]["tables"] = "wrong"
    assert not verify.verify_data(manifest)["ok"]


def test_data_symmetry_and_relaxer(local_home, monkeypatch):
    import spglib
    from orbitol.symmetry_panel import SYMPRECS, ANGLE_TOLERANCE
    from tools.build_nequip import weights_digest
    monkeypatch.setattr(verify, "public_inputs", lambda: {"mp20-raw": [], "mp20-hull": []})
    token = fetch.destination("mp20/processed/mp20_tokens_train.pt")
    token.parent.mkdir(parents=True)
    torch.save([{}], token)
    ref = fetch.destination("mp20/symmetry_reference.json")
    metadata = dict(schema=1, train_cache="/old/tokens.pt", size=token.stat().st_size, mtime_ns=0, count=1,
                    spglib=spglib.__version__, symprecs=list(SYMPRECS), angle_tolerance=ANGLE_TOLERANCE,
                    decoder="crystalite-03603fe:tokens_to_structure", train_cache_sha256=fetch.digest(token))
    ref.write_text(json.dumps(dict(metadata=metadata, histograms={str(p): [1] + [0]*230 for p in SYMPRECS})))
    ts = fetch.destination(C.DATA_LAYOUT["ORBITOL_NEQUIP_TS"])
    ts.parent.mkdir(parents=True)
    with zipfile.ZipFile(ts, "w") as archive:
        archive.writestr("root/data/0", b"weights")
    manifest = dict(files=[spec(b"original", path="mp20/symmetry_reference.json", role="symmetry_reference", origin="mp20"),
                               spec(b"original", path=C.DATA_LAYOUT["ORBITOL_NEQUIP_TS"], role="relaxer")],
                    nequip=dict(weights_digest=dict(torchscript=weights_digest(ts))))
    result = verify.verify_data(manifest)
    assert result["ok"] and [r["status"] for r in result["files"]] == ["content", "content"]
    torch.save([{}, {}], token)
    assert not verify.verify_data(manifest)["ok"]


def pool():
    return dict(rows={"A0": torch.tensor([[1, 2], [3, 4]])}, requested_G=torch.tensor([1, 2]),
                skeletons=[dict(G=1, rows=(1,)), dict(G=2, rows=(2,))], raw={"x": torch.zeros(2, 3)},
                failures={1: "failure"}, metadata={"snapshot": "/old/snapshot"})


def test_pools_equal_and_first_difference(tmp_path):
    left, right = tmp_path / "pool", tmp_path / "sweep.samples.pt"
    left.mkdir()
    a, b = pool(), pool()
    b["metadata"] = {"snapshot": "/new/snapshot"}
    torch.save(a, left / "samples.pt")
    torch.save(b, right)
    assert verify.verify_pools(left, right)["equal"]
    b["rows"]["A0"][1, 0] = 7
    torch.save(b, right)
    result = verify.verify_pools(left, right)
    assert result["first_difference"] == dict(key="rows.A0", index=[1, 0], reason="value")
    assert result["differing_structures"] == 1 and result["structures"] == len(b["rows"]["A0"])
    assert not result["ok"] and verify.verify_pools(left, right, max_differing=1)["ok"]
    b = pool()
    b["skeletons"][1]["rows"] = (9,)
    torch.save(b, right)
    assert verify.verify_pools(left, right)["first_difference"] == dict(key="skeletons.rows", index=[1, 0], reason="value")
    a = {key: value for key, value in pool().items() if key in ("rows", "requested_G", "failures")}
    torch.save(a, left / "samples.pt")
    torch.save(a, right)
    assert verify.verify_pools(left, right)["equal"]
    a["failures"][1] = "different"
    torch.save(a, right)
    assert verify.verify_pools(left, right)["first_difference"] == dict(key="failures", index=[1], reason="value")


def test_pool_tolerance_covers_only_generated_differences(tmp_path):
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"
    torch.save(pool(), left)

    def compare(b, max_differing):
        torch.save(b, right)
        return verify.verify_pools(left, right, max_differing)

    b = pool()  # One attempt differs in its rows, its raw output and its decode outcome.
    b["rows"]["A0"][0, 0] = 7
    b["raw"]["x"][0, 1] = 1.0
    b["failures"][0] = "failure"
    assert compare(b, 1)["ok"] and compare(b, 1)["differing_structures"] == 1 and not compare(b, 0)["ok"]
    changes = dict(requested_G=lambda p: p["requested_G"].fill_(230), missing_row=lambda p: p["rows"].pop("A0"),
                   row_dtype=lambda p: p["rows"].update(A0=p["rows"]["A0"].float()),
                   outcome_only=lambda p: p["failures"].update({0: "failure"}))
    for name, change in changes.items():
        b = pool()
        change(b)
        for max_differing in (0, 10):
            result = compare(b, max_differing)
            assert not result["ok"] and result["differing_structures"] is None, (name, max_differing)


def test_pool_check_is_bitwise_and_rejects_malformed_pools(tmp_path):
    left, right = tmp_path / "left.pt", tmp_path / "right.pt"

    def compare(a, b, max_differing=10):
        torch.save(a, left)
        torch.save(b, right)
        return verify.verify_pools(left, right, max_differing)

    b = pool()
    b["raw"]["x"][0, 0] = -0.0  # Numerically equal to +0.0, but not bitwise.
    assert not compare(pool(), b, 0)["ok"] and compare(pool(), b, 1)["differing_structures"] == 1
    a, b = pool(), pool()
    a["contains_a"], b["contains_a"] = torch.ones(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool)
    b["contains_a"].view(torch.uint8)[0] = 2  # Still True, but a different stored byte.
    assert not compare(a, b, 0)["ok"] and not compare(a, b)["ok"]
    malformed = dict(failure_counts=lambda p: p.update(failure_counts={"failure": 2}),
                     extra_key=lambda p: p.update(extra=torch.ones(2)),
                     no_failures=lambda p: p.pop("failures"),
                     float_failure_key=lambda p: p.update(failures={1.0: "failure"}),
                     nan=lambda p: p["raw"]["x"].fill_(float("nan")),
                     short_rows=lambda p: p["rows"].update(A0=p["rows"]["A0"][:1]),
                     short_skeletons=lambda p: p["skeletons"].pop(),
                     float_skeleton=lambda p: p["skeletons"][0].update(G=1.0),
                     complex_raw=lambda p: p["raw"].update(x=torch.zeros(2, 3, dtype=torch.complex64)),
                     int_row_key=lambda p: p["rows"].update({1: torch.zeros(2, 2)}),
                     bool_raw_key=lambda p: p["raw"].update({True: torch.zeros(2, 2)}))
    for name, change in malformed.items():
        a, b = pool(), pool()
        change(a)
        change(b)  # Equal malformed pools fail too.
        result = compare(a, b)
        assert not result["ok"] and not result["equal"] and result["invalid"], name


def test_table12_and_json_exit_status(tmp_path, capsys):
    model = json.loads((verify.ROOT / "configs/paper/models.json").read_text())[0]
    expected = model["expected"]["table12"]
    row = dict(seed=2026, samples=expected["n"], checkpoint_step=model["step"],
               sampler=dict(expected["chosen"]["overrides"], decode_impl="fast"),
               overrides=expected["chosen"]["overrides"],
               logged={"eval/MSUN": .4506, "eval/un_rate": .7706541308261652, "eval/MSUN_oneshot": .3158})
    path = tmp_path / "chosen_seed2026.json"
    path.write_text(json.dumps(row))
    assert verify.main(["table12", str(tmp_path), "--partial"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    # Without --partial, the missing paper seeds fail the check.
    assert verify.main(["table12", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["missing_seeds"] == {"chosen": [2027, 2028]}
    row["logged"]["eval/MSUN"] += .0002  # Relaxation noise on the same GPU type passes by default.
    path.write_text(json.dumps(row))
    assert verify.verify_table12(tmp_path, partial=True)["ok"]
    row["logged"]["eval/MSUN"] += .02
    path.write_text(json.dumps(row))
    assert verify.main(["table12", str(tmp_path), "--partial"]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]
    assert verify.verify_table12(tmp_path, tolerance=.021, partial=True)["ok"]
    path.unlink()
    assert not verify.verify_table12(tmp_path, partial=True)["ok"]
    assert verify.main(["data", "--manifest", str(tmp_path / "missing")]) == 1
    assert not json.loads(capsys.readouterr().out)["ok"]


def test_public_data_sha_and_checkpoints(local_home, monkeypatch, capsys):
    from orbitol import sample
    path = fetch.destination("mp20/raw/train.csv")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"csv")
    monkeypatch.setattr(verify, "public_inputs", lambda: {
        "mp20-raw": [spec(b"csv", path="mp20/raw/train.csv")], "mp20-hull": []})
    assert verify.verify_data(dict(files=[]))["ok"]
    path.unlink()
    assert verify.verify_data(dict(files=[]))["files"][0]["status"] == "missing"
    path.write_bytes(b"bad")
    assert verify.verify_data(dict(files=[]))["files"][0]["status"] == "wrong"
    snapshot = fetch.destination("checkpoints/tiny/step_0000007.pt")
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"snapshot")
    model = dict(name="tiny", snapshot="checkpoints/tiny/step_0000007.pt", step=7, dataset="mp20",
                 snapshot_sha256=fetch.digest(snapshot))
    calls = []

    def load(path, *, device, overrides):
        assert path == snapshot and device == "cpu" and overrides["compile_trunk"] == "none"
        calls.append(path)
        return {"cfg": SimpleNamespace(dataset="mp20")}, {"step": 7}

    monkeypatch.setattr(sample, "load_snapshot", load)
    assert verify.verify_checkpoints(dict(models=[model]))["ok"] and len(calls) == 1
    wrong = dict(model, step=8)
    assert not verify.verify_checkpoints(dict(models=[wrong]))["ok"]
    model["snapshot_sha256"] = "0" * 64
    assert not verify.verify_checkpoints(dict(models=[model]))["ok"]
    assert len(calls) == 2  # Never unpickle a snapshot with the wrong hash.
    manifest_path = local_home / "MANIFEST.json"
    model["snapshot_sha256"] = fetch.digest(snapshot)
    manifest_path.write_text(json.dumps(dict(models=[model])))

    def noisy_loader(*args, **kwargs):
        print("loader progress")
        return load(*args, **kwargs)

    monkeypatch.setattr(sample, "load_snapshot", noisy_loader)
    assert verify.main(["checkpoints", "--manifest", str(manifest_path)]) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] and "loader progress" in captured.err
    snapshot.unlink()
    assert verify.verify_checkpoints(dict(models=[model]))["models"][0]["status"] == "missing"


def test_stored_table12_and_invalid_values(tmp_path):
    model = json.loads((verify.ROOT / "configs/paper/models.json").read_text())[0]
    table = model["expected"]["table12"]
    row = dict(seed=2027, samples=table["n"], checkpoint_step=model["step"],
               sampler=dict(table["stored"]["overrides"], decode_impl="fast"),
               overrides=table["stored"]["overrides"], logged={"eval/MSUN": .4428})
    path = tmp_path / "arbitrary-name.json"
    path.write_text(json.dumps(row))
    assert verify.verify_table12(tmp_path, partial=True)["ok"]
    for field, value in (("seed", 123), ("samples", 50), ("checkpoint_step", 8)):
        bad = dict(row, **{field: value})
        path.write_text(json.dumps(bad))
        assert not verify.verify_table12(tmp_path, partial=True)["ok"]
    bad = copy.deepcopy(row)
    bad["logged"]["eval/MSUN"] = float("nan")
    path.write_text(json.dumps(bad))
    assert not verify.verify_table12(tmp_path, partial=True)["ok"]
    with pytest.raises(ValueError, match="Tolerance"):
        verify.verify_table12(tmp_path, -1)
