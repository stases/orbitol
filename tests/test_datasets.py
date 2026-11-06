"""Dataset routing and safe MatterGen parsing, with small CPU fixtures."""
import csv
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from orbitol import contracts as C, data, sample, symmetry_panel
from orbitol.posthoc_eval import parse_args, resolve_snapshot_paths
from orbitol.tables import build_tables
from src.data.mp20_tokens import tokens_to_structure


def test_registry_and_legacy_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBITOL_LOCAL_ENV", str(tmp_path / "absent.env"))
    monkeypatch.setenv("ORBITOL_DATA_HOME", str(tmp_path))
    for key in C.DATA_LAYOUT:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(C, "DATASETS", C._DatasetRegistry())
    spec = C.dataset_spec()
    root = str(tmp_path / "mp20")
    assert spec.data_root == C.DATA_ROOT == root
    assert spec.orbit_cache_dir == C.ORBIT_CACHE_DIR == root + "/orbit_cache"
    assert spec.atom_cache_dir == root + "/atom_cache"
    assert spec.symmetry_reference == C.SYMMETRY_REFERENCE == root + "/symmetry_reference.json"
    assert spec.phase_diagram == C.PHASE_DIAGRAM == (
        str(tmp_path / "hulls/mp20/2023-02-07-ppd-mp.pkl"))
    assert C.resolve_dataset_paths() == C.resolve_dataset_paths({}) == spec
    assert C.resolve_dataset_paths(C.TrainConfig()) == spec
    assert C.dataset_spec("alex_mp20").data_root == str(tmp_path / "alex_mp20")
    assert C.dataset_spec("alex_mp20").phase_diagram == (
        str(tmp_path / "hulls/alex_mp2020/variants/v1.pkl"))
    assert C.TrainConfig().wandb_project == ""
    assert [C.dataset_spec(name).wandb_project for name in ("mp20", "alex_mp20")] == ["orbitol", "orbitol_alex_mp20"]
    with pytest.raises(FrozenInstanceError):
        spec.name = "alex_mp20"
    with pytest.raises(ValueError, match="Unknown dataset"):
        C.TrainConfig(dataset="unknown")


def test_resolver_precedence_and_independent_overrides(tmp_path):
    overrides = {key: str(tmp_path / key) for key in (
        "data_root", "cache_dir", "atom_cache_dir", "symmetry_reference", "phase_diagram")}
    cfg = C.TrainConfig(dataset="alex_mp20", extra=overrides)
    paths = C.resolve_dataset_paths(cfg)
    assert paths.name == "alex_mp20" and paths.csv_format == "mattergen"
    for key, value in overrides.items():
        assert getattr(paths, "orbit_cache_dir" if key == "cache_dir" else key) == value
    cli = C.resolve_dataset_paths(cfg, overrides={"data_root": tmp_path / "cli", "cache_dir": None})
    assert cli.data_root == str(tmp_path / "cli") and cli.orbit_cache_dir == overrides["cache_dir"]
    paths = C.resolve_dataset_paths({"dataset": "alex_mp20", "extra": {"data_root": "/input"}})
    assert paths.orbit_cache_dir == C.dataset_spec("alex_mp20").orbit_cache_dir


@pytest.mark.parametrize("kind,key", [("orbitol", "dataset"), ("crystalite", "dataset_name")])
def test_snapshot_paths_and_cli_overrides(kind, key, tmp_path):
    argv = ["--kind", kind, "--snapshot", str(tmp_path / "step_0000001.pt"), "--out", str(tmp_path)]
    args = parse_args(argv)
    spec = C.dataset_spec("alex_mp20")
    resolved = resolve_snapshot_paths(args, {key: "alex_mp20"})
    assert resolved.data_root == Path(spec.data_root)
    assert resolved.symmetry_reference == Path(spec.symmetry_reference)
    assert resolved.ppd_path == Path(spec.phase_diagram)
    assert args.data_root is None  # resolving one queue run cannot affect another
    assert resolve_snapshot_paths(args, {}).data_root == Path(C.DATA_ROOT)
    args = parse_args(argv + ["--data-root", str(tmp_path / "cli"),
                              "--symmetry-reference", str(tmp_path / "reference"),
                              "--ppd-path", str(tmp_path / "hull")])
    resolved = resolve_snapshot_paths(args, {key: "alex_mp20", "extra": {"data_root": "/extra"}})
    assert resolved.data_root == tmp_path / "cli"
    assert resolved.symmetry_reference == tmp_path / "reference" and resolved.ppd_path == tmp_path / "hull"
    if kind == "orbitol":
        resolved = resolve_snapshot_paths(parse_args(argv), {key: "alex_mp20", "extra": {"data_root": "/extra"}})
        assert resolved.data_root == Path("/extra")


@pytest.mark.parametrize("symbol,number", [
    ("R-3", 148), ("R -3:H", 148), ("R-3:R", 148), ("P 21/c", 14),
    ("P2_1/c", 14), ("P 1 2_1/c 1", 14), ("P 2₁/c", 14), ("F m -3 m", 225),
])
def test_space_group_symbols(symbol, number):
    assert data.space_group_number(symbol) == number


@pytest.mark.parametrize("symbol", ["", "not-a-group", "P999", "R-3:unknown", "14"])
def test_unknown_symbols_fail(symbol):
    with pytest.raises(ValueError, match="Unknown or ambiguous"):
        data.space_group_number(symbol)


@pytest.fixture
def mattergen_row():
    return dict(material_id="fixture", space_group="P1", atomic_numbers="[3 11 8]",
                cell="[array([4.1, 0, 0])\n array([.8, 5.2, 0]) array([.3, .7, 6.3])]",
                positions="[array([.1, .2, -.3])\n array([1.5, 2.6, 3.7]) array([3., 4., 5.e0])]",
                chemical_system="ignored")


def test_mattergen_reader_and_failure_accounting(mattergen_row, tmp_path):
    source = tmp_path / "tiny.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(mattergen_row))
        writer.writeheader()
        writer.writerow(mattergen_row)
        writer.writerow({**mattergen_row, "material_id": "bad-symbol", "space_group": "invalid"})
        writer.writerow({**mattergen_row, "material_id": "bad-array", "positions": "[array([nan,0,0])]"})
    records = list(data.read_csv_records(source, csv_format="mattergen"))
    structure = records[0]["structure"]
    assert records[0]["material_id"] == "fixture" and records[0]["spacegroup.number"] == 1
    assert list(structure.atomic_numbers) == [3, 11, 8]
    np.testing.assert_allclose(structure.cart_coords, [[.1, .2, -.3], [1.5, 2.6, 3.7], [3, 4, 5]])
    assert structure.volume == pytest.approx(4.1 * 5.2 * 6.3)
    for malformed in ("[array([1, 2, 3]]", "[array([1, 2, 3]) ...]", "[array([nan, 2, 3])]",
                      "[__import__('os').system('false')]"):
        with pytest.raises(ValueError):
            data.parse_mattergen_record({**mattergen_row, "cell": malformed})
    tables_dir = tmp_path / "tables"
    build_tables(groups=[1]).save(tables_dir)
    out = tmp_path / "cache"
    with pytest.raises(RuntimeError, match="bad-symbol[\\s\\S]*bad-array"):
        data.build_cache("val", tables_dir, out, csv_path=source, dataset="alex_mp20")
    assert not out.exists()
    manifest = data.build_cache("val", tables_dir, out, csv_path=source,
                                dataset="alex_mp20", max_failures=2)
    assert manifest["count"] == 1 and len(manifest["excluded"]) == 2
    assert manifest["excluded"][0].startswith("bad-symbol:")
    assert manifest["excluded"][1].startswith("bad-array:")


@pytest.fixture(scope="module")
def alex_root():
    root = Path(C.dataset_spec("alex_mp20").data_root)
    if not (root / "raw/val.csv").is_file() or not (root / "processed/mp20_tokens_val.pt").is_file():
        pytest.skip(f"Alex validation inputs absent: {root}; set ORBITOL_DATA_HOME in orbitol.local.env")
    return root


def test_real_alex_csv_matches_tokens(alex_root):
    records = list(data.read_csv_records(alex_root / "raw/val.csv", 20, csv_format="mattergen"))
    items = torch.load(alex_root / "processed/mp20_tokens_val.pt", map_location="cpu",
                       weights_only=False, mmap=True)
    assert len(items) == 67521 and len(records) == 20
    for record, item in zip(records, items):
        assert record["material_id"] == item["mp_id"], "CSV and tokens have different orders"
        a, b = record["structure"], tokens_to_structure(item)
        assert a.composition == b.composition and len(a) <= 20
        assert a.volume == pytest.approx(b.volume, rel=1e-6)


def test_real_alex_build_cache_cli(alex_root, tmp_path, capsys):
    if not Path(C.TABLES_DIR).is_dir():
        pytest.skip(f"Prebuilt symmetry tables absent: {C.TABLES_DIR}")
    data.main(["build-cache", "--dataset", "alex_mp20", "--split", "val", "--limit", "20",
               "--workers", "2", "--out", str(tmp_path)])
    result = json.loads(capsys.readouterr().out)
    manifest = json.loads((tmp_path / "val.manifest.json").read_text())
    assert result["count"] == manifest["count"] == 20 and manifest["excluded"] == []
    assert manifest["csv_path"] == str((alex_root / "raw/val.csv").resolve())


@pytest.mark.parametrize("override", [False, True])
def test_checkpoint_sampling_uses_dataset_bank(monkeypatch, tmp_path, override):
    cfg = {"dataset": "alex_mp20", "extra": {"cache_dir": str(tmp_path)} if override else {}}
    checkpoint = dict(cfg=cfg, core_manifest=json.loads((sample.ROOT / "orbitol/core_manifest.json").read_text()),
                      table_sha256="table", encoding_sha256=C.TYPE_ENCODING_SHA256)
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(sample, "load_tables", lambda path: SimpleNamespace(sha256="table"))
    monkeypatch.setattr(sample, "load_encoding", lambda: None)
    calls = []

    class BankReached(Exception):
        pass

    def bank(path):
        calls.append(path)
        raise BankReached

    monkeypatch.setattr(sample.SkeletonBank, "from_cache", bank)
    with pytest.raises(BankReached):
        sample.sample_from_checkpoint(path, n=1, seed=1, device="cpu")
    expected = tmp_path if override else Path(C.dataset_spec("alex_mp20").orbit_cache_dir)
    assert calls == [expected / "train.npz"]


def test_atom_cache_and_symmetry_cli_use_selected_tokens(monkeypatch, tmp_path):
    root = tmp_path / "inputs"
    (root / "processed").mkdir(parents=True)
    items = [dict(A0=torch.tensor([3]), F1=torch.tensor([[.1, .2, .3]]),
                  Y1=torch.tensor([np.log(4.)] * 3 + [0.] * 3, dtype=torch.float32),
                  pad_mask=torch.tensor([False]), mp_id="alex-fixture")]
    source = root / "processed/mp20_tokens_train.pt"
    torch.save(items, source)
    spec = replace(C.dataset_spec("alex_mp20"), data_root=str(root),
                   atom_cache_dir=str(tmp_path / "atoms"), symmetry_reference=str(tmp_path / "symmetry.json"))
    monkeypatch.setitem(C.DATASETS, "alex_mp20", spec)
    tables = tmp_path / "tables"
    build_tables(groups=[1]).save(tables)
    data.main(["build-cache", "--dataset", "alex_mp20", "--view", "atom", "--split", "train",
               "--tables", str(tables), "--limit", "1"])
    manifest = json.loads((tmp_path / "atoms/train.manifest.json").read_text())
    assert manifest["source_token_path"] == str(source)
    assert manifest["count"] == 1
    assert sample.allowed_elements(C.resolve_dataset_paths("alex_mp20").data_root).sum() == 1
    expected = symmetry_panel.SymmetryPanel(source, items, tmp_path / "scratch.json").reference
    symmetry_panel.main(["build", "--dataset", "alex_mp20"])
    actual = json.loads((tmp_path / "symmetry.json").read_text())
    assert actual["histograms"] == expected
    assert actual == json.loads((tmp_path / "scratch.json").read_text())


@pytest.mark.parametrize("override", [False, True])
def test_evaluator_cli_routes_data_symmetry_and_hull(monkeypatch, tmp_path, override):
    from orbitol import eval_adapter

    spec = replace(C.dataset_spec("alex_mp20"), phase_diagram=str(tmp_path / "alex-hull"))
    monkeypatch.setitem(C.DATASETS, "alex_mp20", spec)
    calls = []

    def evaluator(data_root, **kwargs):
        calls.append((data_root, kwargs))
        return SimpleNamespace(data_root=Path(data_root), device="cpu", tag="sample_ema",
                               sample_count=1, sun_k=0, sample_seed=123, no_thermo_before_steps=20000,
                               evaluate=lambda *args, **kwargs: {})

    monkeypatch.setattr(eval_adapter, "DNGEvaluator", evaluator)
    source = tmp_path / "samples.pt"
    torch.save({}, source)
    args = ["--dataset", "alex_mp20", "--samples", str(source), "--out", str(tmp_path / "out"), "--thermo"]
    if override:
        args += ["--data-root", str(tmp_path / "data"), "--symmetry-reference", str(tmp_path / "sym"),
                 "--ppd-path", str(tmp_path / "hull")]
    eval_adapter.main(args)
    root, kwargs = calls[0]
    assert root == (str(tmp_path / "data") if override else spec.data_root)
    assert kwargs["symmetry_cache"] == (str(tmp_path / "sym") if override else spec.symmetry_reference)
    assert kwargs["thermo"].thermo_ppd_mp == (str(tmp_path / "hull") if override else spec.phase_diagram)
