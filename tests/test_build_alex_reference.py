"""Small archive-format fixtures for the common Alex reference table."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import pickle
import warnings

import lmdb
import pyarrow.parquet as pq
import pytest
from pymatgen.analysis.structure_analyzer import oxide_type
from pymatgen.core import Lattice, Structure
from pymatgen.entries.compatibility import CompatibilityError, MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry

import orbitol.core
from src.utils.sample_stats import _build_mp2020_parameters
from tools import build_alex_reference as builder


MG_LEGACY = {"titel": "PAW_PBE Mg_pv 06Sep2000", "hash": "bbcf6f81cc34a3090d483ad641178746"}
MG_NEW = {"titel": "PAW_PBE Mg_pv 13Apr2007", "hash": "7a73c1cd580008de85139c0c254c5bd2"}


def _structure(species):
    return Structure(Lattice.cubic(10), species,
                     [[i / len(species)] * 3 for i in range(len(species))])


def _record(species, entry_id, legacy):
    structure = _structure(species)
    compat = MaterialsProject2020Compatibility(check_potcar=False)
    parameters = _build_mp2020_parameters(structure.composition, compat)
    parameters["potcar_spec"] = [legacy[el] for el in sorted(structure.composition.get_el_amt_dict())]
    entry = ComputedStructureEntry(structure, energy=-4 * len(species), entry_id=entry_id,
                                   parameters=parameters, data={"oxidation_states": {},
                                                               "oxide_type": oxide_type(structure, 1.05)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return compat.process_entry(entry, on_error="raise").as_dict()


def _write_lmdb(path, records):
    with lmdb.open(str(path), subdir=False, map_size=16 * 1024 * 1024) as env:
        with env.begin(write=True) as txn:
            for key, record in records.items():
                txn.put(key.encode(), pickle.dumps(record))
            txn.put(b"Mg-O.MgO.length", pickle.dumps(2))
            txn.put(b"__index__", b"non-pickle index data")
            txn.put(b"__keys__", pickle.dumps(list(records)))


@pytest.fixture
def archive(tmp_path):
    legacy = {el: {"titel": f"PAW_PBE {el} 01Jan2001", "hash": f"legacy-{el}"}
              for el in ("Ag", "Al", "Ce", "Co", "Mg", "Na", "O", "Si")}
    legacy["Mg"] = MG_LEGACY
    elemental = tmp_path / "elemental.json.gz"
    references = {el: _record([el], f"mp-{el}", legacy) for el in legacy}
    with gzip.open(elemental, "wt") as stream:
        json.dump(references, stream)
    shifts = tmp_path / "fixture_shifts.json"
    shifts.write_text(json.dumps({"schema": 1, "elements": {"Mg": {
        "legacy": MG_LEGACY, "new": MG_NEW, "shift": 0.09,
        "status": "applied", "target": "legacy", "reason": "test fixture, not fitted values",
    }}}))
    known_bad = tmp_path / "known_bad.json"
    known_bad.write_text(json.dumps({"entries": [
        {"key": "Co-O.CoO2.34", "action": "exclude"},
        {"key": "Ag-Ce-Si.Ce(SiAg)2.0", "action": "retain_flagged"},
        {"key": None, "action": "retain_flagged", "formula": "MgO"},
    ]}))
    records = {
        "Mg-O.MgO.0": _record(["Mg", "O"], "mp-legacy", legacy),
        "Mg-O.MgO.1": _record(["Mg", "O", "Mg", "O"], "msft-new", legacy),
        "Na-O.Na2O.0": _record(["Na", "Na", "O"], "custom-unlisted", legacy),
        "Ag-Ce-Si.Ce(SiAg)2.0": _record(["Ce", "Si", "Si", "Ag", "Ag"], "mp-silicide", legacy),
        "Co-O.CoO2.34": _record(["Co", "O", "O"], "mp-known-bad", legacy),
    }
    records["Mg-O.MgO.1"]["parameters"]["potcar_spec"][0] = MG_NEW
    records["Na-O.Na2O.0"]["parameters"]["potcar_spec"][0] = {
        "titel": legacy["Na"]["titel"], "hash": "unlisted-Na"}
    records["Mg-O.MgO.0"]["parameters"]["hubbards"] = {"Mg": 0, "O": 0.0}
    lmdb_path = tmp_path / "reference.lmdb"
    _write_lmdb(lmdb_path, records)
    source_gz = tmp_path / "source.gz"
    source_gz.write_bytes(gzip.compress(b"test provenance input", mtime=0))
    return {"lmdb": lmdb_path, "mp_elemental": elemental, "known_bad": known_bad,
            "shifts": shifts, "source_gz": source_gz}


def _rows(path):
    return {row["key"]: row for row in pq.read_table(path / "entries.parquet").to_pylist()}


def _diagnostics(path, manifest):
    payload = (path / "diagnostics.jsonl").read_bytes()
    assert manifest["diagnostics_sha256"] == hashlib.sha256(payload).hexdigest()
    assert "diagnostics" not in manifest
    return [json.loads(line) for line in payload.splitlines()]


def test_archive_fix_sets_and_provenance(archive, tmp_path):
    out = tmp_path / "build"
    manifest = builder.build(**archive, out=out)
    rows = _rows(out)
    assert len(rows) == 5
    assert list(rows) == sorted(rows)
    assert all(row["stored_reproduced"] and row["stored_error"] is None for row in rows.values())
    assert all(not row["synth_params_mismatch"] for row in rows.values())
    assert all(row["oxidation_states_nonempty"] is False for row in rows.values())
    legacy = rows["Mg-O.MgO.0"]
    assert legacy["elements"] == ["Mg", "O"]
    assert legacy["amounts"] == [1, 1]
    assert legacy["natoms"] == 2
    assert legacy["raw_energy"] == -8
    assert legacy["stored_correction"] == pytest.approx(-0.687)
    assert legacy["potcar_shift"] == 0
    assert legacy["nonlegacy_elements"] == []
    assert not legacy["unlisted_nonlegacy"]
    assert legacy["known_bad"] is None
    new = rows["Mg-O.MgO.1"]
    assert new["source"] == "msft"
    assert new["nonlegacy_elements"] == ["Mg"]
    assert new["potcar_shift"] == pytest.approx(0.18)
    assert new["raw_energy"] - new["potcar_shift"] == pytest.approx(-16.18)
    assert not new["unlisted_nonlegacy"]
    unlisted = rows["Na-O.Na2O.0"]
    assert unlisted["source"] == "custom"
    assert unlisted["unlisted_nonlegacy"]
    assert unlisted["nonlegacy_elements"] == ["Na"]
    assert unlisted["potcar_shift"] == 0
    silicide = rows["Ag-Ce-Si.Ce(SiAg)2.0"]
    assert silicide["chemsys"] == "Ag-Ce-Si"
    assert silicide["amounts"] == [2, 1, 2]
    assert silicide["stored_correction"] == 0
    assert silicide["recorrected_correction"] == pytest.approx(0.142)
    assert silicide["recorrect_error"] is None
    assert silicide["known_bad"] == "retain_flagged"
    assert rows["Co-O.CoO2.34"]["known_bad"] == "exclude"
    assert manifest["counts"] == {
        "entries": 5, "stored_reproduced_false": 0, "recorrect_changed": 1, "recorrect_failures": 0,
        "synth_params_mismatch": 0, "oxide_type_changed": 0, "unlisted_nonlegacy": 1,
        "oxidation_states_nonempty": 0,
        "per_source": {"mp": 3, "msft": 1, "custom": 1},
        "known_bad_by_action": {"exclude": 1, "retain_flagged": 1},
        "nonzero_shift_by_element": {"Mg": 1},
    }
    assert manifest == json.loads((out / "manifest.json").read_text())
    assert manifest["legacy_potcars"]["Mg"] == MG_LEGACY
    triples = {(p["element"], p["hash"]): p for p in manifest["potcar_inventory"]}
    assert triples["Mg", MG_LEGACY["hash"]]["status"] == "legacy"
    assert triples["Mg", MG_NEW["hash"]]["status"] == "listed"
    assert triples["Na", "unlisted-Na"]["status"] == "unlisted"
    assert triples["O", "legacy-O"]["count"] == 4
    for name, path in archive.items():
        assert manifest["inputs"][name] == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert manifest["table_sha256"] == hashlib.sha256((out / "entries.parquet").read_bytes()).hexdigest()
    assert len(manifest["git_commit"]) == 40
    assert manifest["pymatgen_version"]
    assert _diagnostics(out, manifest) == []


def test_determinism_with_multiple_ranges_and_processes(archive, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "CHUNK_SIZE", 2)
    with lmdb.open(str(archive["lmdb"]), subdir=False) as env, env.begin(write=True) as txn:
        for key, data in [
            (b"Ag-Ce-Si.Ce(SiAg)2.0", {"oxide_type": "oxide"}),
            (b"Mg-O.MgO.1", {"oxidation_states": {"Mg": 2, "O": -2}}),
        ]:
            record = pickle.loads(txn.get(key))
            record["data"].update(data)
            txn.put(key, pickle.dumps(record))
    a = builder.build(**archive, out=tmp_path / "a", workers=1)
    b = builder.build(**archive, out=tmp_path / "b", workers=2)
    assert pq.read_table(tmp_path / "a/entries.parquet").equals(pq.read_table(tmp_path / "b/entries.parquet"))
    assert (tmp_path / "a/entries.parquet").read_bytes() == (tmp_path / "b/entries.parquet").read_bytes()
    assert _diagnostics(tmp_path / "a", a) == _diagnostics(tmp_path / "b", b) == [
        {"key": "Ag-Ce-Si.Ce(SiAg)2.0", "flags": ["oxide_type_changed"]},
        {"key": "Mg-O.MgO.1", "flags": ["oxidation_states_nonempty"]},
    ]
    assert (tmp_path / "a/diagnostics.jsonl").read_bytes() == (tmp_path / "b/diagnostics.jsonl").read_bytes()
    a.pop("elapsed_seconds")
    b.pop("elapsed_seconds")
    assert a == b


def test_stored_oxidation_states_are_preserved_and_counted(archive, tmp_path):
    legacy = builder._load_legacy(archive["mp_elemental"])
    missing = _record(["Ce", "Si", "Si", "Ag", "Ag"], "mp-missing-states", legacy)
    missing["data"].pop("oxidation_states")
    stored = ComputedStructureEntry.from_dict(missing)
    stored.entry_id = "mp-stored-states"
    stored.data["oxidation_states"] = {"Ce": 4, "Si": -3, "Ag": 1}
    compat = MaterialsProject2020Compatibility(check_potcar=False)
    nonempty = compat.process_entry(stored, on_error="raise").as_dict()
    assert nonempty["correction"] == pytest.approx(0.142)
    archive["lmdb"] = tmp_path / "oxidation-states.lmdb"
    _write_lmdb(archive["lmdb"], {"Ag-Ce-Si.Ce(SiAg)2.0": missing,
                                  "Ag-Ce-Si.Ce(SiAg)2.1": nonempty})
    out = tmp_path / "oxidation-states"
    manifest = builder.build(**archive, out=out)
    rows = _rows(out)
    assert all(row["stored_reproduced"] for row in rows.values())
    assert rows["Ag-Ce-Si.Ce(SiAg)2.0"]["stored_correction"] == 0
    assert rows["Ag-Ce-Si.Ce(SiAg)2.0"]["oxidation_states_nonempty"] is False
    assert rows["Ag-Ce-Si.Ce(SiAg)2.1"]["stored_correction"] == pytest.approx(0.142)
    assert rows["Ag-Ce-Si.Ce(SiAg)2.1"]["oxidation_states_nonempty"] is True
    assert manifest["counts"]["oxidation_states_nonempty"] == 1
    assert manifest["counts"]["stored_reproduced_false"] == 0
    assert _diagnostics(out, manifest) == [
        {"key": "Ag-Ce-Si.Ce(SiAg)2.1", "flags": ["oxidation_states_nonempty"]},
    ]


def test_target_new_and_unlisted_same_element(archive, tmp_path):
    shifts = json.loads(archive["shifts"].read_text())
    shifts["elements"]["Mg"]["target"] = "new"
    archive["shifts"].write_text(json.dumps(shifts))
    legacy = builder._load_legacy(archive["mp_elemental"])
    extra = _record(["Mg", "O"], "mp-third-hash", legacy)
    extra["parameters"]["potcar_spec"][0] = {"titel": MG_NEW["titel"], "hash": "third-Mg"}
    with lmdb.open(str(archive["lmdb"]), subdir=False) as env, env.begin(write=True) as txn:
        txn.put(b"Mg-O.MgO.2", pickle.dumps(extra))
    manifest = builder.build(**archive, out=tmp_path / "new-scale")
    rows = _rows(tmp_path / "new-scale")
    assert rows["Mg-O.MgO.0"]["potcar_shift"] == pytest.approx(-0.09)
    assert rows["Mg-O.MgO.0"]["raw_energy"] - rows["Mg-O.MgO.0"]["potcar_shift"] == pytest.approx(-7.91)
    assert rows["Mg-O.MgO.1"]["potcar_shift"] == 0
    assert rows["Mg-O.MgO.2"]["potcar_shift"] == 0
    assert rows["Mg-O.MgO.2"]["unlisted_nonlegacy"]
    assert manifest["counts"]["nonzero_shift_by_element"] == {"Mg": 1}


def test_missing_ambiguous_and_hash_based_potcars(archive, tmp_path):
    legacy = builder._load_legacy(archive["mp_elemental"])
    original = _record(["Mg", "O"], "mp-1", legacy)
    variants = {}
    for i in range(6):
        variants[f"Mg-O.MgO.{i}"] = copy.deepcopy(original)
    variants["Mg-O.MgO.0"]["parameters"].pop("potcar_spec")
    variants["Mg-O.MgO.1"]["parameters"]["potcar_spec"].append(MG_NEW)
    variants["Mg-O.MgO.2"]["parameters"]["potcar_spec"][0]["hash"] = None
    variants["Mg-O.MgO.3"]["parameters"]["potcar_spec"][0]["titel"] = "PAW_PBE Mg_pv renamed"
    variants["Mg-O.MgO.4"]["parameters"]["potcar_spec"].append(copy.deepcopy(MG_LEGACY))
    variants["Mg-O.MgO.5"]["parameters"]["potcar_spec"][0]["hash"] = MG_NEW["hash"]
    archive["lmdb"] = tmp_path / "metadata.lmdb"
    _write_lmdb(archive["lmdb"], variants)
    builder.build(**archive, out=tmp_path / "metadata")
    rows = list(_rows(tmp_path / "metadata").values())
    assert [row["unlisted_nonlegacy"] for row in rows] == [True, True, True, False, False, False]
    assert [row["potcar_shift"] for row in rows] == pytest.approx([0, 0, 0, 0, 0, 0.09])
    assert rows[3]["nonlegacy_elements"] == []
    assert rows[5]["nonlegacy_elements"] == ["Mg"]


def test_diagnostics_use_stored_metadata(archive, tmp_path):
    legacy = builder._load_legacy(archive["mp_elemental"])
    u_mismatch = _record(["Co", "O", "O"], "mp-u", legacy)
    u_mismatch["parameters"].update(run_type="GGA", hubbards={"Co": 0})
    oxide_mismatch = _record(["Mg", "O"], "mp-oxide", legacy)
    oxide_mismatch["data"]["oxide_type"] = "peroxide"
    archive["lmdb"] = tmp_path / "diagnostics.lmdb"
    _write_lmdb(archive["lmdb"], {"Co-O.CoO2.0": u_mismatch, "Mg-O.MgO.0": oxide_mismatch})
    manifest = builder.build(**archive, out=tmp_path / "diagnostics")
    rows = _rows(tmp_path / "diagnostics")
    assert rows["Co-O.CoO2.0"]["synth_params_mismatch"]
    assert not rows["Co-O.CoO2.0"]["stored_reproduced"]
    assert "CompatibilityError" in rows["Co-O.CoO2.0"]["stored_error"]
    assert rows["Co-O.CoO2.0"]["recorrect_error"] is None
    assert rows["Mg-O.MgO.0"]["oxide_type_changed"]
    assert not rows["Mg-O.MgO.0"]["stored_reproduced"]
    assert rows["Mg-O.MgO.0"]["recorrected_correction"] == pytest.approx(-0.687)
    assert manifest["counts"]["synth_params_mismatch"] == 1
    assert manifest["counts"]["oxide_type_changed"] == 1
    assert manifest["counts"]["stored_reproduced_false"] == 2
    assert _diagnostics(tmp_path / "diagnostics", manifest) == [
        {"key": "Co-O.CoO2.0", "flags": ["stored_reproduced_false", "synth_params_mismatch"]},
        {"key": "Mg-O.MgO.0", "flags": ["oxide_type_changed", "stored_reproduced_false"]},
    ]


@pytest.mark.parametrize("failure", ["removed", "rejected"])
def test_recorrect_failure_is_a_row(archive, tmp_path, monkeypatch, failure):
    original = MaterialsProject2020Compatibility.process_entry

    def fail_candidate(self, entry, **kwargs):
        if "oxidation_states" not in entry.data:
            if failure == "removed":
                return None
            raise CompatibilityError("fixture rejection")
        return original(self, entry, **kwargs)

    monkeypatch.setattr(MaterialsProject2020Compatibility, "process_entry", fail_candidate)
    manifest = builder.build(**archive, out=tmp_path / "failures")
    assert manifest["counts"]["entries"] == manifest["counts"]["recorrect_failures"] == 5
    assert manifest["counts"]["recorrect_changed"] == 0
    rows = _rows(tmp_path / "failures")
    for row in rows.values():
        assert row["stored_reproduced"]
        assert math.isnan(row["recorrected_correction"])
        assert row["recorrect_error"] == ("mp2020_removed" if failure == "removed"
                                          else "CompatibilityError: fixture rejection")
    assert _diagnostics(tmp_path / "failures", manifest) == [
        {"key": row["key"], "flags": ["recorrect_failure"], "recorrect_error": row["recorrect_error"]}
        for row in rows.values()
    ]


def test_limit_cli_and_missing_shifts(archive, tmp_path, capsys):
    arguments = [arg for name, path in archive.items() for arg in ("--" + name.replace("_", "-"), str(path))]
    builder.main([*arguments, "--out", str(tmp_path / "limit"), "--limit", "2"])
    assert json.loads(capsys.readouterr().out)["counts"]["entries"] == 2
    assert list(_rows(tmp_path / "limit")) == ["Ag-Ce-Si.Ce(SiAg)2.0", "Co-O.CoO2.34"]
    archive["shifts"].unlink()
    with pytest.raises(FileNotFoundError, match="Missing shifts input:.*--shifts"):
        builder.build(**archive, out=tmp_path / "absent")


def test_invalid_shift_provenance(archive, tmp_path):
    asset = json.loads(archive["shifts"].read_text())
    asset["elements"]["Mg"]["legacy"]["hash"] = "wrong-legacy"
    archive["shifts"].write_text(json.dumps(asset))
    with pytest.raises(ValueError, match="legacy hash disagrees"):
        builder.build(**archive, out=tmp_path / "wrong")


def test_not_applied_shift_hash_is_unlisted(archive, tmp_path):
    asset = json.loads(archive["shifts"].read_text())
    asset["elements"]["Mg"].update(shift=0.0, status="not_applied")
    archive["shifts"].write_text(json.dumps(asset))
    out = tmp_path / "not-applied"
    manifest = builder.build(**archive, out=out)
    row = _rows(out)["Mg-O.MgO.1"]
    assert row["unlisted_nonlegacy"] is True and row["potcar_shift"] == 0
    triples = {(t["element"], t["hash"]): t for t in manifest["potcar_inventory"]}
    assert triples["Mg", MG_NEW["hash"]]["status"] == "unlisted"
