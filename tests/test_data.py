"""Small CPU data gates: real MP-20 cells, geometric oracles and exact resume."""
import csv
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure
import pytest
import spglib
import torch
from torch.utils.data import DataLoader

from orbitol import contracts as C
from orbitol.data import (
    OrbitDataset, ResumableSampler, _pack_cache, augment_tokens, build_cache,
    collate_orbits, expand_tokens, family_ltri, lattice_to_ltri, ltri_to_lattice,
    read_csv_records, tokenise, validate_round_trip,
)
from orbitol.standardise import assert_frozen_cell, species_permutation, standardise
from orbitol.tables import build_tables, load_tables

GROUPS = (1, 5, 7, 14, 15, 19, 44, 62, 63, 88, 129, 136, 141, 143, 166, 186, 194, 216, 225, 227)


@pytest.fixture(scope="module")
def tables():
    return load_tables(C.TABLES_DIR) if Path(C.TABLES_DIR).is_dir() else build_tables(groups=GROUPS)


@pytest.fixture(scope="module")
def examples(tables):
    path = Path(C.DATA_ROOT) / "raw" / "train.csv"
    if not path.exists():
        pytest.skip(f"MP-20 CSV unavailable: {path}")
    selected = {}
    for record in read_csv_records(path):
        g = int(record["spacegroup.number"])
        if g in GROUPS and g not in selected:
            selected[g] = record
        if len(selected) == len(GROUPS):
            break
    assert set(selected) == set(GROUPS)
    result = []
    for g in GROUPS:
        record = selected[g]
        structure = Structure.from_str(record["cif"], fmt="cif")
        ideal = standardise(structure, tables=tables)
        tokens = tokenise(ideal, tables, mp_id=record["material_id"])
        assert tokens["G"] == g
        result.append((record, structure, ideal, tokens))
    return result


def _manifest(tables, split="train"):
    return dict(format_version=1, split=split, table_sha256=tables.sha256,
                symprec=0.1, angle_tolerance=5.0)


@pytest.fixture(scope="module")
def cache_path(examples, tables, tmp_path_factory):
    path = tmp_path_factory.mktemp("orbit_data") / "train.npz"
    np.savez_compressed(path, **_pack_cache([t for _, _, _, t in examples], _manifest(tables)))
    return path


@pytest.mark.data
def test_twenty_mp20_round_trips(examples, tables):
    assert len(examples) == 20
    for _, _, ideal, tokens in examples:
        frac, lattice = validate_round_trip(tokens, ideal, tables)
        assert frac <= 1e-6 and lattice <= 1e-8 and tokens["round_trip_ok"]
        assert tokens["U"].dtype == tokens["lat"].dtype == np.float64
        assert tokens["n_prim"] * tables.arrays["group_ncent"][ideal.G] == tokens["n_conv"]
        assert np.isfinite([ideal.disp_max, ideal.disp_mean, ideal.strain]).all()
        assert ideal.disp_max >= ideal.disp_mean >= 0 and ideal.strain >= 0
        np.testing.assert_array_equal(tokens["lat"], family_ltri(tokens["lat"], tokens["family"]))
    # This MP-20 rutile-like entry contains both fixed metal and free anion orbits.
    rutile = next(t for _, _, _, t in examples if t["G"] == 136)
    assert {tables.row_id(136, "a"), tables.row_id(136, "f")} <= set(rutile["row"])


def test_one_standardisation_call_and_origin_choice(examples, tables, monkeypatch):
    original = spglib.get_symmetry_dataset
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(spglib, "get_symmetry_dataset", counted)
    structure = next(s for _, s, _, t in examples if t["G"] == 227)
    ideal = standardise(structure, tables=tables)
    tokenise(ideal, tables)
    assert calls == [dict(symprec=0.1, angle_tolerance=5.0)]
    np.testing.assert_allclose(ideal.frozen_shift, [1/8] * 3, atol=0, rtol=0)
    assert ideal.transformation_matrix.shape == ideal.rotation.shape == (3, 3)
    assert ideal.origin_shift.shape == (3,)


def test_displacement_and_strain_separate_affine_change(tables):
    perfect = Structure.from_spacegroup(136, Lattice.tetragonal(4.6, 3.0), [22, 8],
                                       [[0, 0, 0], [.305, .305, 0]])
    clean = standardise(perfect, tables=tables)
    assert clean.disp_max < 1e-12 and clean.strain < 1e-12
    distorted = Structure(perfect.lattice.matrix @ np.diag([1.002, 1, 1]),
                          perfect.atomic_numbers, perfect.frac_coords)
    distorted.translate_sites([2], [.001, -.001, 0], frac_coords=True)
    noisy = standardise(distorted, tables=tables)
    assert noisy.G == 136 and 0 < noisy.disp_mean < noisy.disp_max < .02
    assert 0 < noisy.strain < .01
    # Input rotations and supercells do not masquerade as strain or internal motion.
    supercell = perfect.copy()
    supercell.make_supercell([2, 1, 1])
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated = Structure(supercell.lattice.matrix @ rotation, supercell.atomic_numbers, supercell.frac_coords)
    result = standardise(rotated, tables=tables)
    assert result.G == 136 and result.disp_max < 1e-12 and result.strain < 1e-12


def test_round_trip_is_fatal_and_bijective(examples, tables):
    _, _, ideal, tokens = next(e for e in examples if e[3]["G"] == 7)
    for name, index, value in (("U", (0, 0), .1), ("lat", 0, .01), ("A0", 0, 1)):
        broken = dict(tokens)
        broken[name] = tokens[name].copy()
        broken[name][index] += value
        with pytest.raises(ValueError):
            validate_round_trip(broken, ideal, tables)
    with pytest.raises(ValueError):
        species_permutation([[0, 0, 0], [0, 0, 0]], [1, 1], [[0, 0, 0], [.5, 0, 0]], [1, 1])


def _distance_oracle(row, atoms, coords, lat, tables):
    # Independent periodic-neighbour enumeration, including lattice translations.
    xyz, species = expand_tokens(row, atoms, coords, tables)
    structure = Structure(ltri_to_lattice(lat), species, xyz)
    i, j, _, distances = structure.get_neighbor_list(8.0)
    pairs = np.sort(np.stack((species[i], species[j]), axis=1), axis=1)
    grouped = {tuple(pair): np.sort(distances[np.all(pairs == pair, axis=1)])
               for pair in np.unique(pairs, axis=0)}
    return structure, grouped


@pytest.mark.data
def test_augmentation_geometry_oracle(examples, tables):
    changed_basis, relabelled = False, False
    for _, _, _, t in examples:
        original, distances = _distance_oracle(t["row"], t["A0"], t["U"], t["lat"], tables)
        for seed in range(3):
            row, U, lat, P = augment_tokens(t["row"], t["U"], t["lat"], t["G"],
                                          tables, np.random.default_rng(seed))
            augmented, after = _distance_oracle(row, t["A0"], U, lat, tables)
            assert after.keys() == distances.keys()
            for pair in after:
                np.testing.assert_allclose(after[pair], distances[pair], rtol=1e-8, atol=1e-8)
            assert augmented.volume == pytest.approx(original.volume, rel=1e-8)
            assert np.linalg.det(P) == pytest.approx(1, abs=1e-12)
            np.testing.assert_array_equal(lat, family_ltri(lat, t["family"]))
            assert_frozen_cell(augmented.lattice.matrix, augmented.frac_coords,
                               np.asarray(augmented.atomic_numbers), t["G"], tables)
            changed_basis |= not np.allclose(lat, t["lat"])
            relabelled |= not np.array_equal(row, t["row"])
    assert changed_basis and relabelled


def test_chart_images_include_pure_translations(tables):
    r = tables.row_id(166, "h")
    a = tables.arrays
    d = int(a["row_dof"][r])
    count = int(a["row_img_count"][r])
    pure = [i for i in range(count)
            if np.array_equal(a["row_img_S"][r, i, :d, :d], np.eye(d))
            and np.any(a["row_img_s"][r, i] % 1)]
    assert pure
    # Record the actual image choices; all table entries, translations included, are reachable.
    class RecordingGenerator:
        def __init__(self, seed):
            self.rng, self.image = np.random.default_rng(seed), None

        def choice(self, *args, **kwargs):
            return self.rng.choice(*args, **kwargs)

        def random(self, *args, **kwargs):
            return self.rng.random(*args, **kwargs)

        def integers(self, high):
            self.image = int(self.rng.integers(high))
            return self.image

    lat = lattice_to_ltri(Lattice.hexagonal(4, 7).matrix, C.FAMILIES.index("hexagonal"))
    hits = set()
    for seed in range(64):
        generator = RecordingGenerator(seed)
        rows, U, _, _ = augment_tokens([r], [[.123, .234, 0]], lat, 166, tables, generator)
        assert rows.tolist() == [r]
        if generator.image in pure:
            hits.add(generator.image)
            # Normaliser may move the orbit, so compare its full periodic distance spectrum.
            _, expected = _distance_oracle([r], [8], [[.123, .234, 0]], lat, tables)
            _, actual = _distance_oracle(rows, [8], U, lat, tables)
            np.testing.assert_allclose(actual[(8, 8)], expected[(8, 8)], rtol=1e-8, atol=1e-8)
    assert hits == set(pure)


def _equal_batches(left, right):
    assert left.keys() == right.keys() == C.ORBIT_BATCH_KEYS.keys()
    for name in left:
        if name == "mp_id":
            assert left[name] == right[name]
        else:
            assert torch.equal(left[name], right[name]), name


def _collated(loader):
    # Transport the dataset's NumPy items through the worker queue, then collate.
    # This checks real multiprocessing without requiring torch's AF_UNIX tensor
    # descriptor service, which restricted execution environments can prohibit.
    return [collate_orbits(items, 20) for items in loader]


def test_dataset_collation_and_worker_independence(cache_path, tables):
    data = OrbitDataset(cache_path, tables, augment=True, aug_seed=37)
    data.set_epoch(4)
    kwargs = dict(batch_size=4, collate_fn=list)
    single = _collated(DataLoader(data, num_workers=0, **kwargs))
    loader = DataLoader(data, num_workers=2, persistent_workers=True, timeout=10, **kwargs)
    multi = _collated(loader)
    for a, b in zip(single, multi):
        _equal_batches(a, b)
        assert a["row"].shape == (4, 20) and a["U"].dtype == torch.float32
        assert a["dof_mask"].dtype == a["pad_mask"].dtype == torch.bool
        assert not a["row"][a["pad_mask"]].any()
        assert not a["dof_mask"][a["pad_mask"]].any()
        assert torch.all((a["U"] >= 0) & (a["U"] < 1))
    data.set_epoch(5)
    after = _collated(loader)
    assert any(not torch.equal(a["U"], b["U"]) for a, b in zip(multi, after))
    for a, b in zip(after, _collated(DataLoader(data, num_workers=0, **kwargs))):
        _equal_batches(a, b)
    with pytest.raises(ValueError, match="do not fit"):
        collate_orbits([data[0]], 1)


def test_collate_owns_output_and_rejects_invalid_items(cache_path, tables):
    dataset = OrbitDataset(cache_path, tables, augment=False, aug_seed=0)
    items = [dataset[0], dataset[7], dataset[19]]
    actual = collate_orbits(items, 20)
    expected = {k: v.clone() if torch.is_tensor(v) else v for k, v in actual.items()}
    for name in ("row", "A0", "U", "dof_mask", "pad_mask", "lat"):
        items[0][name][...] = 0
    _equal_batches(actual, expected)
    for invalid, K in (([], 20), ([dataset[0]], 0)):
        with pytest.raises(ValueError):
            collate_orbits(invalid, K)
    item = dataset[0]
    item["extra"] = 1
    with pytest.raises(ValueError, match="keys"):
        collate_orbits([item], 20)


def test_cache_metadata_and_train_only_augmentation(cache_path, tables, tmp_path):
    with np.load(cache_path) as cache:
        arrays = {k: cache[k] for k in cache.files}
    assert arrays["U"].dtype == arrays["lat"].dtype == np.float32
    arrays["manifest"] = np.asarray(json.dumps(_manifest(tables, "val")))
    path = tmp_path / "val.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="train split"):
        OrbitDataset(path, tables, augment=True, aug_seed=1)
    fixed = OrbitDataset(path, tables, augment=False, aug_seed=1)
    first = collate_orbits([fixed[1]], 20)
    fixed.set_epoch(9)
    _equal_batches(first, collate_orbits([fixed[1]], 20))
    bad = _manifest(tables)
    bad["table_sha256"] = "wrong"
    arrays["manifest"] = np.asarray(json.dumps(bad))
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="sha256"):
        OrbitDataset(path, tables, augment=False, aug_seed=1)


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_counts_consumed_batches_not_prefetch(cache_path, tables, workers):
    data = OrbitDataset(cache_path, tables, augment=True, aug_seed=41)
    data.set_epoch(3)
    uninterrupted = ResumableSampler(20, seed=19, batch_size=2)
    uninterrupted.set_epoch(3)
    kwargs = dict(num_workers=workers, collate_fn=list, timeout=10 if workers else 0)
    expected = _collated(DataLoader(data, batch_sampler=uninterrupted, **kwargs))[:6]
    assert uninterrupted.cursor == 0  # Even exhausting a loader is not optimiser consumption.
    sampler = ResumableSampler(20, seed=19, batch_size=2)
    sampler.set_epoch(3)
    loader = DataLoader(data, batch_sampler=sampler, **kwargs)
    iterator = iter(loader)
    first = []
    for _ in range(3):
        first.append(collate_orbits(next(iterator), 20))
        sampler.mark_consumed()
    state = sampler.state_dict()
    assert state["cursor"] == 3 and len(sampler) == 7
    del iterator, loader
    resumed = ResumableSampler(20, seed=19, batch_size=2)
    resumed.load_state_dict(state)
    data.set_epoch(resumed.epoch)
    iterator = iter(DataLoader(data, batch_sampler=resumed, **kwargs))
    second = []
    for _ in range(3):
        second.append(collate_orbits(next(iterator), 20))
        resumed.mark_consumed()
    for a, b in zip(first + second, expected):
        _equal_batches(a, b)
    assert resumed.cursor == 6
    del iterator


def test_sampler_tail_epoch_and_invalid_resume():
    sampler = ResumableSampler(21, seed=8, batch_size=2, drop_last=False)
    first = list(sampler)
    assert len(first) == 11 and len(first[-1]) == 1
    assert sorted(sum(first, [])) == list(range(21))
    sampler.mark_consumed(11)
    assert not list(sampler)
    with pytest.raises(ValueError):
        sampler.mark_consumed()
    sampler.set_epoch(1)
    assert sampler.cursor == 0 and list(sampler) != first
    state = sampler.state_dict()
    state["batch_size"] = 3
    with pytest.raises(ValueError, match="batch_size"):
        sampler.load_state_dict(state)


def test_cache_builder_limit_and_failure_report(tables, examples, tmp_path):
    table_dir = tmp_path / "tables"
    tables.save(table_dir)
    source = tmp_path / "source.csv"
    fields = ["material_id", "cif", "spacegroup.number"]
    with source.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerow(examples[0][0])
        for name in ("bad-one", "bad-two"):
            writer.writerow(dict(material_id=name, cif="broken", **{"spacegroup.number": 1}))
    out = tmp_path / "cache"
    manifest = build_cache("val", table_dir, out, limit=1, csv_path=source)
    assert manifest["count"] == 1 and manifest["table_sha256"] == tables.sha256
    assert manifest["round_trip_precision"] == "float64 before float32 storage"
    assert manifest["seconds_per_structure"] > 0
    with np.load(out / "val.npz") as cache:
        assert cache["round_trip_ok"].tolist() == [True]
        assert cache["row"].shape == (1, 20)
    previous = (out / "val.npz").read_bytes()
    with pytest.raises(RuntimeError, match="bad-one[\\s\\S]*bad-two"):
        build_cache("val", table_dir, out, csv_path=source)
    assert (out / "val.npz").read_bytes() == previous


def test_p1_items_get_a_random_translation():
    """P1's polar space is R^3: augmentation translates P1 structures, as Crystalite's training does."""
    import numpy as np
    from orbitol.data import augment_tokens
    from orbitol.tables import build_tables
    ts = build_tables(groups=[1])
    row = np.array([1, 1, 1]); U = np.array([[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.9, 0.1, 0.4]]); lat = np.zeros(6)
    r2, U2, lat2, P = augment_tokens(row, U, lat, 1, ts, np.random.default_rng(0))
    shift = np.mod(U2 - U, 1.0)
    assert np.allclose(shift, shift[0], atol=1e-12) and not np.allclose(shift[0], 0)  # one common translation
    assert np.array_equal(r2, row) and np.array_equal(lat2, lat) and np.array_equal(P, np.eye(3))
