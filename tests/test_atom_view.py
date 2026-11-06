"""Crystalite atom items through the unchanged Orbitol training path."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
import pytest
import spglib
import torch

import orbitol.core  # noqa: F401
from orbitol import contracts as C, edm, lattice
from orbitol.data import (OrbitDataset, _pack_cache, build_atom_cache, collate_orbits,
                          file_sha256, ltri_to_lattice, main, tokenise)
from orbitol.decode import decode_batch
from orbitol.model import build_model
from orbitol.sample import allowed_elements, load_encoding
from orbitol.skeleton_prior import SkeletonBank
from orbitol.standardise import Standardised
from orbitol.symmetry_panel import SymmetryPanel
from orbitol.tables import build_tables
from orbitol.train import Trainer, parser
from src.crystalite.crystalite import CrystaliteModel
from src.crystalite.edm_utils import compute_edm_loss, denoise_edm, sample_sigma
from src.data.mp20_tokens import collate_mp20_tokens, tokens_to_structure
from src.models.lattice_repr import ltri_params_to_lattice_matrix, y1_to_lattice_latent
from test_model import SMALL_KW

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.data


@pytest.fixture(scope="module")
def atom_fixture():
    source = Path(C.DATA_ROOT) / "processed/mp20_tokens_train.pt"
    if not source.is_file():
        pytest.skip(f"Processed Crystalite train tokens unavailable: {source}")
    items = torch.load(source, map_location="cpu", weights_only=False)
    assert len(items) == 27_138
    items = items[:12]  # Mixed sizes, padding, repeated N, and a full 20-atom cell.
    with tempfile.TemporaryDirectory(prefix=".atom-tests-", dir=ROOT) as name:
        root = Path(name)
        tables = build_tables(groups=[1])
        tables_dir = root / "tables"
        tables.save(tables_dir)
        data_root = root / "data"
        processed = data_root / "processed"
        processed.mkdir(parents=True)
        cache = root / "cache"
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(C.DATASETS, "mp20", replace(C.dataset_spec("mp20"), data_root=str(data_root)))

            def no_symmetry_search(*args, **kwargs):
                pytest.fail("Atom cache must not call spglib")

            patch.setattr(spglib, "get_symmetry_dataset", no_symmetry_search)
            for split in ("train", "val", "test"):
                torch.save(items, processed / f"mp20_tokens_{split}.pt")
                build_atom_cache(split, tables_dir, cache)
        dataset = OrbitDataset(cache / "train.npz", tables, augment=False, aug_seed=0)
        batch = collate_orbits([dataset[i] for i in range(len(dataset))], C.TrainConfig().K)
        yield dict(root=root, items=items, tables=tables, tables_dir=tables_dir,
                   data_root=data_root, cache=cache, dataset=dataset, batch=batch,
                   core_batch=collate_mp20_tokens(items))


def p1_tokens(item, tables):
    # Impose P1 on the stored cell, as atom view requires. Passing Standardised
    # bypasses symmetry detection while exercising tokenise's existing P1 chart.
    structure = tokens_to_structure(item)
    ideal = Standardised(
        G=1, hall=1, lattice=structure.lattice.matrix, positions=structure.frac_coords,
        types=np.asarray(structure.atomic_numbers), transformation_matrix=np.eye(3),
        origin_shift=np.zeros(3), rotation=np.eye(3), frozen_shift=np.zeros(3),
        disp_max=0.0, disp_mean=0.0, strain=0.0,
    )
    return dict(tokenise(ideal, tables, mp_id=item["mp_id"]), csv_G=1)


def test_schema_provenance_and_p1_conventions(atom_fixture):
    f = atom_fixture
    data, batch, tables = f["dataset"], f["batch"], f["tables"]
    reference = p1_tokens(f["items"][0], tables)
    expected = _pack_cache([reference], {})
    assert data.cache.keys() == expected.keys()
    for key, value in expected.items():
        if value.dtype.kind == "U":
            assert data.cache[key].dtype.kind == "U"
        else:
            assert data.cache[key].dtype == value.dtype, key
    assert len(data) == len(f["items"])
    assert torch.all(batch["G"] == 1)
    assert torch.all(batch["family"] == C.FAMILIES.index("triclinic"))
    assert torch.all(batch["row"][~batch["pad_mask"]] == tables.row_id(1, "a"))
    assert batch["dof_mask"].equal((~batch["pad_mask"])[..., None].expand_as(batch["U"]))
    for key in ("A0", "pad_mask"):
        assert batch[key].equal(f["core_batch"][key])
    assert batch["U"].equal(f["core_batch"]["F1"])
    assert batch["n_prim"].equal(f["core_batch"]["num_atoms"])
    for i, item in enumerate(f["items"]):
        reference = p1_tokens(item, tables)
        np.testing.assert_array_equal(data[i]["U"], reference["U"])
        # tokens_to_structure rounds angles to float32 degrees. Compare cells
        # at the atom manifest's relative lattice tolerance; the training latent
        # comparison below bypasses that decoder and remains bitwise exact.
        actual_lattice = ltri_to_lattice(data[i]["lat"])
        reference_lattice = ltri_to_lattice(reference["lat"])
        assert np.linalg.norm(actual_lattice - reference_lattice) / np.linalg.norm(reference_lattice) <= 1e-6
    for split in ("train", "val", "test"):
        path = f["cache"] / f"{split}.npz"
        manifest = json.loads((f["cache"] / f"{split}.manifest.json").read_text())
        assert manifest["view"] == "atom" and manifest["count"] == len(f["items"])
        assert manifest["excluded"] == [] and manifest["max_failures"] == 0
        assert manifest["table_sha256"] == tables.sha256
        assert manifest["cache_sha256"] == file_sha256(path)
        source = f["data_root"] / "processed" / f"mp20_tokens_{split}.pt"
        assert manifest["source_token_path"] == str(source.resolve())
        assert manifest["source_token_sha256"] == file_sha256(source)
        assert manifest["csv_path"] is manifest["csv_sha256"] is manifest["label_agreement"] is None
        with np.load(path) as cache:
            embedded = json.loads(str(cache["manifest"].item()))
            assert all(manifest[key] == value for key, value in embedded.items())
            assert cache["round_trip_ok"].all()
            assert cache["round_trip_lattice_error"].max() <= manifest["lattice_relative_tolerance"]


def test_decode_round_trip(atom_fixture):
    f, encoding = atom_fixture, load_encoding()
    batch = f["batch"]
    decoded = decode_batch(
        dict(type=encoding.encode_from_A0(batch["A0"], batch["pad_mask"]),
             u=batch["U"] - 0.5, z=batch["lat"]),
        batch, f["tables"], encoding, allowed_elements(f["data_root"]), n_max=20,
    )
    assert not decoded.failures and decoded.contains_a.all()
    matcher = StructureMatcher(primitive_cell=False, scale=False, stol=1e-4, ltol=1e-5, angle_tol=1e-3)
    for i, item in enumerate(f["items"]):
        expected = tokens_to_structure(item)
        emitted = tokens_to_structure({key: value[i] for key, value in decoded.rows.items()})
        for actual in (decoded.structures[i], emitted):
            assert len(actual) == len(expected)
            assert Counter(actual.atomic_numbers) == Counter(expected.atomic_numbers)
            assert matcher.fit(actual, expected)


def test_lattice_parity(atom_fixture):
    f = atom_fixture
    batch = f["batch"]
    expected = y1_to_lattice_latent(f["core_batch"]["Y1"], "ltri")
    torch.testing.assert_close(batch["lat"], expected, atol=0, rtol=0)
    # The trainer's family chart and the core latent describe the same oriented L.
    torch.testing.assert_close(lattice.to_matrix(batch["lat"], batch["family"]),
                               ltri_params_to_lattice_matrix(expected.double()), atol=1e-6, rtol=0)
    noise = edm.orbit_noise(batch, torch.Generator().manual_seed(8))
    clean = load_encoding().encode_from_A0(batch["A0"], batch["pad_mask"])
    state = edm.corrupt(batch, clean, torch.ones(len(batch["G"])), noise,
                        f["tables"].torch("cpu"), objective="M")
    torch.testing.assert_close(state["z_clean"], expected, atol=0, rtol=0)


def parity_models(tables, *, perturb=False):
    torch.manual_seed(23)
    base = CrystaliteModel(**SMALL_KW).eval()
    # As in test_model, also test nonzero readouts to avoid vacuous zero parity.
    if perturb:
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.add_(0.05 * torch.randn_like(parameter))
    cfg = C.TrainConfig(d_model=32, n_heads=4, n_layers=2, sg_embed="none", row_embed="categorical",
                        compile_trunk="none")
    tb = tables.torch("cpu")
    model = build_model(cfg, tb.row_desc, tb.group_desc).eval()
    loaded = model.load_state_dict(base.state_dict(), strict=False)
    assert not loaded.unexpected_keys
    assert set(loaded.missing_keys) == {name for name, _ in model.named_parameters()
                                      if name.startswith(("row_embed.", "sg_embed."))}
    assert all(torch.count_nonzero(p) == 0 for p in model.row_embed.parameters())
    return model, base


@pytest.mark.parametrize("perturb", [False, True])
def test_model_parity_at_initialisation(atom_fixture, perturb):
    f = atom_fixture
    batch, core = f["batch"], f["core_batch"]
    model, base = parity_models(f["tables"], perturb=perturb)
    encoding = load_encoding()
    c_noise = torch.linspace(-2, 1, len(f["items"]))
    with torch.no_grad():
        actual = model(encoding.encode_from_A0(batch["A0"], batch["pad_mask"]), batch["U"],
                       batch["lat"], batch["pad_mask"], c_noise, row=batch["row"], G=batch["G"])
        expected = base(encoding.encode_from_A0(core["A0"], core["pad_mask"]), core["F1"],
                        y1_to_lattice_latent(core["Y1"], "ltri"), core["pad_mask"], c_noise)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_objective_m_parity(atom_fixture):
    # Same corruption/denoising/loss comparison as test_objective_acceptance,
    # now with independent real Crystalite and collated atom-cache clean targets.
    f = atom_fixture
    batch, core = f["batch"], f["core_batch"]
    model, base = parity_models(f["tables"], perturb=True)
    tb, encoding = f["tables"].torch("cpu"), load_encoding()
    clean = encoding.encode_from_A0(core["A0"], core["pad_mask"])
    generator = torch.Generator().manual_seed(3)
    sigma = sample_sigma(len(f["items"]), torch.device("cpu"), -1.2, 1.2, .002, 80., generator=generator)
    noise = edm.orbit_noise(batch, generator)
    frac_c = core["F1"] - 0.5
    frac_noisy = torch.where((~core["pad_mask"])[..., None],
                             frac_c + sigma[:, None, None] * noise["coord"], 0.0)
    lat = y1_to_lattice_latent(core["Y1"], "ltri")
    with torch.no_grad():
        ref = denoise_edm(base, clean + sigma[:, None, None] * noise["type"], frac_noisy,
                          lat + sigma[:, None] * noise["lat"], core["pad_mask"], sigma,
                          .3, .3, .3, .002, 80.)
        ref_loss = compute_edm_loss(ref, dict(type=clean, frac_c=frac_c, lat=lat), frac_noisy,
                                    sigma, core["pad_mask"], .3, .3, .3,
                                    {"A": 1., "F": 50., "Y": 5.},
                                    coord_loss_mode="frac_mse", lattice_repr="ltri")
        orbit_clean = encoding.encode_from_A0(batch["A0"], batch["pad_mask"])
        state = edm.corrupt(batch, orbit_clean, sigma, noise, tb, objective="M")
        torch.testing.assert_close(state["u_noisy"], frac_noisy, atol=0, rtol=0)
        den = edm.denoise_orbit(model, state, sigma, batch, tb, objective="M", sigma_data=(.3, .3, .3))
        loss = edm.compute_orbit_loss(den, state, sigma, batch, tb, objective="M",
                                      sigma_data=(.3, .3, .3), loss_weights=(1., 50., 5.))
    for actual, expected in (("D_type", "type"), ("D_u", "frac"), ("D_z", "lat")):
        torch.testing.assert_close(den[actual], ref[expected], atol=0, rtol=0)
    for actual, expected in (("total", "loss_total"), ("A", "loss_a"), ("F", "loss_f"), ("Y", "loss_y")):
        torch.testing.assert_close(loss[actual], ref_loss[expected], atol=0, rtol=0)


def test_empirical_atom_count_bank(atom_fixture):
    f = atom_fixture
    bank = SkeletonBank.from_cache(f["cache"] / "train.npz")
    histogram = Counter(f["batch"]["n_prim"].tolist())
    assert Counter(len(s.rows) for s in bank.training) == histogram
    assert {len(rows): count for (g, rows), count in bank.counts.items()} == histogram
    samples = bank.sample(4096, generator=np.random.default_rng(13))
    drawn = Counter(len(s.rows) for s in samples)
    assert drawn.keys() == histogram.keys()
    for n, count in histogram.items():
        p = count / len(bank.training)
        assert abs(drawn[n] / len(samples) - p) < 6 * np.sqrt(p * (1 - p) / len(samples))
    assert all(s.G == 1 and set(s.rows) == {f["tables"].row_id(1, "a")} for s in samples)


def test_cli_limit_and_no_exclusions(atom_fixture, monkeypatch, capsys):
    f = atom_fixture
    monkeypatch.setitem(C.DATASETS, "mp20", replace(
        C.dataset_spec("mp20"), data_root=str(f["data_root"]),
        atom_cache_dir=str(f["data_root"] / "atom_cache")))
    main(["build-cache", "--view", "atom", "--split", "train",
          "--tables", str(f["tables_dir"]), "--limit", "2"])
    path = f["data_root"] / "atom_cache/train.npz"
    output = json.loads(capsys.readouterr().out)
    assert output["cache"] == str(path) and output["count"] == 2
    previous = path.read_bytes()
    for split, limit in (("bad", None), ("train", 0), ("train", -1)):
        with pytest.raises(ValueError):
            build_atom_cache(split, f["tables_dir"], path.parent, limit=limit)
    with pytest.raises(SystemExit):
        main(["build-cache", "--view", "atom", "--split", "train", "--max-failures", "1"])
    source = f["data_root"] / "processed/mp20_tokens_train.pt"
    original = source.read_bytes()
    broken = {**f["items"][0], "F1": torch.full_like(f["items"][0]["F1"], float("nan"))}
    try:
        torch.save([f["items"][0], broken], source)
        with pytest.raises(ValueError, match="invalid atom count, species, coordinates or lattice"):
            build_atom_cache("train", f["tables_dir"], path.parent)
        assert path.read_bytes() == previous
    finally:
        source.write_bytes(original)


def test_three_steps_and_sampling_evaluation(atom_fixture, monkeypatch):
    f = atom_fixture
    monkeypatch.setenv("WANDB_MODE", "disabled")
    reference = f["root"] / "reference.json"
    SymmetryPanel(f["data_root"] / "processed/mp20_tokens_train.pt", f["items"], reference).reference
    cfg = C.TrainConfig(
        d_model=32, n_layers=2, n_heads=4, sg_embed="none", row_embed="categorical",
        batch_size=4, num_workers=0, max_steps=3, warmup_steps=1, precision="fp32",
        objective="P", sample_count=8, sample_num_steps=5, sample_frequency=3,
        compile_trunk="none", pad_multiple=0, sample_token_budget=0,
        eval_mode="sync", nequip_eval=False,
        val_every=2, ckpt_every=2, log_every=1,
        extra=dict(tables_dir=str(f["tables_dir"]), cache_dir=str(f["cache"]),
                   data_root=str(f["data_root"]), symmetry_reference=str(reference),
                   val_batches=1, sample_chunk_size=4),
    )
    run = Trainer(cfg, run_name="atom-smoke", out_root=f["root"] / "runs", device="cpu", no_wandb=True)
    try:
        assert run.evaluator.thermo is None and run.evaluator.sun_k == 0
        run.run()
        saved = torch.load(run.checkpoint_dir / "latest.pt", weights_only=False)
        assert saved["step"] == saved["last_sample_step"] == saved["sampler"]["cursor"] == 3
        samples = torch.load(run.run_dir / "samples/step_0000003.pt", weights_only=False)
        assert samples["rows"]["A0"].shape == (8, 20)
        assert len(samples["skeletons"]) == 8 and torch.all(samples["requested_G"] == 1)
        metrics = [json.loads(line) for line in (run.run_dir / "metrics.jsonl").read_text().splitlines()]
        assert "val/loss_total" in metrics[1]
        assert "eval/valid_rate" in metrics[-1]
        assert "decode/contains_a_rate" in metrics[-1]
        assert len(list((run.run_dir / "samples").glob("*.pt"))) == 1
    finally:
        run.close()


def test_launch_extra_defaults():
    parsed = vars(parser().parse_args([
        "--run-name", "crystalite_p", "--objective", "P", "--sg-embed", "none",
        "--row-embed", "categorical", "--extra",
        '{"cache_dir": "/data/mp20/atom_cache"}',
    ]))
    assert C.TrainConfig().extra == {}
    assert parsed["extra"] == {"cache_dir": "/data/mp20/atom_cache"}
