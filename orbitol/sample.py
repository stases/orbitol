"""Shared EMA sampling path for training monitoring and post-hoc pools."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import random
from typing import get_args, get_origin, get_type_hints, Literal

import numpy as np
from pymatgen.io.cif import CifWriter
import torch

import orbitol.core as core
from orbitol import contracts as C
from orbitol.decode import decode_batch

_REFERENCE_DECODE = decode_batch
from orbitol.model import build_model
from orbitol.sampler import orbit_edm_sampler
from orbitol.skeleton_prior import SkeletonBank
from orbitol.tables import load_tables
from src.data.mp20_tokens import tokens_to_structure
from src.models.type_encoding_state import restore_type_encoding
from src.utils.dataset import compute_allowed_elements
from src.utils.ema import EMA

ROOT = Path(__file__).resolve().parents[1]
# Execution choices include CPU decode overlap, GPU settings (docs/hardware.md), Heun/churn settings and the
# noise schedule. Snapshot sampling never trains, so sigma_min/sigma_max here leave the training noise range untouched.
SAMPLER_OVERRIDES = ("precision", "compile_trunk", "sample_token_budget", "decode_impl", "decode_overlap",
                     "sample_num_steps", "S_churn", "S_min", "S_max", "S_noise", "sigma_min", "sigma_max", "rho")


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def load_encoding():
    state = torch.load(ROOT / C.TYPE_ENCODING_STATE, map_location="cpu", weights_only=False)
    encoding = restore_type_encoding(state)  # Recomputes the digest, without fitting PCA.
    if state["state_sha256"] != C.TYPE_ENCODING_SHA256:
        raise ValueError("Frozen type-encoding sha256 differs from the contract")
    return encoding


def allowed_elements(data_root):
    # Use Crystalite's exact mask rule on its prebuilt, unaugmented train items.
    # Avoid MP20Tokens.__init__, which can create directories or preprocess data.
    path = Path(data_root) / "processed/mp20_tokens_train.pt"
    items = torch.load(path, map_location="cpu", weights_only=False)
    allowed = compute_allowed_elements(items)
    if allowed is None:
        raise ValueError(f"Empty allowed-element mask from {path.resolve()}")
    return allowed


def autocast_dtype(cfg, device):
    if torch.device(device).type != "cuda":
        return None
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg.precision]


def skeleton_batch(skeletons, tables, *, K, device):
    rows = torch.zeros((len(skeletons), K), dtype=torch.long, device=device)
    groups = torch.tensor([s.G for s in skeletons], dtype=torch.long, device=device)
    for i, skeleton in enumerate(skeletons):
        if not 1 <= len(skeleton.rows) <= K:
            raise ValueError(f"Skeleton does not fit K={K}")
        rows[i, :len(skeleton.rows)] = torch.tensor(skeleton.rows, device=device)
    pad = rows == 0
    return dict(G=groups, row=rows, pad_mask=pad, family=tables.group_family[groups].long(),
                dof_mask=(torch.arange(3, device=device) < tables.row_dof[rows][..., None])
                & ~pad[..., None])


@torch.no_grad()
def sample_pool(model, ema, cfg, tables, tb, encoding, allowed_mask, bank, *, n, seed,
                device, chunk=256, sampler=None):
    """Draw skeletons, sample with EMA, decode and retain failures in place.

    ``seed`` is used as given; in-training evaluation passes the configured sample seed. Failed
    decodes remain padded rows, so all evaluation denominators still equal n.
    ``decode_overlap="thread"`` decodes on one CPU worker while the next chunk samples;
    ``"none"`` keeps synchronous decoding. Both accumulate identical outputs in chunk order.
    """
    if n < 1 or chunk < 1:
        raise ValueError("n and chunk must be positive")
    if cfg.decode_impl not in ("reference", "fast"):
        raise ValueError("decode_impl must be reference or fast")
    if cfg.decode_overlap not in ("none", "thread"):
        raise ValueError("decode_overlap must be none or thread")
    # orbitol.sample.decode_batch is the patch point for tools and tests: a patched seam is used as is,
    # whatever decode_impl says. Unpatched, "fast" selects the fast decoder (identical outputs).
    decoder = decode_batch
    if cfg.decode_impl == "fast" and decoder is _REFERENCE_DECODE:
        from orbitol.decode_fast import decode_batch as decoder
    # Resolved per call, so tools that patch orbitol.sample.orbit_edm_sampler keep working.
    sampler = orbit_edm_sampler if sampler is None else sampler
    settings = {**asdict(cfg), "num_steps": cfg.sample_num_steps,
                "aa_rho_types": 0.0, "autocast_dtype": autocast_dtype(cfg, device)}
    skeletons = bank.sample(n, generator=np.random.default_rng(seed))
    if cfg.sample_token_budget:
        from orbitol.packed import PackedDenoiser, token_chunks

        chunks = token_chunks(skeletons, cfg.sample_token_budget)
    else:
        chunks = ((start, min(start + chunk, n)) for start in range(0, n, chunk))
    generator = torch.Generator(device=device).manual_seed(seed)
    rows, outputs, contains, requested, failures = [], [], [], [], {}

    def accumulate(start, output, decoded):
        rows.append(decoded.rows)
        outputs.append(output)
        contains.append(torch.from_numpy(decoded.contains_a))
        requested.append(torch.from_numpy(decoded.requested_G))
        failures.update({start + i: reason for i, reason in decoded.failures.items()})

    pending = deque()

    def consume_pending():
        start, output, future = pending.popleft()
        accumulate(start, output, future.result())

    was_training = model.training
    executor = None
    with isolated_rng():
        backup = ema.apply(model)
        model.eval()
        try:
            if cfg.decode_overlap == "thread":
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="orbitol-decode")
            for start, end in chunks:
                # Collect completed chunks in order without delaying the next GPU launch.
                while pending and pending[0][2].done():
                    consume_pending()
                selected = skeletons[start:end]
                batch = skeleton_batch(selected, tb, K=cfg.K, device=device)
                batch["index"] = torch.arange(start, end, device=device)
                kwargs = {}
                if cfg.sample_token_budget:
                    adapter = PackedDenoiser(model, batch, token_budget=cfg.sample_token_budget,
                                             compile_mode=cfg.compile_trunk,
                                             autocast_dtype=settings["autocast_dtype"],
                                             skip_type_scaling=settings.get("skip_type_scaling", False))
                    kwargs["denoise_fn"] = adapter.denoise
                output = sampler(model, batch, tb, objective=cfg.objective,
                                 settings=settings, generator=generator, **kwargs)
                output = {k: v.cpu() for k, v in output.items()}
                if executor is None:
                    accumulate(start, output, decoder(output, selected, tables, encoding, allowed_mask,
                                                      n_max=cfg.n_prim_max))
                else:
                    pending.append((start, output, executor.submit(decoder, output, selected, tables, encoding,
                                                                   allowed_mask, n_max=cfg.n_prim_max)))
            while pending:
                consume_pending()
        finally:
            try:
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
            finally:
                ema.restore(model, backup)
                model.train(was_training)
    return dict(rows={k: torch.cat([r[k] for r in rows]) for k in rows[0]},
                requested_G=torch.cat(requested), skeletons=[asdict(s) for s in skeletons],
                raw={k: torch.cat([o[k] for o in outputs]) for k in outputs[0]},
                contains_a=torch.cat(contains), failures=failures,
                failure_counts=dict(Counter(failures.values())),
                metadata=dict(n=n, seed=seed, chunk=chunk, weights="ema"))


def load_snapshot(snapshot_path, *, device, overrides=None):
    """Restore a snapshot as sample_pool's inputs plus metadata; call inside isolated_rng.

    ``overrides`` may change only SAMPLER_OVERRIDES, e.g. fp32 for a bf16 run on a GPU without bf16, or S_min.
    """
    overrides = dict(overrides or {})
    if not set(overrides) <= set(SAMPLER_OVERRIDES):
        raise ValueError(f"Sampler overrides are limited to {SAMPLER_OVERRIDES}: {sorted(overrides)}")
    checkpoint = torch.load(snapshot_path, map_location="cpu", weights_only=False)
    if checkpoint["core_manifest"] != json.loads((ROOT / "orbitol/core_manifest.json").read_text()):
        raise ValueError("Snapshot/core manifest mismatch")
    cfg = replace(C.TrainConfig(**C.normalize_historical_config(checkpoint["cfg"])), **overrides)
    tables = load_tables(cfg.extra.get("tables_dir") or C.TABLES_DIR)
    if checkpoint["table_sha256"] != tables.sha256:
        raise ValueError("Snapshot/table sha256 mismatch")
    if checkpoint["encoding_sha256"] != C.TYPE_ENCODING_SHA256:
        raise ValueError("Snapshot/type-encoding sha256 mismatch")
    encoding = load_encoding()
    paths = C.resolve_dataset_paths(cfg)
    bank = SkeletonBank.from_cache(Path(paths.orbit_cache_dir) / "train.npz")
    if bank.table_sha256 != tables.sha256 or bank.sha256 != checkpoint["bank_sha256"]:
        raise ValueError("Snapshot/skeleton-bank sha256 mismatch")
    tb = tables.torch(device)
    model = build_model(cfg, tb.row_desc, tb.group_desc, tables=tb).to(device)
    model.load_state_dict(checkpoint["model"])
    ema = EMA(model, cfg.ema_decay)
    ema.load_state_dict(checkpoint["ema"])
    inputs = dict(model=model, ema=ema, cfg=cfg, tables=tables, tb=tb, encoding=encoding,
                  allowed_mask=checkpoint["allowed_mask"], bank=bank)
    metadata = dict(snapshot=str(Path(snapshot_path).resolve()), step=checkpoint["step"],
                    table_sha256=checkpoint["table_sha256"],
                    encoding_sha256=checkpoint["encoding_sha256"], overrides=overrides)
    return inputs, metadata


def sample_from_checkpoint(snapshot_path, *, n, seed, device, chunk=256, overrides=None) -> dict:
    """Restore a snapshot and use precisely the train-time EMA sampler."""
    core.verify_core()
    with isolated_rng():
        inputs, metadata = load_snapshot(snapshot_path, device=device, overrides=overrides)
        result = sample_pool(**inputs, n=n, seed=seed, device=device, chunk=chunk)
        result["metadata"].update(metadata)
        return result


def write_cifs(result, directory) -> int:
    """Write each decoded structure of a pool as <attempt>.cif; failed attempts are skipped."""
    rows = {key: torch.as_tensor(result["rows"][key]).cpu() for key in ("A0", "F1", "Y1", "pad_mask")}
    n = len(rows["A0"])
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(n):
        if i in result["failures"] or bool(rows["pad_mask"][i].all()):
            continue
        try:
            structure = tokens_to_structure({key: value[i] for key, value in rows.items()})
        except Exception:  # an unbuildable lattice or species counts as a failed attempt, as in evaluation
            continue
        CifWriter(structure).write_file(directory / f"{i:0{len(str(n - 1))}d}.cif")
        written += 1
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--cif-dir", type=Path, help="also write each decoded structure as <attempt>.cif here")
    hints = get_type_hints(C.TrainConfig)
    for name in SAMPLER_OVERRIDES:  # omitted flags keep the snapshot's values
        kind = hints[name]
        parser.add_argument("--" + name.replace("_", "-"), dest=name, default=argparse.SUPPRESS,
                            **(dict(choices=get_args(kind)) if get_origin(kind) is Literal else dict(type=kind)))
    args = parser.parse_args(argv)
    overrides = {name: getattr(args, name) for name in SAMPLER_OVERRIDES if hasattr(args, name)}
    result = sample_from_checkpoint(args.snapshot, n=args.n, seed=args.seed,
                                    device=args.device, chunk=args.chunk, overrides=overrides)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    print(f"Saved {args.n} samples to {args.out.resolve()}; failures={result['failure_counts']}")
    if args.cif_dir:
        print(f"Wrote {write_cifs(result, args.cif_dir)} CIF files to {args.cif_dir.resolve()}")


if __name__ == "__main__":
    main()
