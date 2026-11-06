"""Matched Crystalite optimisation with resumable orbit batches.

Additional run controls live in TrainConfig.extra: val_batches (10), val_seed
(123), sample_seed (123), sample_chunk_size (256), and optional fixture paths
tables_dir, cache_dir, data_root, symmetry_reference. No references are built here.
"""
from __future__ import annotations

import argparse
import gc
from contextlib import contextmanager
from dataclasses import asdict, fields
from functools import partial
import itertools
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import traceback
from typing import get_args, get_origin, get_type_hints, Literal

import numpy as np
import spglib
import torch
from torch.utils.data import DataLoader

import orbitol.core as core
from orbitol import contracts as C, lattice
from orbitol.data import OrbitDataset, ResumableSampler, collate_orbits, file_sha256
from orbitol.edm import (PeriodicTarget, compute_orbit_loss, corrupt, denoise_orbit, map3,
                         orbit_noise, sample_sigma, wrap_exact)
from orbitol.encode import FastEncoder
from orbitol.eval_adapter import DNGEvaluator, ThermoSettings
from orbitol.eval_diagnostics import variant_metrics
from orbitol.model import build_model
from orbitol.optim import CachedClip, ExactAdamW, _MultiTensorUpdate
from orbitol.metric_names import log_view
from orbitol.posthoc_io import atomic_writer
from orbitol.sample import (ROOT, allowed_elements, autocast_dtype, isolated_rng,
                            load_encoding, restore_rng, rng_state, sample_pool)
from orbitol.skeleton_prior import SkeletonBank
from orbitol.symmetry_panel import ANGLE_TOLERANCE, SYMPRECS, reference_valid
from orbitol.tables import load_tables
from src.utils.ema import EMA as CoreEMA

DECODE_FAILURES = ("invalid_group", "invalid_shape", "empty_structure", "non_finite",
                   "invalid_rows", "reused_fixed_row", "invalid_species", "invalid_lattice",
                   "coinciding_orbits", "collapsed_orbit", "containment_failed", "failed_decode", "not_emitted",
                   "too_many_atoms", "decode_error")


class EMA(CoreEMA):
    """Keep the core's two rounded operations and checkpoint representation."""

    def __init__(self, model, decay, *, fast_update=False):
        super().__init__(model, decay)
        self.fast_update = fast_update
        self._batched_update = None
        self._model = None

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self._model = None
        self._batched_update = None

    def _cache_pairs(self, model):
        # Keep Parameter objects: detach() would retain old storage after .to()
        # or a .data replacement. Changing the parameter topology needs a new EMA.
        self._pairs = [(self.shadow[name], param) for name, param in model.named_parameters()
                       if name in self.shadow]
        self._shadows = [pair[0] for pair in self._pairs]
        self._parameters = [pair[1] for pair in self._pairs]
        self._model = model

    @torch.no_grad()
    def update(self, model):
        if self.fast_update:
            if model is not self._model:
                self._cache_pairs(model)
            pairs = self._pairs
        else:
            pairs = [(self.shadow[name], param.detach()) for name, param in model.named_parameters()
                     if name in self.shadow]
        if pairs:
            if (self.fast_update and all(t.device.type == "cuda" and t.dtype == torch.float32
                                        and t.is_contiguous() for pair in pairs for t in pair)
                    and len({t.device for pair in pairs for t in pair}) == 1):
                if self._batched_update is None:
                    self._batched_update = _MultiTensorUpdate()
                self._batched_update(pairs, decay=self.decay)
                return
            shadows, parameters = ((self._shadows, self._parameters) if self.fast_update else zip(*pairs))
            torch._foreach_mul_(shadows, self.decay)
            torch._foreach_add_(shadows, parameters, alpha=1.0 - self.decay)


def atomic_save(payload, path):
    """Keep the previous full checkpoint intact until replacement succeeds."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_resume(run_dir, resume):
    if str(resume) == "none":
        return None
    if str(resume) == "auto":
        path = Path(run_dir) / "checkpoints/latest.pt"
        return path if path.is_file() else None
    path = Path(resume).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def lr_multiplier(step, *, warmup_steps, max_steps):
    """The pinned train_crystalite.py warm-up and cosine, including endpoints."""
    maximum = max(1, max_steps)
    warmup = min(max(0, warmup_steps), maximum - 1)
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)
    if step >= maximum:
        return 0.0
    progress = (step - warmup) / max(1, maximum - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def verify_symmetry_reference(data_root, reference):
    """Reject stale/missing references before the evaluator can rebuild one."""
    reference = Path(reference).resolve()
    if not reference.is_file():
        raise FileNotFoundError(f"Prebuilt symmetry reference required: {reference}")
    source = (Path(data_root) / "processed/mp20_tokens_train.pt").resolve()
    items = torch.load(source, map_location="cpu", weights_only=False)
    stat = source.stat()
    expected = dict(schema=1, train_cache=str(source), size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, count=len(items), spglib=spglib.__version__,
                    symprecs=list(SYMPRECS), angle_tolerance=ANGLE_TOLERANCE,
                    decoder="crystalite-03603fe:tokens_to_structure")
    saved = json.loads(reference.read_text())
    histograms = saved.get("histograms", {})
    valid = all(len(histograms.get(str(p), [])) == 231
                and all(isinstance(c, int) and c >= 0 for c in histograms[str(p)])
                and sum(histograms[str(p)]) == len(items) for p in SYMPRECS)
    if not reference_valid(saved, expected, source) or not valid:
        raise ValueError(f"Stale symmetry reference; rebuild outside training: {reference}")


def validate_config(cfg):
    for name in ("batch_size", "max_steps", "K", "n_prim_max", "log_every", "sample_num_steps"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be positive")
    if cfg.num_workers < 0 or cfg.sample_count < 0 or cfg.seed < 0:
        raise ValueError("num_workers, sample_count and seed must be nonnegative")
    if cfg.objective not in C.OBJECTIVES or cfg.precision not in ("fp32", "bf16", "fp16"):
        raise ValueError("Invalid objective or precision")
    if C.OBJECTIVES[cfg.objective][0] == "posterior" and cfg.sigma_min <= 0:
        raise ValueError("Posterior objectives require sigma_min > 0 for the periodic target")
    if not 0 < cfg.ema_decay < 1:
        raise ValueError("EMA decay must be between zero and one for EMA sampling")
    if cfg.aa_rho_coords != 0 or cfg.aa_rho_lattice != 0:
        raise ValueError("Anti-annealing is not supported: aa_rho_coords and aa_rho_lattice must be 0")
    if len(cfg.sigma_data) != 3 or any(s <= 0 for s in cfg.sigma_data):
        raise ValueError("sigma_data requires three positive values")
    if len(cfg.loss_weights) != 3:
        raise ValueError("loss_weights requires three values")
    if cfg.extra.get("val_batches", 10) < 1 or cfg.extra.get("sample_chunk_size", 256) < 1:
        raise ValueError("val_batches and sample_chunk_size must be positive")
    if cfg.nequip_eval and cfg.sample_count > 0:
        if cfg.nequip_relax_mode not in ("batch", "sequential"):
            raise ValueError("nequip_relax_mode must be batch or sequential")
        if cfg.nequip_relax_batch_size < 1:
            raise ValueError("nequip_relax_batch_size must be positive")


class Trainer:
    """Single-device trainer; run(until_step=...) supports exact interruption tests."""

    def __init__(self, cfg, *, run_name, out_root=None, resume="auto",
                 device=None, no_wandb=False):
        validate_config(cfg)
        if Path(run_name).name != run_name or run_name in ("", ".", ".."):
            raise ValueError("run_name must be one directory name")
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        core.verify_core()
        self.tables = load_tables(cfg.extra.get("tables_dir") or C.TABLES_DIR)
        self.table_sha256 = self.tables.sha256
        self.encoding = load_encoding()
        root = Path(out_root or C.RUNS_ROOT).resolve()
        ancestor = root
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < 15_000_000_000:
            raise RuntimeError(f"At least 15 GB free is required under {root}")
        self.run_dir = root / run_name
        resume_path = resolve_resume(self.run_dir, resume)
        checkpoint = (torch.load(resume_path, map_location="cpu", weights_only=False)
                      if resume_path is not None else None)
        if checkpoint is None and (self.run_dir / "checkpoints/latest.pt").exists():
            raise FileExistsError(f"Existing run requires resume or a new name: {self.run_dir}")
        if checkpoint is not None:
            # Default-fill old configs (including dataset=mp20) before comparing.
            # Reject a dataset switch before opening the other dataset's caches.
            saved_cfg = asdict(C.TrainConfig(**C.normalize_historical_config(checkpoint["cfg"])))
            # Execution choices may change on resume: compilation, training padding (which preserves the RNG
            # stream), the sampling budget/decoder/overlap and the training-step speed toggles. Training
            # toggles can change floating-point rounding only; decoder and overlap choices preserve emitted rows.
            for name in ("compile_trunk", "pad_multiple", "sample_token_budget", "decode_impl", "decode_overlap",
                         "periodic_target_impl", "gem_compile", "fast_update", "fast_encode", "compile_step"):
                saved_cfg[name] = getattr(cfg, name)
            if saved_cfg != asdict(cfg):
                raise ValueError("Resume mismatch: cfg")
        paths = C.resolve_dataset_paths(cfg)
        cache_dir = Path(paths.orbit_cache_dir)
        self.train_data = OrbitDataset(cache_dir / "train.npz", self.tables, augment=True, aug_seed=cfg.seed)
        self.val_data = OrbitDataset(cache_dir / "val.npz", self.tables, augment=False, aug_seed=cfg.seed)
        for data in (self.train_data, self.val_data):
            if not len(data) or data.cache["n_orbits"].max() > cfg.K:
                raise ValueError("Empty dataset or orbit tokens exceed K")
            if data.cache["n_prim"].max() > cfg.n_prim_max:
                raise ValueError("Cached primitive atom count exceeds n_prim_max")
        self.fast_encoder = (FastEncoder(self.encoding, self.device,
                                         caches=(self.train_data.cache, self.val_data.cache))
                             if cfg.fast_encode else None)
        self.bank = SkeletonBank.from_cache(cache_dir / "train.npz")
        self.bank_sha256 = self.bank.sha256
        self.data_hashes = {split: file_sha256(cache_dir / f"{split}.npz") for split in ("train", "val")}
        for split, digest in self.data_hashes.items():
            manifest = json.loads((cache_dir / f"{split}.manifest.json").read_text())
            if manifest.get("table_sha256") != self.table_sha256 or manifest.get("cache_sha256") != digest:
                raise ValueError(f"Orbit cache manifest sha256 mismatch: {split}")
        data_root = paths.data_root
        self.allowed_mask = allowed_elements(data_root)
        self.evaluator = None
        self.nequip_model_sha256 = self.phase_diagram_sha256 = None
        if cfg.sample_count > 0 and cfg.eval_mode == "sync":
            reference = paths.symmetry_reference
            verify_symmetry_reference(data_root, reference)
            if cfg.nequip_eval:
                thermo = ThermoSettings.from_train_config(cfg, device=self.device)
                model_path = Path(thermo.model_path).expanduser().resolve()
                phase_diagram = Path(paths.phase_diagram).expanduser().resolve()
                for path in (model_path, phase_diagram):
                    if not path.is_file():
                        raise FileNotFoundError(path)
                self.nequip_model_sha256 = file_sha256(model_path)
                self.phase_diagram_sha256 = file_sha256(phase_diagram)
                relaxer = thermo.make_relaxer()
                self.evaluator = DNGEvaluator(
                    data_root, device=self.device, sample_count=cfg.sample_count,
                    sun_k=cfg.sun_k, thermo=thermo, relaxer=relaxer,
                    no_thermo_before_steps=0, symmetry_cache=reference,
                )
            else:
                self.evaluator = DNGEvaluator(data_root, device=self.device, sample_count=cfg.sample_count,
                                              sun_k=0, thermo=None, symmetry_cache=reference)
            self.evaluator.sample_seed = cfg.extra.get("sample_seed", 123)
            # Resolve the validated cache now, before any optimiser steps.
            self.evaluator.symmetry_panel.reference
        self.sampler = ResumableSampler(len(self.train_data), seed=cfg.seed, batch_size=cfg.batch_size)
        if not self.sampler.total_batches:
            raise ValueError("Training cache is smaller than one batch")
        self.commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.core_manifest = json.loads((ROOT / "orbitol/core_manifest.json").read_text())
        device_name = torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else "CPU"
        capability = torch.cuda.get_device_capability(self.device) if self.device.type == "cuda" else None
        nequip_hash = (f" nequip_model_sha256={self.nequip_model_sha256}"
                       f" phase_diagram_sha256={self.phase_diagram_sha256}"
                       if self.nequip_model_sha256 is not None else "")
        print(f"[startup] commit={self.commit} table_sha={self.table_sha256} "
              f"encoding_sha={C.TYPE_ENCODING_SHA256} core_manifest={ROOT / 'orbitol/core_manifest.json'} "
              f"core_sha={file_sha256(ROOT / 'orbitol/core_manifest.json')}{nequip_hash} "
              f"device={device_name} capability={capability}",
              flush=True)
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.tb = self.tables.torch(self.device)
        self.periodic_target = (PeriodicTarget(self.tb.row_H, self.tb.row_dof, impl=cfg.periodic_target_impl)
                                if C.OBJECTIVES[cfg.objective][0] == "posterior" else None)
        self.model = build_model(cfg, self.tb.row_desc, self.tb.group_desc, tables=self.tb).to(self.device)
        if cfg.compile_step:
            self._configure_train_forward()
        # ExactAdamW reproduces foreach AdamW bit for bit in one pass per tensor; it supports only zero weight decay.
        optimizer = ExactAdamW if cfg.weight_decay == 0 else torch.optim.AdamW
        optimizer_options = {"fast_update": cfg.fast_update} if optimizer is ExactAdamW else {}
        self.optimizer = optimizer(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                                   **optimizer_options)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, partial(lr_multiplier, warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps))
        if cfg.fast_update:
            self._clip = CachedClip(self.model.parameters())
            if isinstance(self.optimizer, ExactAdamW):
                self.optimizer.configure_schedule(self.scheduler, cfg.max_steps)
        self.ema = EMA(self.model, cfg.ema_decay, fast_update=cfg.fast_update)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda" and cfg.precision == "fp16")
        self.loader_generator = torch.Generator().manual_seed(cfg.seed)
        self.loader = DataLoader(self.train_data, batch_sampler=self.sampler, num_workers=cfg.num_workers,
                                 pin_memory=True, persistent_workers=cfg.num_workers > 0,
                                 collate_fn=partial(collate_orbits, K=cfg.K, pad_multiple=cfg.pad_multiple),
                                 generator=self.loader_generator)
        self.step, self.last_sample_step = 0, -1
        self.wandb_id = None
        if checkpoint is not None:
            for key, expected in (("table_sha256", self.table_sha256),
                                  ("encoding_sha256", C.TYPE_ENCODING_SHA256),
                                  ("bank_sha256", self.bank_sha256), ("data_hashes", self.data_hashes),
                                  ("core_manifest", self.core_manifest)):
                if checkpoint[key] != expected:
                    raise ValueError(f"Resume mismatch: {key}")
            if (self.nequip_model_sha256 is not None
                    and checkpoint.get("nequip_model_sha256") != self.nequip_model_sha256):
                raise ValueError("Resume mismatch: nequip_model_sha256")
            # Snapshots from before the hull hash was recorded resume without that check.
            if (self.phase_diagram_sha256 is not None and "phase_diagram_sha256" in checkpoint
                    and checkpoint["phase_diagram_sha256"] != self.phase_diagram_sha256):
                raise ValueError("Resume mismatch: phase_diagram_sha256")
            if not torch.equal(checkpoint["allowed_mask"], self.allowed_mask):
                raise ValueError("Resume mismatch: allowed elements")
            self.model.load_state_dict(checkpoint["model"])
            self.ema.load_state_dict(checkpoint["ema"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self.scaler.load_state_dict(checkpoint["scaler"])
            self.sampler.load_state_dict(checkpoint["sampler"])
            self.loader_generator.set_state(checkpoint["loader_rng"])
            self.step, self.last_sample_step = checkpoint["step"], checkpoint["last_sample_step"]
            self.wandb_id = checkpoint["wandb_id"]
        self.train_data.set_epoch(self.sampler.epoch)
        self._iterator = None
        self._resuming_iterator = checkpoint is not None
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with atomic_writer(self.run_dir / "config.json") as stream:
            stream.write((json.dumps(asdict(cfg), indent=2) + "\n").encode())
        (self.run_dir / "samples").mkdir(exist_ok=True)
        self.wandb_run = None
        # W&B setup and monitoring may use RNG internally. Never consume train draws.
        with isolated_rng():
            if not no_wandb:
                import wandb
                self.wandb_id = self.wandb_id or wandb.util.generate_id()
                project = cfg.wandb_project or C.dataset_spec(cfg.dataset).wandb_project
                self.wandb_run = wandb.init(project=project, name=cfg.wandb_name,
                                            entity=C.local_value("ORBITOL_WANDB_ENTITY"),
                                            id=self.wandb_id, resume="allow", config=asdict(cfg), dir=str(self.run_dir))
        if checkpoint is not None:
            restore_rng(checkpoint["rng"])
        self._save_requested, self._stop_requested = False, False
        # The loader's workers fork at the first batch; frozen objects (the evaluator's 540k-structure Alex novelty
        # reference) are never traversed by their garbage collector, so they stay shared instead of being copied.
        gc.freeze()

    def _next_batch(self):
        if self.sampler.cursor == self.sampler.total_batches:
            self.sampler.set_epoch(self.sampler.epoch + 1)
            self.train_data.set_epoch(self.sampler.epoch)
            self._iterator = None
            self._resuming_iterator = False
        if self._iterator is None:
            state = self.loader_generator.get_state()
            self._iterator = iter(self.loader)
            if self._resuming_iterator:
                self.loader_generator.set_state(state)
                self._resuming_iterator = False
        batch = next(self._iterator)
        if self.cfg.compile_step:
            # Device group lookup belongs to the whole forward region.
            return self._to_device(batch), None
        if self.periodic_target is not None and self.periodic_target.impl == "batched":
            batch = self._to_device(batch)
            return batch, self.periodic_target.groups(batch["row"])
        groups = self.periodic_target.groups(batch["row"]) if self.periodic_target is not None else None
        return self._to_device(batch), groups

    def _to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _configure_train_forward(self):
        """One callable with at most five static training specializations.

        Config validation bounds ceil(K / pad_multiple), or fixed K gives one
        width. The drop-last sampler fixes B; _loss checks B and width before
        tracing. No metadata, logging flags or step counters enter the graph.
        """
        cfg = self.cfg
        self._step_widths = (tuple(range(cfg.pad_multiple, cfg.K, cfg.pad_multiple)) + (cfg.K,)
                             if cfg.pad_multiple else (cfg.K,))
        self._run_train_forward = torch.compile(self._train_forward, fullgraph=True, dynamic=False,
                                                mode="reduce-overhead")

    def _train_forward(self, batch, sigma, noise):
        """Tensor-only encoding -> corruption -> denoising -> scalar losses.

        RNG and assertions stay in _loss. Only the tensors used by training
        metrics escape alongside losses; validation uses the original path.
        """
        cfg = self.cfg
        clean = self.fast_encoder(batch["A0"], batch["pad_mask"])
        state = corrupt(batch, clean, sigma, noise, self.tb, objective=cfg.objective)
        den = denoise_orbit(self.model, state, sigma, batch, self.tb, objective=cfg.objective,
                            sigma_data=cfg.sigma_data, autocast_dtype=autocast_dtype(cfg, self.device))
        groups = self.periodic_target.groups(batch["row"])
        target = self.periodic_target(state["u_noisy"], state["u_clean"], sigma, groups)
        losses = compute_orbit_loss(den, state, sigma, batch, self.tb, objective=cfg.objective,
                                    sigma_data=cfg.sigma_data, loss_weights=cfg.loss_weights, coord_target=target)
        return (losses, {"u_clean": state["u_clean"]},
                {key: den[key] for key in ("D_type", "D_u", "D_z")},
                self.model.member_capacity_ok(batch["row"], batch["pad_mask"]))

    def _loss(self, batch, generator=None, *, sigma=None, groups=None):
        cfg = self.cfg
        sampled_sigma = sigma is None
        if sigma is None:
            sigma = sample_sigma(len(batch["G"]), self.device, cfg.P_mean, cfg.P_std,
                                 cfg.sigma_min, cfg.sigma_max, generator=generator)
        if cfg.compile_step and sampled_sigma and torch.is_grad_enabled():
            if (batch["row"].shape[0] != cfg.batch_size
                    or batch["row"].shape[1] not in self._step_widths):
                raise ValueError("compile_step requires the configured batch_size and padded width buckets")
            tensors = {key: batch[key] for key in ("A0", "U", "lat", "row", "G", "family", "pad_mask", "dof_mask")}
            # Same eager draws and capacity slicing as the original path. Encoding
            # moves after the draws but consumes no RNG; compilation sees no RNG.
            noise = orbit_noise(batch, generator, K=cfg.K)
            losses, state, den, capacity_ok = self._run_train_forward(tensors, sigma, noise)
            if capacity_ok is not None:
                # Inductor drops graph-local _assert_async, so execute it here.
                torch._assert_async(capacity_ok, "primitive member count exceeds n_max")
            return losses, sigma, state, den
        encode = self.fast_encoder if cfg.fast_encode else self.encoding.encode_from_A0
        clean = encode(batch["A0"], batch["pad_mask"])
        state = corrupt(batch, clean, sigma, orbit_noise(batch, generator, K=cfg.K), self.tb,
                        objective=cfg.objective)
        den = denoise_orbit(self.model, state, sigma, batch, self.tb, objective=cfg.objective,
                            sigma_data=cfg.sigma_data, autocast_dtype=autocast_dtype(cfg, self.device))
        target = (self.periodic_target(state["u_noisy"], state["u_clean"], sigma, groups)
                  if sampled_sigma and groups is not None else None)
        losses = compute_orbit_loss(den, state, sigma, batch, self.tb, objective=cfg.objective,
                                    sigma_data=cfg.sigma_data, loss_weights=cfg.loss_weights, coord_target=target)
        return losses, sigma, state, den

    def _coord_diagnostics(self, batch, sigma, state, den, class_masks):
        """Reuse the corruption and loss with a zero coordinate correction."""
        hook = self.model.register_forward_hook(
            lambda module, args, raw: {**raw, "coord_vel": torch.zeros_like(raw["coord_vel"])})
        try:
            reference = denoise_orbit(
                self.model, state, sigma, batch, self.tb, objective=self.cfg.objective,
                sigma_data=self.cfg.sigma_data, autocast_dtype=autocast_dtype(self.cfg, self.device))
        finally:
            hook.remove()
        loss = partial(compute_orbit_loss, state=state, sigma=sigma, tb=self.tb,
                       objective=self.cfg.objective, sigma_data=self.cfg.sigma_data,
                       loss_weights=self.cfg.loss_weights)
        do_nothing = float(loss(reference, batch=batch)["F"])
        by_class = {}
        for name, mask in class_masks.items():
            if mask.any():
                # Only F is used; the denoiser and all targets keep the full batch.
                restricted = {**batch, "pad_mask": ~mask}
                by_class[name] = (float(loss(den, batch=restricted)["F"]),
                                  float(loss(reference, batch=restricted)["F"]))
        return do_nothing, by_class

    @torch.no_grad()
    def validate(self):
        seed = int(np.random.SeedSequence([self.cfg.extra.get("val_seed", 123), self.step]).generate_state(1)[0])
        loader = DataLoader(self.val_data, batch_size=self.cfg.batch_size, shuffle=False,
                            num_workers=0, collate_fn=partial(collate_orbits, K=self.cfg.K,
                                                           pad_multiple=self.cfg.pad_multiple),
                            generator=torch.Generator().manual_seed(seed))
        generator = torch.Generator(device=self.device).manual_seed(seed)
        # Separate draws preserve every existing validation metric and its RNG.
        grid_generator = torch.Generator(device=self.device).manual_seed(seed)
        # Extra points must not advance the legacy grid stream between batches.
        extra_grid_generator = torch.Generator(device=self.device).manual_seed(seed)
        sigma_grid = (0.02, 0.1, 0.5, 2, 10, 0.2, 0.3)
        grid_totals = {s: dict(A=0., F=0., Y=0.) for s in sigma_grid}
        coord_totals = {s: 0. for s in sigma_grid}
        class_totals = {s: {} for s in sigma_grid}
        row_dof = self.tb.row_dof
        row_classes = {f"d{d}": row_dof == d for d in (1, 2, 3)}
        # Table blocks beyond d are identity, so the full determinant is det(B[:d, :d]), one of 1, 1/2, 1/3 or 1/4.
        row_classes["wrap"] = torch.linalg.det(self.tb.row_lambda_basis.double()).abs() < 0.75
        offdiag = ~torch.eye(3, dtype=torch.bool, device=self.device)
        row_classes["nondiag"] = ((self.tb.row_H != 0) & offdiag).any(dim=(-2, -1))
        was_training = self.model.training
        totals = dict(total=0., A=0., F=0., Y=0.)
        count = 0
        with isolated_rng():
            self.model.eval()
            try:
                for batch in itertools.islice(loader, self.cfg.extra.get("val_batches", 10)):
                    batch = self._to_device(batch)
                    losses, _, _, _ = self._loss(batch, generator)
                    for key in totals:
                        totals[key] += float(losses[key])
                    active = ~batch["pad_mask"] & (row_dof[batch["row"]] > 0)
                    class_masks = {name: active & rows[batch["row"]]
                                   for name, rows in row_classes.items()}
                    for s in sigma_grid:
                        sigma = torch.full((len(batch["G"]),), float(s), device=self.device)
                        draws = extra_grid_generator if s in (0.2, 0.3) else grid_generator
                        losses, _, state, den = self._loss(batch, draws, sigma=sigma)
                        for key in grid_totals[s]:
                            grid_totals[s][key] += float(losses[key])
                        do_nothing, by_class = self._coord_diagnostics(batch, sigma, state, den, class_masks)
                        coord_totals[s] += do_nothing
                        for name, (value, reference) in by_class.items():
                            prev_value, prev_reference = class_totals[s].get(name, (0., 0.))
                            class_totals[s][name] = (prev_value + value, prev_reference + reference)
                    count += 1
            finally:
                self.model.train(was_training)
        metrics = {f"val/loss_{name}": totals[key] / count
                   for name, key in (("total", "total"), ("type", "A"), ("coord", "F"), ("lattice", "Y"))}
        for s in sigma_grid:
            for name, key, weight in zip(("type", "coord", "lattice"), ("A", "F", "Y"), self.cfg.loss_weights):
                metrics[f"val/sigma_{s}/loss_{name}"] = grid_totals[s][key] * weight / count
            metrics[f"val/sigma_{s}/coord_do_nothing"] = coord_totals[s] * self.cfg.loss_weights[1] / count
            # The common coordinate loss weight and batch count cancel in each ratio.
            if coord_totals[s] > 0:
                metrics[f"val/sigma_{s}/coord_ratio"] = grid_totals[s]["F"] / coord_totals[s]
            for name, (value, reference) in class_totals[s].items():
                if reference > 0:
                    metrics[f"val/sigma_{s}/coord_ratio/{name}"] = value / reference
        return metrics

    @torch.no_grad()
    def training_metrics(self, batch, losses, sigma, state, den):
        scalars = {"loss/total": losses["total"], "sigma/mean": sigma.mean(),
                   "sigma/std": sigma.std() if sigma.numel() > 1 else sigma.new_zeros(())}
        for name, key in zip(("type", "coord", "lattice"), ("A", "F", "Y")):
            scalars[f"loss/{name}"] = losses[key]
        real = ~batch["pad_mask"]
        predicted = self.encoding.decode_logits_to_A0(type_logits=den["D_type"], pad_mask=batch["pad_mask"])
        scalars["stats/type_acc"] = ((predicted == batch["A0"]) & real).sum() / real.sum().clamp_min(1)
        delta = wrap_exact(den["D_u"] - state["u_clean"], batch["row"], self.tb)
        cart = map3(self.tb.row_A[batch["row"]].to(delta), delta)
        scalars["stats/coord_l2"] = (cart.square() * real[..., None]).sum() / real.sum().clamp_min(1)
        scalars["stats/lattice_l2"] = (lattice.ell(den["D_z"], batch["family"]) - batch["lat"]).square().mean()
        values = torch.stack([value.double() for value in scalars.values()]).cpu().tolist()
        metrics = dict(zip(scalars, values))
        metrics["lr"] = self.optimizer.param_groups[0]["lr"]
        if self.scaler.is_enabled():
            metrics["perf/grad_scale"] = self.scaler.get_scale()
        for name, weight in zip(("type", "coord", "lattice"), self.cfg.loss_weights):
            metrics[f"loss/{name}"] *= weight
        return metrics

    def evaluate_samples(self):
        if self.cfg.eval_mode == "async":
            from orbitol.eval_jobs import submit_eval
            from orbitol.eval_snapshot import pending_eval_snapshots, write_eval_snapshot

            if len(pending_eval_snapshots(self.run_dir)) >= self.cfg.eval_backlog_max:
                self.last_sample_step = self.step
                return {"orbit/eval_skipped_backlog": 1}
            write_eval_snapshot(self, self.run_dir, self.step)
            job_id = submit_eval(self.run_dir, self.step, self.cfg, wandb=self.wandb_run is not None)
            metrics = {"orbit/eval_snapshot_step": self.step}
            if job_id:
                metrics["orbit/eval_job_id"] = job_id
            else:
                metrics["orbit/eval_submit_failed"] = 1
            self.last_sample_step = self.step
            return metrics
        started = time.perf_counter()
        with isolated_rng():
            result = sample_pool(self.model, self.ema, self.cfg, self.tables, self.tb, self.encoding,
                                 self.allowed_mask, self.bank, n=self.cfg.sample_count,
                                 seed=self.cfg.extra.get("sample_seed", 123),
                                 device=self.device, chunk=self.cfg.extra.get("sample_chunk_size", 256))
            result["metadata"].update(step=self.step, table_sha256=self.table_sha256,
                                      encoding_sha256=C.TYPE_ENCODING_SHA256)
            atomic_save(result, self.run_dir / "samples" / f"step_{self.step:07d}.pt")
            if self.cfg.nequip_eval:
                evaluation_started = time.perf_counter()
                metrics = {}
                try:
                    metrics = self.evaluator.evaluate(result["rows"], step=self.step,
                                                      requested_G=result["requested_G"])
                    records = self.evaluator.relaxation_diagnostics()
                    metrics.update(variant_metrics(
                        records, checked_count=metrics.get("joint/checked_count", len(records))))
                    path = self.run_dir / "samples" / f"step_{self.step:07d}.relaxed.jsonl"
                    with atomic_writer(path) as stream:
                        for record in records:
                            stream.write((json.dumps(record, allow_nan=False) + "\n").encode())
                except Exception:
                    metrics["orbit/nequip_error"] = 1
                    traceback.print_exc()
                finally:
                    metrics["orbit/nequip_seconds"] = time.perf_counter() - evaluation_started
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()
            else:
                metrics = self.evaluator.evaluate(result["rows"], step=self.step, requested_G=result["requested_G"])
            metrics["orbit/contains_a_rate"] = float(result["contains_a"].float().mean())
            reasons = set(DECODE_FAILURES) | result["failure_counts"].keys()
            metrics.update({f"orbit/failures/{key}": result["failure_counts"].get(key, 0) for key in reasons})
            metrics["orbit/eval_seconds"] = time.perf_counter() - started
            print(f"[sample step={self.step}] contains_a={metrics['orbit/contains_a_rate']:.6f} "
                  f"failures={result['failure_counts']}", flush=True)
            self.track_best_msun(metrics)
        self.last_sample_step = self.step
        return metrics

    def track_best_msun(self, metrics):
        """Keep the snapshot with the highest in-process MSUN as checkpoints/best_msun.pt (survives resume)."""
        msun = metrics.get(f"{self.evaluator.tag}/MSUN")
        if msun is None or not math.isfinite(msun):
            return
        record = self.checkpoint_dir / "best_msun.json"
        best = json.loads(record.read_text())["MSUN"] if record.exists() else -math.inf
        if msun > best:
            atomic_save(self.snapshot(), self.checkpoint_dir / "best_msun.pt")
            with atomic_writer(record) as stream:
                stream.write(json.dumps({"MSUN": msun, "step": self.step}).encode())
        metrics["orbit/best_msun"] = max(best, msun)

    def snapshot(self):
        payload = dict(format_version=1, model=self.model.state_dict(), ema=self.ema.state_dict(),
                       cfg=asdict(self.cfg), step=self.step, table_sha256=self.table_sha256,
                       encoding_sha256=C.TYPE_ENCODING_SHA256, core_manifest=self.core_manifest,
                       commit=self.commit, bank_sha256=self.bank_sha256, data_hashes=self.data_hashes,
                       allowed_mask=self.allowed_mask)
        if self.nequip_model_sha256 is not None:
            payload.update(nequip_model_sha256=self.nequip_model_sha256,
                           phase_diagram_sha256=self.phase_diagram_sha256)
        return payload

    def save(self, *, snapshot=False):
        payload = self.snapshot()
        if snapshot:
            atomic_save(payload, self.checkpoint_dir / f"step_{self.step:07d}.pt")
            if self.cfg.ckpt_keep > 0:
                # Keep the snapshot just written plus the newest ckpt_keep - 1 earlier ones.
                older = sorted(path for path in self.checkpoint_dir.glob("step_*.pt")
                               if path.stem[5:].isdigit() and int(path.stem[5:]) < self.step)
                for path in older[:max(0, len(older) - self.cfg.ckpt_keep + 1)]:
                    path.unlink(missing_ok=True)
        payload.update(optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
                       scaler=self.scaler.state_dict(), rng=rng_state(), sampler=self.sampler.state_dict(),
                       loader_rng=self.loader_generator.get_state(), wandb_id=self.wandb_id,
                       last_sample_step=self.last_sample_step)
        # Only latest.pt has optimizer state; its temporary replacement is the second full copy.
        atomic_save(payload, self.checkpoint_dir / "latest.pt")

    def _signal(self, signum, frame):
        # Saving mid-backward or between optimizer/scheduler/EMA updates is unsafe.
        self._save_requested = True
        self._stop_requested |= signum == signal.SIGTERM

    @contextmanager
    def _signals(self):
        previous = {sig: signal.signal(sig, self._signal) for sig in (signal.SIGUSR1, signal.SIGTERM)}
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def log(self, metrics):
        if not metrics:
            return
        metrics = log_view(metrics, legacy_subset=self.cfg.sun_k > 0)
        with isolated_rng():
            if self.wandb_run is not None:
                self.wandb_run.log(metrics, step=self.step)
            with (self.run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": self.step, **metrics}) + "\n")

    def _sampling_due(self):
        return self.step > 0 and self.cfg.sample_count > 0 and self.last_sample_step != self.step and (
            self.step == self.cfg.max_steps or
            (self.cfg.sample_frequency > 0 and self.step % self.cfg.sample_frequency == 0))

    def _handle_requested_save(self):
        if self._save_requested:
            self.save()
            self._save_requested = False
        return self._stop_requested

    def run(self, *, until_step=None, on_step=None):
        target = self.cfg.max_steps if until_step is None else min(until_step, self.cfg.max_steps)
        started, logged_step, seen = time.perf_counter(), self.step, 0
        self.model.train()
        with self._signals():
            # A pre-evaluation checkpoint can already be at max_steps. Complete
            # its pending evaluation even when there are no optimizer steps left.
            if self._sampling_due():
                self.log(self.evaluate_samples())
                self.save(snapshot=self.step == self.cfg.max_steps or
                          (self.cfg.ckpt_every > 0 and self.step % self.cfg.ckpt_every == 0))
            while self.step < target:
                if self._handle_requested_save():
                    return
                batch, groups = self._next_batch()
                # In-process sampling marks its own CUDA-graph steps, which turns off the
                # automatic step detection; unmarked training then re-recorded its graphs on
                # every step (~30 to 0.01 steps/s). Outputs of earlier steps are never read again.
                torch.compiler.cudagraph_mark_step_begin()
                losses, sigma, state, den = self._loss(batch, groups=groups)
                logging_step = (self.step + 1) % self.cfg.log_every == 0
                if self.cfg.compile_step:
                    # Copy outside the graph, before backward/replay. Callbacks
                    # may retain losses indefinitely; metrics need only these
                    # three predictions and u_clean. Ordinary steps copy nothing.
                    observed_losses = ({key: value.detach().clone() for key, value in losses.items()}
                                       if on_step is not None or logging_step else None)
                    if logging_step:
                        state = {key: value.detach().clone() for key, value in state.items()}
                        den = {key: value.detach().clone() for key, value in den.items()}
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(losses["total"]).backward()
                self.scaler.unscale_(self.optimizer)
                # fp16: on an overflow the scaler skips the update and halves its scale (perf/grad_scale).
                if self.cfg.fast_update:
                    norm = self._clip(self.cfg.grad_clip)
                    if not self.scaler.is_enabled():
                        # CPU fails immediately. CUDA aborts the stream before the update;
                        # the host sees it at a checked launch/sync (at latest logging/save).
                        # A device assertion is fatal: restart from the last good checkpoint.
                        torch._assert_async(torch.isfinite(norm), "non-finite gradient norm (fast_update)")
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip,
                                                   error_if_nonfinite=not self.scaler.is_enabled())
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                if self.cfg.fast_update and isinstance(self.optimizer, ExactAdamW):
                    self.optimizer.finish_schedule_step()
                self.ema.update(self.model)
                self.sampler.mark_consumed()
                self.step += 1
                seen += len(batch["G"])
                if on_step is not None:
                    on_step(self.step, observed_losses if self.cfg.compile_step else losses)
                metrics = {}
                if logging_step:
                    metrics.update(self.training_metrics(batch, observed_losses if self.cfg.compile_step else losses,
                                                         sigma, state, den))
                    elapsed = time.perf_counter() - started
                    metrics.update({"perf/steps_per_second": (self.step - logged_step) / elapsed,
                                    "perf/samples_per_second": seen / elapsed,
                                    "perf/gpu_memory_allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9
                                    if self.device.type == "cuda" else 0.,
                                    "perf/gpu_memory_reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9
                                    if self.device.type == "cuda" else 0.})
                    if getattr(self.model, "lll_exceedances", None) is not None:
                        # Cells still unreduced after the fixed compiled LLL sweeps (expected 0).
                        metrics["perf/lll_cap_exceedances"] = int(self.model.lll_exceedances)
                    print(f"[step={self.step}] loss={metrics['loss/total']:.6g} "
                          f"steps/s={metrics['perf/steps_per_second']:.3f}", flush=True)
                    started, logged_step, seen = time.perf_counter(), self.step, 0
                # Save a coherent state promptly on signals, before potentially long evaluation.
                if self._handle_requested_save():
                    self.log(metrics)
                    return
                if self.cfg.val_every > 0 and self.step % self.cfg.val_every == 0:
                    metrics.update(self.validate())
                if self._handle_requested_save():
                    self.log(metrics)
                    return
                if self._sampling_due():
                    # Even a signal during a long evaluation has this resumable boundary on disk.
                    if self.cfg.eval_mode == "sync":
                        self.save()
                    metrics.update(self.evaluate_samples())
                self.log(metrics)
                if self.cfg.ckpt_every > 0 and self.step % self.cfg.ckpt_every == 0:
                    self.save(snapshot=True)
                if self.cfg.compile_step:
                    del losses, state, den, observed_losses
            self.save(snapshot=self.step == self.cfg.max_steps)

    def close(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()


def _boolean(value):
    if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
        raise argparse.ArgumentTypeError("expected true or false")
    return value.lower() in ("true", "1", "yes")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path,
                        help="JSON object of TrainConfig fields; explicit flags override it")
    result.add_argument("--run-name", required=True)
    result.add_argument("--out-root", default=None)
    result.add_argument("--resume", default="auto")
    result.add_argument("--no-wandb", action="store_true")
    result.add_argument("--device", default=None)
    hints = get_type_hints(C.TrainConfig)
    for field in fields(C.TrainConfig):
        annotation = hints[field.name]
        kwargs = dict(default=argparse.SUPPRESS, dest=field.name)
        if get_origin(annotation) is Literal:
            kwargs.update(type=str, choices=get_args(annotation))
        elif get_origin(annotation) is tuple:
            kwargs.update(type=float, nargs=len(get_args(annotation)))
        else:
            kwargs["type"] = json.loads if annotation is dict else _boolean if annotation is bool else annotation
        options = list(dict.fromkeys(("--" + field.name, "--" + field.name.replace("_", "-"))))
        result.add_argument(*options, **kwargs)
    return result


def main(argv=None):
    args = vars(parser().parse_args(argv))
    config = args.pop("config")
    controls = {key: args.pop(key) for key in ("run_name", "out_root", "resume", "no_wandb", "device")}
    controls["out_root"] = controls["out_root"] or C.RUNS_ROOT
    resume_path = resolve_resume(Path(controls["out_root"]) / controls["run_name"], controls["resume"])
    # Resolve model-size autos only after applying config and CLI overrides.
    base = dict(wandb_name=controls["run_name"])
    if resume_path is not None:
        base = C.normalize_historical_config(
            torch.load(resume_path, map_location="cpu", weights_only=False)["cfg"])
    merged = {**base, **(json.loads(config.read_text()) if config else {}), **args}
    merged = C.drop_retired(merged)
    for key in ("sigma_data", "loss_weights"):
        if key in merged:
            merged[key] = tuple(merged[key])
    trainer = Trainer(C.TrainConfig(**merged), **controls)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
