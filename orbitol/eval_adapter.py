"""Crystalite 03603fe train-time DNG scalar metrics for emitted samples.

The default protocol relaxes every constructed sample and reports SUN/MSUN over
all attempts, using relaxed-endpoint uniqueness and novelty. Positive ``sun_k``
explicitly selects the historical subset estimator instead.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch

import orbitol.core as core  # must precede every src import
from orbitol.contracts import DATASET_NAMES, dataset_spec, local_path, resolve_dataset_paths
from orbitol.eval_protocol import (
    RelaxedRecord, _construct, _raw_metrics, _requested_groups, _with_core_fallback,
    joint_metrics, relax_all, run_in_child, score,
)
from orbitol.symmetry_panel import SymmetryPanel
from src.eval.dng_eval import compute_sun_metrics
from src.eval.stability import load_phase_diagram
from src.utils.dataset import dataset_to_structures
from src.utils.sample_stats import (
    compute_e_above_hull_mp2020_like,
    compute_e_above_hull_uncorrected,
)
from src.utils.stability_logger import StabilityLogger, _ThermoConfig


@dataclass(frozen=True)
class ThermoSettings:
    """Fixed-step FIRE settings shared by synchronous and queued evaluation.

    The backends ignore nequip_fmax in batch mode; there is no early stopping.
    ``thermo=None`` on DNGEvaluator disables thermodynamics entirely.
    """

    thermo_stability_check: bool = True
    thermo_stability_count: int = 0
    thermo_stability_batch: int = 128
    thermo_relax_steps: int = 500
    thermo_ppd_mp: str = ""
    thermo_stability_device: str = "cuda"
    thermo_ehull_method: str = "mp2020_like"
    thermo_mlip: str = "nequip"
    nequip_backend: Literal["oeq", "torchscript"] = "oeq"
    nequip_oeq_path: str = ""  # lazily resolve the OEQ model artifact
    nequip_model_kind: Literal["aoti", "torchscript"] = "torchscript"
    nequip_ts_path: str = ""
    nequip_compile_path: str = ""
    nequip_relax_mode: str = "batch"
    nequip_optimizer: str = "FIRE"
    nequip_cell_filter: str = "frechet"
    nequip_fmax: float = 0.005
    nequip_max_force_abort: float = 1e6

    def __post_init__(self):
        if self.thermo_stability_count != 0:
            raise ValueError("thermo_stability_count must be 0: the all-relaxed protocol has no count cap")
        if self.nequip_backend not in {"oeq", "torchscript"}:
            raise ValueError(f"Unsupported NequIP backend: {self.nequip_backend}")
        if self.nequip_model_kind not in {"aoti", "torchscript"}:
            raise ValueError(f"Unsupported NequIP model kind: {self.nequip_model_kind}")
        if self.nequip_relax_mode not in {"batch", "sequential"}:
            raise ValueError(f"Unsupported relaxation mode: {self.nequip_relax_mode}")
        for name in ("thermo_relax_steps", "thermo_stability_batch"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

        # Persist the historical resolved strings in pool/checkpoint settings.
        if not self.thermo_ppd_mp:
            object.__setattr__(self, "thermo_ppd_mp", dataset_spec("mp20").phase_diagram)
        for name in ("nequip_ts_path", "nequip_compile_path"):
            if not getattr(self, name):
                object.__setattr__(self, name, local_path("ORBITOL_NEQUIP_TS"))

    @property
    def model_path(self):
        if self.nequip_model_kind == "aoti":
            return self.nequip_compile_path
        if self.nequip_backend == "oeq":
            if self.nequip_oeq_path:
                return self.nequip_oeq_path
            from orbitol.nequip_relax import DEFAULT_OEQ_PATH
            return str(DEFAULT_OEQ_PATH)
        return self.nequip_ts_path

    @classmethod
    def from_train_config(cls, cfg, *, device):
        paths = resolve_dataset_paths(cfg)
        kwargs = dict(nequip_backend=cfg.nequip_backend, thermo_relax_steps=cfg.thermo_relax_steps,
                      nequip_relax_mode=cfg.nequip_relax_mode, thermo_stability_device=str(device),
                      thermo_ppd_mp=paths.phase_diagram, thermo_stability_batch=cfg.nequip_relax_batch_size)
        if cfg.extra.get("nequip_model"):
            kwargs["nequip_oeq_path" if cfg.nequip_backend == "oeq" else "nequip_ts_path"] = cfg.extra["nequip_model"]
        return cls(**kwargs)

    def make_relaxer(self):
        from orbitol.nequip_relax import RobustRelaxer, make_batch_relaxer, make_sequential_relaxer
        if self.thermo_mlip != "nequip" or self.nequip_model_kind != "torchscript":
            return None  # the core owns the historical AOTI/CHGNet loaders
        if self.nequip_optimizer != "FIRE" or self.nequip_cell_filter != "frechet":
            raise ValueError("TorchScript relaxation requires FIRE and frechet")
        kwargs = dict(max_force_abort=self.nequip_max_force_abort)
        if self.nequip_backend == "oeq":
            from orbitol.nequip_relax import make_fast_batch_relaxer
            if self.nequip_relax_mode != "batch":
                raise ValueError("oeq requires batch relaxation")
            factory = make_fast_batch_relaxer
            kwargs["batch_size"] = self.thermo_stability_batch
        elif self.nequip_relax_mode == "sequential":
            factory = make_sequential_relaxer
            kwargs["fmax"] = self.nequip_fmax
        else:
            factory = make_batch_relaxer
        return RobustRelaxer(factory(self.model_path, self.thermo_stability_device, **kwargs))

    def to_core(self) -> _ThermoConfig:
        # train_crystalite.py:387-409. No checkpoint/model_args lookup.
        return _ThermoConfig(
            batch_size=max(1, int(self.thermo_stability_batch)),
            relax_steps=int(self.thermo_relax_steps),
            ppd_path=str(self.thermo_ppd_mp),
            device=str(self.thermo_stability_device),
            ehull_method=self.thermo_ehull_method,
            mlip=self.thermo_mlip,
            nequip_compile_path=str(self.nequip_compile_path),
            nequip_relax_mode=self.nequip_relax_mode,
            nequip_optimizer=self.nequip_optimizer,
            nequip_cell_filter=self.nequip_cell_filter,
            nequip_fmax=float(self.nequip_fmax),
            nequip_max_force_abort=float(self.nequip_max_force_abort),
        )


class _ConfiguredLogger(StabilityLogger):
    """Use the original logger, with an optional CPU test backend injection."""

    def __init__(self, settings: ThermoSettings, relaxer=None):
        cfg = settings.to_core()
        if relaxer is None:
            super().__init__(gamma_cfg=None, thermo_cfg=cfg)
            return
        # Injection replaces backend creation only. Hull corrections, relaxation
        # batching, failure handling and aggregation still run the core code.
        # Mirrors stability_logger.py:87-163 without loading an MLIP.
        super().__init__(gamma_cfg=None, thermo_cfg=None)
        if cfg.mlip not in {"chgnet", "nequip"}:
            raise ValueError(f"Unsupported thermo backend: {cfg.mlip}")
        if cfg.ehull_method not in {"uncorrected", "mp2020_like"}:
            raise ValueError(f"Unsupported hull method: {cfg.ehull_method}")
        if cfg.nequip_relax_mode not in {"sequential", "batch"}:
            raise ValueError(f"Unsupported relaxation mode: {cfg.nequip_relax_mode}")
        self.thermo_cfg = cfg
        self._thermo_relaxer = relaxer
        self._thermo_backend = cfg.mlip
        self._thermo_nequip_relax_mode = cfg.nequip_relax_mode
        if cfg.mlip == "nequip" and cfg.nequip_relax_mode == "batch":
            self._thermo_batch_relaxer = relaxer
        self._thermo_ehull_method = cfg.ehull_method
        if cfg.ehull_method == "mp2020_like":
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
            self._thermo_mp2020_compat = MaterialsProject2020Compatibility(check_potcar=False)
        self._thermo_ppd = load_phase_diagram(cfg.ppd_path)
        self._thermo_ready = True


class _CaptureRelaxer:
    """Return the backend's original results unchanged, retaining endpoints."""

    def __init__(self, backend, logger):
        self.backend, self.logger = backend, logger

    def relax(self, structure, **kwargs):
        try:
            result = self.backend.relax(structure, **kwargs)
        except Exception as exc:
            self.logger.capture_failure(structure, exc)
            raise
        self.logger.capture(structure, result)
        return result

    def relax_many(self, structures, **kwargs):
        try:
            results = self.backend.relax_many(structures, **kwargs)
        except Exception as exc:
            # A failed batch may succeed after the core bisects it. Retain
            # only failures the core attributes to individual structures.
            indices = self.logger._extract_batched_failure_indices(exc, len(structures))
            if len(structures) == 1:
                indices = [0]
            for index in indices:
                self.logger.capture_failure(structures[index], exc)
            raise
        if len(results) == len(structures):
            for structure, result in zip(structures, results):
                self.logger.capture(structure, result)
        return results


class CapturingStabilityLogger(_ConfiguredLogger):
    """Composition around relaxation; the core logger's metric path is intact."""

    def __init__(self, settings: ThermoSettings, relaxer=None):
        super().__init__(settings, relaxer)
        wrapper = _CaptureRelaxer(self._thermo_relaxer, self)
        self._thermo_relaxer = wrapper
        if self._thermo_batch_relaxer is not None:
            self._thermo_batch_relaxer = wrapper
        self.begin({})

    def begin(self, index_by_identity: dict[int, tuple[int, bool]]) -> None:
        self.index_by_identity = index_by_identity
        self.attempted: dict[int, bool] = {}
        self.endpoints: dict[int, tuple[bool, Any]] = {}
        self.failures: dict[int, tuple[str, str]] = {}
        self.initial_volume_per_atom: dict[int, float] = {}

    def update(self, structures, **kwargs):
        structures = list(structures)
        if kwargs.get("enabled", False):
            for structure in structures:
                pair = self.index_by_identity.get(id(structure))
                if pair is not None:
                    index, is_un = pair
                    self.attempted[index] = is_un
                    self.initial_volume_per_atom[index] = (
                        float(structure.volume / len(structure)) if len(structure) else float("nan")
                    )
        return super().update(structures, **kwargs)

    def capture(self, structure, result) -> None:
        pair = self.index_by_identity.get(id(structure))
        if pair is not None:
            index, is_un = pair
            self.endpoints[index] = (is_un, result)
            self.failures.pop(index, None)

    def capture_failure(self, structure, exc) -> None:
        pair = self.index_by_identity.get(id(structure))
        if pair is not None:
            self.failures[pair[0]] = (self._thermo_exception_reason(exc)[0], str(exc))

    def records(self) -> list[RelaxedRecord]:
        # record_relaxation is a closure (stability_logger.py:390-416), so the
        # extra per-index hull values are recomputed using the same core helper
        # after aggregation. Nothing is injected into that closure or module.
        records = []
        for index, (is_un, result) in self.endpoints.items():
            final = result.get("final_structure")
            if final is None:
                continue
            try:
                energy = result["trajectory"].energies[-1]
                if self._thermo_ehull_method == "mp2020_like":
                    ehull, reason = compute_e_above_hull_mp2020_like(
                        self._thermo_ppd, final, energy,
                        mp2020_compat=self._thermo_mp2020_compat,
                    )
                else:
                    ehull, reason = compute_e_above_hull_uncorrected(self._thermo_ppd, final, energy)
            except Exception as exc:
                ehull, reason = None, f"exception:{type(exc).__name__}"
            ok = reason is None and ehull is not None and np.isfinite(ehull)
            if not ok and reason is None:
                ehull, reason = None, "non_finite_hull"
            records.append(RelaxedRecord(
                index, final, ehull, bool(ok and ehull <= 0.0),
                bool(ok and ehull <= 0.1), is_un, reason,
            ))
        return records


@lru_cache(maxsize=1)
def _verify_pin() -> None:
    core.verify_core()


def _load_items(path: Path) -> list:
    # Direct cache loading is intentional: MP20Tokens.__init__ can preprocess
    # absent caches. This has the same items, without any input-directory writes.
    if not path.is_file():
        raise FileNotFoundError(f"Required token cache missing (no preprocessing): {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _sample_items(samples: dict) -> list[dict]:
    values = {key: torch.as_tensor(samples[key]).detach().cpu() for key in ("A0", "F1", "Y1", "pad_mask")}
    a, f, y, mask = (values[k] for k in ("A0", "F1", "Y1", "pad_mask"))
    if a.ndim != 2 or not 1 <= a.shape[1] <= 20:
        raise ValueError("A0 must have shape (S,N), 1 <= N <= 20")
    s, n = a.shape
    if f.shape != (s, n, 3) or y.shape != (s, 6) or mask.shape != (s, n):
        raise ValueError("Samples must use the emitted A0/F1/Y1/pad_mask shapes")
    if a.dtype != torch.int64 or mask.dtype != torch.bool or not f.is_floating_point() or not y.is_floating_point():
        raise ValueError("Expected int64 A0, bool pad_mask, and floating F1/Y1")
    return [{key: value[i] for key, value in values.items()} for i in range(s)]


class DNGEvaluator:
    """Build once per run. ``symmetry_cache`` defaults to the current run dir.

    Backend initialization is lazy until the first thermo-eligible evaluation.
    ``relaxer`` must implement the core's relax/relax_many protocol and return
    final_structure plus trajectory.energies; the configured hull is still used.
    """

    tag = "sample_ema"
    sample_seed = 123

    def __init__(self, data_root, *, device, sample_count=10000, sun_k=0,
                 no_thermo_before_steps=0, thermo: ThermoSettings | None,
                 relaxer=None, symmetry_cache: str | Path | None = None):
        _verify_pin()
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if sun_k < 0:
            raise ValueError("sun_k must be nonnegative (0 relaxes all)")
        self.data_root = Path(data_root).resolve()
        self.device = torch.device(device)
        self.sample_count, self.sun_k = int(sample_count), int(sun_k)
        self.no_thermo_before_steps = int(no_thermo_before_steps)
        self.thermo, self.relaxer = thermo, relaxer
        self.thermo_logger = self._legacy_logger = None
        self._ppd = None
        self.per_sample: list[dict] = []
        self.relaxed_records: list[RelaxedRecord] = []
        self.attempted_indices: dict[int, bool] = {}
        self.raw_validity: dict[int, dict[str, bool]] = {}
        # train_crystalite.py:315-384: novelty uses train; Wasserstein uses val
        # (train fallback only when val is empty). Never auto-create a split.
        train_path = self.data_root / "processed/mp20_tokens_train.pt"
        train = _load_items(train_path)
        val = _load_items(self.data_root / "processed/mp20_tokens_val.pt")
        if not train:
            raise ValueError("Metric reference train split is empty")
        self.novelty_ref_structs = dataset_to_structures(train)
        self.ref_structs = dataset_to_structures(val) if val else self.novelty_ref_structs
        self.symmetry_panel = SymmetryPanel(
            train_path, train, symmetry_cache or Path.cwd() / "symmetry_reference.json",
        )

    @property
    def references(self):
        # Warm the only lazy reference before calling the pure scoring function.
        self.symmetry_panel.reference
        return dict(ref_structs=self.ref_structs, novelty_ref_structs=self.novelty_ref_structs,
                    symmetry_panel=self.symmetry_panel)

    @property
    def settings(self):
        return dict(sample_count=self.sample_count, sample_seed=self.sample_seed, sun_k=self.sun_k,
                    no_thermo_before_steps=self.no_thermo_before_steps,
                    thermo=asdict(self.thermo) if self.thermo is not None else None)

    @property
    def ppd(self):
        if self.thermo is not None and self._ppd is None:
            self._ppd = load_phase_diagram(self.thermo.thermo_ppd_mp)
        return self._ppd

    def evaluate(self, samples, *, step, requested_G=None) -> dict[str, float]:
        items = _with_core_fallback(_sample_items(samples))[:self.sample_count]
        requested_G = _requested_groups(requested_G, len(items))
        self.relaxed_records, self.attempted_indices, self.raw_validity = [], {}, {}
        self.per_sample, self.thermo_logger = [], None
        enabled = (self.thermo is not None and self.thermo.thermo_stability_check
                   and step >= self.no_thermo_before_steps)
        if self.sun_k > 0:
            return self._evaluate_legacy(items, step=step, requested_G=requested_G, enabled=enabled)
        relaxed = None
        if enabled:
            if self.relaxer is None:
                self.relaxer = self.thermo.make_relaxer()
                if self.relaxer is None:
                    # Retain the core's explicitly selected AOTI/CHGNet loaders.
                    logger = _ConfiguredLogger(self.thermo)
                    self.relaxer = logger._thermo_batch_relaxer or logger._thermo_relaxer
            relaxed = relax_all(items, self.relaxer, self.thermo.thermo_relax_steps)
        metrics, self.per_sample = run_in_child(
            score, items, relaxed, references=self.references, ppd=self.ppd if enabled else None,
            settings=self.settings, tag=self.tag, step=step, requested_G=requested_G,
        )
        self.raw_validity = {row["index"]: {key: row[key] for key in
                            ("raw_struct_valid", "raw_comp_valid", "raw_valid")} for row in self.per_sample}
        if relaxed is not None:
            self._retain_diagnostics(items, relaxed)
        return metrics

    def _retain_diagnostics(self, items, relaxed):
        # Preserve the existing observational API from the same saved records.
        # No second hull calculation or relaxation is needed.
        from orbitol.eval_diagnostics import RELAXATION_FIELDS
        raw, _ = _construct(items)
        endpoints, failures, volumes = {}, {}, {}
        for record, row in zip(relaxed, self.per_sample, strict=True):
            index = row["index"]
            if not record["constructed"]:
                continue
            self.attempted_indices[index] = row["relaxed_is_un"]
            volumes[index] = float(raw[index].volume / len(raw[index])) if len(raw[index]) else float("nan")
            if record["relax_status"] == "ok":
                self.relaxed_records.append(RelaxedRecord(
                    index, record["final_structure"], row["e_above_hull"], row["stable"], row["metastable"],
                    row["relaxed_is_un"], row["hull_failure_reason"],
                ))
                endpoints[index] = (row["relaxed_is_un"], dict(
                    final_structure=record["final_structure"], nsteps=record["nsteps"],
                    trajectory=SimpleNamespace(energies=[record["energy_eV"]]),
                    final_max_force=record.get("final_max_force"),
                    **{key: row.get(key) for key in RELAXATION_FIELDS},
                ))
            else:
                reason = StabilityLogger._thermo_exception_reason(RuntimeError(record["reason"]))[0]
                failures[index] = (reason, record["reason"])
        self.thermo_logger = SimpleNamespace(endpoints=endpoints, failures=failures, initial_volume_per_atom=volumes)

    def _evaluate_legacy(self, items, *, step, requested_G, enabled):
        metrics, rows, raw, novelty, _ = _raw_metrics(
            items, references=self.references, settings=self.settings, tag=self.tag)
        self.raw_validity = {row["index"]: {key: row[key] for key in
                            ("raw_struct_valid", "raw_comp_valid", "raw_valid")} for row in rows}
        if enabled:
            if self._legacy_logger is None:
                relaxer = self.relaxer if self.relaxer is not None else self.thermo.make_relaxer()
                self._legacy_logger = CapturingStabilityLogger(self.thermo, relaxer)
            self.thermo_logger = logger = self._legacy_logger
            constructed_indices = [i for i, structure in enumerate(raw) if structure is not None]
            logger.begin({id(structure): (i, rows[i]["raw_is_un"])
                          for i, structure in zip(constructed_indices, novelty.novelty_structs, strict=True)})
            if novelty.novelty_metrics:
                result = compute_sun_metrics(
                    novelty.novelty_metrics, thermo_logger=logger, tag=self.tag, step=step,
                    enabled=True, base_seed=self.sample_seed + step,
                    sun_target=self.sun_k, show_progress=False,
                )
                metrics.update(result.thermo_metrics)
                metrics.update(result.summary_metrics)
            self.relaxed_records = logger.records()
            self.attempted_indices = dict(logger.attempted)
        metrics.update(joint_metrics(self.relaxed_records, self.attempted_indices))
        metrics.update(self.symmetry_panel.evaluate(raw, self.relaxed_records, requested_G=requested_G))
        return metrics

    def relaxation_diagnostics(self) -> list[dict]:
        """Observe the last evaluation without changing any headline metric.

        Geometry diagnostics are computed only when requested by post-hoc
        evaluation, and require no extra MLIP calls.
        """
        from orbitol.eval_diagnostics import relaxation_records
        return relaxation_records(
            self.relaxed_records, self.attempted_indices, self.raw_validity, self.thermo_logger,
        )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASET_NAMES, default="mp20")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--ppd-path", type=Path)
    parser.add_argument("--symmetry-reference", type=Path)
    parser.add_argument("--thermo", action="store_true")
    parser.add_argument("--step", type=int, default=20000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sample-count", type=int, default=10000)
    parser.add_argument("--sun-k", type=int, default=0, help="positive: legacy UN subset estimator")
    parser.add_argument("--nequip-backend", choices=["oeq", "torchscript"], default="oeq")
    parser.add_argument("--nequip-oeq-path", default="")
    parser.add_argument("--relax-steps", type=int, default=500)
    parser.add_argument("--nequip-model-kind", choices=["aoti", "torchscript"], default="torchscript")
    parser.add_argument("--nequip-ts-path", default="")
    parser.add_argument("--relax-mode", choices=["batch", "sequential"], default="batch")
    args = parser.parse_args(argv)
    paths = resolve_dataset_paths(args.dataset, overrides=dict(
        data_root=args.data_root, phase_diagram=args.ppd_path,
        symmetry_reference=args.symmetry_reference,
    ))
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = ThermoSettings(
        nequip_backend=args.nequip_backend, nequip_oeq_path=args.nequip_oeq_path,
        thermo_relax_steps=args.relax_steps,
        nequip_model_kind=args.nequip_model_kind, nequip_ts_path=args.nequip_ts_path,
        nequip_relax_mode=args.relax_mode, thermo_stability_device=args.device,
        thermo_ppd_mp=paths.phase_diagram,
    ) if args.thermo else None
    evaluator = DNGEvaluator(
        paths.data_root, device=args.device, sample_count=args.sample_count, sun_k=args.sun_k,
        thermo=settings, symmetry_cache=paths.symmetry_reference,
    )
    samples = torch.load(args.samples, map_location="cpu", weights_only=False)
    metrics = evaluator.evaluate(samples, step=args.step, requested_G=samples.get("requested_G"))
    # JSON null is the portable representation of undefined statistics.
    output = args.out / "metrics.json"
    output.write_text(json.dumps({k: v if np.isfinite(v) else None for k, v in metrics.items()}, indent=2, allow_nan=False) + "\n")
    (args.out / "evaluation_settings.json").write_text(json.dumps({
        "data_root": str(evaluator.data_root), "core_root": core.CORE_ROOT,
        "device": str(evaluator.device), "tag": evaluator.tag,
        "sample_count": evaluator.sample_count, "sun_k": evaluator.sun_k,
        "sample_seed": evaluator.sample_seed, "step": args.step,
        "no_thermo_before_steps": evaluator.no_thermo_before_steps,
        "thermo": asdict(settings) if settings is not None else None,
        "protocol": "legacy_raw_un_subset" if evaluator.sun_k > 0 else "relaxed_endpoints_v1",
        "joint_subset": "raw_UN" if evaluator.sun_k > 0 else "all_attempts",
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
