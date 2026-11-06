"""Shared evaluation of raw samples and their relaxed endpoints.

``relax_all`` produces one portable record per input index. ``score`` uses only
those records, loaded references and a phase diagram; it never loads an MLIP,
relaxes a structure or writes an artifact.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import multiprocessing
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import orbitol.core  # noqa: F401 -- bind the pin before src
from orbitol.eval_diagnostics import _finite, relaxation_records, variant_metrics
from orbitol.relax_geometry import relaxation_geometry
from orbitol.symmetry_panel import SYMPRECS, detect_many
from src.data.mp20_tokens import tokens_to_structure
from src.eval.dng_eval import compute_evaluator_metrics, compute_novelty_metrics
from src.eval.stability import _aggregate_thermo_records
from src.eval.uniqueness_novelty import (
    _filter_by_nary, _is_finite_structure, compute_uniqueness_novelty,
)
from src.utils.sample_stats import (
    compute_e_above_hull_mp2020_like, compute_e_above_hull_uncorrected,
)
from src.utils.stability_logger import StabilityLogger


CONVERGED_FMAX = 0.05  # eV/Å, Frechet filter forces (atoms and cell DOFs).


def run_in_child(target, *args, **kwargs):
    """Return target(*args, **kwargs) computed in a forked child process.

    Scoring calls C extensions (spglib, HiGHS, pymatgen) that can corrupt the heap on a
    pathological generated pool and abort the whole process, which no Python exception
    handler can catch. In a child, such a crash raises RuntimeError here instead. Fork keeps the loaded references without
    reloading them; the child must not touch CUDA, which scoring never does.
    """
    context = multiprocessing.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)

    def child():
        try:
            sender.send(("ok", target(*args, **kwargs)))
        except BaseException:
            sender.send(("error", traceback.format_exc()))

    process = context.Process(target=child)
    process.start()
    sender.close()
    try:
        status, payload = receiver.recv()
    except EOFError:
        process.join()
        raise RuntimeError(f"scoring process died with exit code {process.exitcode}") from None
    process.join()
    if status == "error":
        raise RuntimeError(f"scoring failed in the child process:\n{payload}")
    return payload


@dataclass(frozen=True)
class RelaxedRecord:
    sample_index: int
    final_structure: Any
    e_above_hull: float | None
    stable: bool
    metastable: bool
    is_un: bool
    failure_reason: str | None = None


def _with_core_fallback(items):
    """Use the core's invalid stand-in for Orbitol rows with no emitted atoms."""
    result = []
    for item in items:
        if bool(item["pad_mask"].all()):
            item = {key: torch.zeros_like(value) for key, value in item.items()}
            item["pad_mask"] = torch.ones_like(item["pad_mask"])
            item["pad_mask"][0] = False
        result.append(item)
    return result


def _construct(items):
    structures, reasons = [], []
    for item in _with_core_fallback(items):
        try:
            structures.append(tokens_to_structure(item))
            reasons.append(None)
        except Exception as exc:
            structures.append(None)
            reasons.append(f"{type(exc).__name__}: {exc}")
    return structures, reasons


def relax_all(items, relaxer, steps) -> list[dict]:
    """Relax every constructed token row with explicit fixed steps.

    Records contain ``index, constructed, relax_status, reason, final_structure,
    energy_eV, nsteps`` plus optional initial energy and force telemetry.
    Structures are pymatgen objects; the artifact writer
    serializes them. Status is ``ok``, ``failed`` or ``not_constructed``.
    RobustRelaxer retains healthy results during the subset retries below.
    """
    from orbitol.nequip_relax import RobustRelaxer

    # Use the same construction guards as the core evaluator (including NaNs
    # and unrealistically small cells), without filtering on raw validity.
    crystals = compute_evaluator_metrics(
        _with_core_fallback(items), enabled=True, limit=len(items), ref_structs=None,
        sample_seed=0, include_wasserstein=False,
    ).pred_crys_list
    structures = [c.structure if c.constructed else None for c in crystals]
    reasons = [None if c.constructed else c.invalid_reason for c in crystals]
    records = [dict(index=i, constructed=s is not None,
                    relax_status="failed" if s is not None else "not_constructed",
                    reason=reasons[i], final_structure=None, energy_eV=None, nsteps=None,
                    initial_energy_eV=None, initial_fmax=None, final_fmax=None, final_max_force=None)
               for i, s in enumerate(structures)]
    backend = relaxer if isinstance(relaxer, RobustRelaxer) else RobustRelaxer(relaxer)

    def run(indices):
        if not indices:
            return
        try:
            results = backend.relax_many([structures[i] for i in indices], steps=steps, verbose=False)
        except Exception as exc:
            if len(indices) == 1:
                records[indices[0]]["reason"] = str(exc)
            else:
                middle = len(indices) // 2
                run(indices[:middle])
                run(indices[middle:])
            return
        for i, result in zip(indices, results, strict=True):
            try:
                final = result["final_structure"]
                if final is None:
                    raise ValueError("missing final_structure")
                energy = float(result["trajectory"].energies[-1])
                nsteps = result.get("nsteps")
                nsteps = int(nsteps) if nsteps is not None else None
            except Exception as exc:
                records[i]["reason"] = f"{type(exc).__name__}: {exc}"
                continue
            records[i].update(relax_status="ok", reason=None, final_structure=final,
                              energy_eV=energy, nsteps=nsteps,
                              initial_energy_eV=_finite(result.get("initial_energy")),
                              **{key: _finite(result.get(key)) for key in
                                 ("initial_fmax", "final_fmax", "final_max_force")})

    run([i for i, s in enumerate(structures) if s is not None])
    return records


def _requested_groups(requested_G, count):
    if requested_G is None:
        return None
    groups = torch.as_tensor(requested_G).detach().cpu().numpy()
    if (groups.shape != (count,) or not np.isfinite(groups).all()
            or np.any(groups != groups.astype(int)) or np.any((groups < 1) | (groups > 230))):
        raise ValueError("requested_G must contain one integer space group (1..230) per sample")
    return groups.astype(int)


def _eligible(structures):
    # The core returns flag lists after finite/n-ary filtering, not input-indexed
    # arrays. Preserve that exact filtering when mapping back to attempt indices.
    finite = [(i, s) for i, s in enumerate(structures) if s is not None and _is_finite_structure(s)]
    return [finite[i] for i, _, _ in _filter_by_nary([s for _, s in finite], minimum_nary=1)]


def _raw_metrics(items, *, references, settings, tag):
    count = len(items)
    evaluator = compute_evaluator_metrics(
        items, enabled=True, limit=count, ref_structs=references["ref_structs"],
        sample_seed=settings["sample_seed"], include_wasserstein=True,
        wasserstein_max_samples=max(count, 1),
    )
    metrics = {}
    if evaluator.eval_count:
        for key in ("valid_rate", "comp_valid_rate", "struct_valid_rate"):
            metrics[f"{tag}/{key}"] = float(getattr(evaluator, key) or 0.0)
        for key in ("wdist_nary", "wdist_density_mass"):
            if f"sample_dist/{key}" in evaluator.dist_metrics:
                metrics[f"{tag}/{key}"] = evaluator.dist_metrics[f"sample_dist/{key}"]
    novelty = compute_novelty_metrics(items, references["novelty_ref_structs"], limit=count, minimum_nary=1)
    if novelty.novelty_metrics:
        for key in ("unique_rate", "novel_rate", "un_rate"):
            metrics[f"{tag}/{key}"] = float(getattr(novelty, key) or 0.0)
    raw, _ = _construct(items)
    rows = [dict(index=i, raw_struct_valid=False, raw_comp_valid=False, raw_valid=False,
                 raw_is_unique=False, raw_is_novel=False, raw_is_un=False)
            for i in range(count)]
    for crystal in evaluator.pred_crys_list:
        rows[crystal.sample_idx].update(raw_struct_valid=bool(crystal.struct_valid),
                                       raw_comp_valid=bool(crystal.comp_valid), raw_valid=bool(crystal.valid))
    if novelty.novelty_metrics:
        for (index, _), unique, novel, un in zip(
                _eligible(raw), *(novelty.novelty_metrics[key] for key in ("is_unique", "is_novel", "is_un")),
                strict=True):
            rows[index].update(raw_is_unique=bool(unique), raw_is_novel=bool(novel), raw_is_un=bool(un))
    return metrics, rows, raw, novelty, evaluator


def joint_metrics(records, attempted, *, total_attempts=None):
    """Joint endpoint U/N and stability; failed attempts keep their denominator."""
    n = len(attempted) if total_attempts is None else total_attempts
    sun = sum(r.is_un and r.stable for r in records)
    msun = sum(r.is_un and r.metastable for r in records)
    return {
        "joint/SUN": sun / n if n else float("nan"),
        "joint/MSUN": msun / n if n else float("nan"),
        "joint/SUN_count": float(sun), "joint/MSUN_count": float(msun),
        "joint/checked_count": float(n), "joint/relaxed_count": float(len(records)),
        "joint/UN_count": float(sum(attempted.values())),
    }


def stability_metrics(records, *, attempts, constructed, tag="sample_ema"):
    """Relaxed-ok includes endpoints whose hull calculation failed."""
    hulls = [r.e_above_hull for r in records if r.failure_reason is None
             and r.e_above_hull is not None and np.isfinite(r.e_above_hull)]
    relaxed_ok = len(records)
    values = dict(attempt_count=attempts, constructed_count=constructed,
                  construction_failure_count=attempts - constructed, relaxed_ok_count=relaxed_ok,
                  relax_failure_count=constructed - relaxed_ok, hull_ok_count=len(hulls),
                  hull_failure_count=relaxed_ok - len(hulls))
    for flag in ("stable", "metastable"):
        count = sum(getattr(r, flag) for r in records)
        values[f"{flag}_rate"] = count / attempts if attempts else float("nan")
        values[f"{flag}_rate_relaxed_ok"] = count / relaxed_ok if relaxed_ok else float("nan")
    for name, percentile in (("median", 50), ("p10", 10), ("p90", 90)):
        values[f"e_above_hull_{name}"] = float(np.percentile(hulls, percentile)) if hulls else float("nan")
    return {f"{tag}/stability/{key}": float(value) for key, value in values.items()}


def _thermo_metrics(rows, *, tag, backend):
    """Retain the core's existing all_relaxed aggregate names without relaxing twice."""
    checked = [row for row in rows if row["constructed"]]
    if not checked:
        return {}
    hulls = [row["e_above_hull"] for row in checked if row["hull_failure_reason"] is None
             and row["relax_status"] == "ok"]
    failures = Counter()
    divergence = 0
    for row in checked:
        if row["relax_status"] != "ok":
            reason, diverged = StabilityLogger._thermo_exception_reason(RuntimeError(row["reason"]))
            divergence += diverged
            failures[reason] += 1
        elif row["hull_failure_reason"] is not None:
            failures[row["hull_failure_reason"]] += 1
    counts = dict(thermo_checked=len(checked), thermo_success=len(hulls),
                  thermo_failed=len(checked) - len(hulls), thermo_divergence=divergence,
                  thermo_stable_count=sum(row["stable"] for row in checked),
                  thermo_metastable_count=sum(row["metastable"] for row in checked))
    if hulls:
        counts["thermo_e_above_hull_mean"] = float(np.mean(hulls))
    counts.update({f"thermo_fail/{reason}": count for reason, count in failures.items()})
    metrics = _aggregate_thermo_records([{f"{tag}/{key}": value for key, value in counts.items()}], tag)
    for suffix, source in dict(checked_total="checked_total", success_total="success_total",
                               failed_total="failed_total", divergence_total="divergence_total",
                               success_rate="success_rate", failure_rate="failure_rate",
                               stability="stable_rate", metastability="metastable_rate",
                               e_above_hull_mean="e_above_hull_mean").items():
        if f"{tag}/thermo_{source}" in metrics:
            metrics[f"{tag}/{backend}_stability/{suffix}"] = metrics[f"{tag}/thermo_{source}"]
    return metrics


def _relax_metrics(rows, tag):
    """Finite-only aggregates; old backends emit no new metric keys.

    Initial telemetry means a finite initial energy on a relaxed-ok row. Empty
    measured subsets omit their keys. One-shot yields retain every attempt in
    the denominator; geometry/force statistics use relaxed-ok rows, convergence
    requires final_fmax, and symmetry splits/rates require both detected groups.
    """
    relaxed = [r for r in rows if r.get("relax_status") == "ok"]
    count = sum(r["initial_energy_eV"] is not None for r in relaxed)
    if not count:
        return {}
    metrics = {f"{tag}/relax/telemetry_count": count}
    for key in ("SUN_oneshot", "MSUN_oneshot"):
        metrics[f"{tag}/{key}"] = sum(r[key] for r in rows) / len(rows)

    def summarize(pool, key, suffixes, prefix):
        values = [r[key] for r in pool if r.get(key) is not None and np.isfinite(r[key])]
        if values:
            for name, percentile in suffixes:
                metrics[f"{prefix}{name}"] = float(np.mean(values) if percentile is None
                                                   else np.percentile(values, percentile))

    summarize(relaxed, "e_above_hull_oneshot", (("median", 50), ("p10", 10), ("p90", 90)),
              f"{tag}/stability/e_above_hull_oneshot_")
    geometry = ("rms_disp", "max_disp", "strain", "volume_change_pct", "energy_drop_meV")
    for key in (*geometry, "initial_fmax", "final_fmax"):
        summarize(relaxed, key, (("median", 50), ("mean", None), ("p90", 90)), f"{tag}/relax/{key}_")
    for key in geometry:
        summarize([r for r in relaxed if r["MSUN"]], key, (("median_msun", 50),), f"{tag}/relax/{key}_")
    detected = [r for r in relaxed if r["raw_space_group_0.1"] > 0 and r["relaxed_space_group_0.1"] > 0]
    for kept, suffix in ((True, "sg_kept"), (False, "sg_changed")):
        for key in ("rms_disp", "strain", "energy_drop_meV"):
            summarize([r for r in detected if r["sg_kept"] == kept], key,
                      ((f"median_{suffix}", 50),), f"{tag}/relax/{key}_")
    for name, pool, flag in (
            ("converged_rate", [r for r in relaxed if r["final_fmax"] is not None], "converged"),
            ("sg_kept_rate", detected, "sg_kept"),
            ("sg_kept_rate_msun", [r for r in detected if r["MSUN"]], "sg_kept")):
        if pool:
            metrics[f"{tag}/relax/{name}"] = sum(r[flag] for r in pool) / len(pool)
    return metrics


def score(items, relaxed_records, *, references, ppd, settings, tag, step, requested_G=None) -> tuple[dict, list[dict]]:
    """Score a saved pool with the same inputs as ``DNGEvaluator``.

    ``references`` maps ``ref_structs``, ``novelty_ref_structs`` and a
    ``symmetry_panel`` whose reference is already loaded. ``ppd`` is the loaded
    phase diagram. ``settings`` is the evaluator's settings dict: sample_count,
    sample_seed, no_thermo_before_steps, sun_k and thermo (ThermoSettings asdict,
    or None). ``relaxed_records=None`` disables thermodynamics. Positive sun_k
    uses the adapter's explicit historical subset path instead.

    One-shot hulls use the evaluator's exact relaxation input and initial energy.
    Failed/unconstructed attempts have no initial energy and count as not stable
    in one-shot SUN/MSUN; they remain in the full-attempt denominator.
    """
    prepared = score_raw(items, references=references, settings=settings, tag=tag, requested_G=requested_G)
    return complete_score(prepared, relaxed_records, references=references, ppd=ppd,
                          settings=settings, tag=tag, step=step)


def score_raw(items, *, references, settings, tag, requested_G=None):
    """Prepare scoring that depends only on the raw pool and CPU references."""
    if settings.get("sun_k", 0) > 0:
        raise ValueError("score requires all-attempt records; use DNGEvaluator for legacy sun_k > 0")
    if references["symmetry_panel"]._reference is None:
        raise ValueError("Load the symmetry reference before calling score")
    items = _with_core_fallback(items)[:settings["sample_count"]]
    requested_G = _requested_groups(requested_G, len(items))
    metrics, rows, raw, _, evaluator = _raw_metrics(items, references=references, settings=settings, tag=tag)
    raw_groups = detect_many([raw[row["index"]] for row in rows])
    return items, requested_G, metrics, rows, raw, evaluator, raw_groups


def complete_score(prepared, relaxed_records, *, references, ppd, settings, tag, step):
    """Complete a prepared raw score with the relaxed endpoints."""
    items, requested_G, metrics, rows, raw, evaluator, raw_groups = prepared
    thermo = settings["thermo"]
    enabled = (thermo is not None and thermo["thermo_stability_check"]
               and step >= settings.get("no_thermo_before_steps", 0) and relaxed_records is not None)
    records = []
    if enabled:
        by_index = {record["index"]: record for record in relaxed_records}
        if len(relaxed_records) != len(items) or set(by_index) != set(range(len(items))):
            raise ValueError("relaxed_records must contain exactly one record per attempt index")
        method = thermo["thermo_ehull_method"]
        if method not in {"mp2020_like", "uncorrected"}:
            raise ValueError(f"Unsupported hull method: {method}")
        compat = None
        if method == "mp2020_like":
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
            compat = MaterialsProject2020Compatibility(check_potcar=False)

        def hull_for(structure, energy):
            try:
                if method == "mp2020_like":
                    return compute_e_above_hull_mp2020_like(ppd, structure, energy, mp2020_compat=compat)
                return compute_e_above_hull_uncorrected(ppd, structure, energy)
            except Exception as exc:
                return None, f"exception:{type(exc).__name__}"

        inputs = {c.sample_idx: c.structure for c in evaluator.pred_crys_list if c.constructed}
        endpoints = []
        for row in rows:
            record = by_index[row["index"]]
            status = record["relax_status"]
            if status not in {"ok", "failed", "not_constructed"}:
                raise ValueError(f"Invalid relax_status: {status}")
            if bool(record["constructed"]) != (status != "not_constructed"):
                raise ValueError("constructed and relax_status disagree")
            if status == "ok" and record["final_structure"] is None:
                raise ValueError("Successful relaxation requires final_structure")
            row.update(constructed=bool(record["constructed"]), relax_status=status, reason=record["reason"],
                       energy_eV=record["energy_eV"], nsteps=record["nsteps"], e_above_hull=None,
                       hull_failure_reason=None, stable=False, metastable=False,
                       relaxed_is_unique=False, relaxed_is_novel=False, relaxed_is_un=False,
                       initial_energy_eV=None, initial_fmax=None, final_fmax=None, final_max_force=None,
                       rms_disp=None, max_disp=None, strain=None, volume_change_pct=None, energy_drop_meV=None,
                       converged=None, sg_kept=False, e_above_hull_oneshot=None,
                       stable_oneshot=False, metastable_oneshot=False)
            if status != "ok":
                continue
            final = record["final_structure"]
            endpoints.append((row["index"], final))
            hull, reason = hull_for(final, record["energy_eV"])
            ok = reason is None and hull is not None and np.isfinite(hull)
            row.update(e_above_hull=float(hull) if ok else None,
                       hull_failure_reason=None if ok else reason or "non_finite_hull",
                       stable=bool(ok and hull <= 0.0), metastable=bool(ok and hull <= 0.1))
            row.update({key: _finite(record.get(key)) for key in
                        ("initial_energy_eV", "initial_fmax", "final_fmax", "final_max_force")})
            if row["final_fmax"] is not None:
                row["converged"] = row["final_fmax"] <= CONVERGED_FMAX
            initial = inputs.get(row["index"])
            geometry = relaxation_geometry(initial, final, row["initial_energy_eV"], record["energy_eV"])
            if geometry is not None:
                row.update(geometry)
            if initial is not None and row["initial_energy_eV"] is not None:
                hull, reason = hull_for(initial, row["initial_energy_eV"])
                ok = reason is None and hull is not None and np.isfinite(hull)
                row.update(e_above_hull_oneshot=float(hull) if ok else None,
                           stable_oneshot=bool(ok and hull <= 0.0),
                           metastable_oneshot=bool(ok and hull <= 0.1))
        eligible = _eligible([final for _, final in endpoints])
        un = compute_uniqueness_novelty([s for _, s in eligible], references["novelty_ref_structs"], minimum_nary=1)
        for (position, _), unique, novel in zip(eligible, un["is_unique"], un["is_novel"], strict=True):
            # Unlike the raw novelty-then-dedup mask, endpoint uniqueness is
            # computed before novelty; the first original attempt index wins.
            rows[endpoints[position][0]].update(relaxed_is_unique=bool(unique), relaxed_is_novel=bool(novel),
                                                relaxed_is_un=bool(unique and novel))
        records = [RelaxedRecord(i, final, rows[i]["e_above_hull"], rows[i]["stable"], rows[i]["metastable"],
                                 rows[i]["relaxed_is_un"], rows[i]["hull_failure_reason"])
                   for i, final in endpoints]
        metrics.update(stability_metrics(records, attempts=len(items),
                                         constructed=sum(row["constructed"] for row in rows), tag=tag))
        metrics.update(_thermo_metrics(rows, tag=f"{tag}/all_relaxed", backend=thermo["thermo_mlip"]))
    relaxed_groups = detect_many([by_index[row["index"]]["final_structure"] if enabled else None for row in rows])
    for row, raw_row, relaxed_row in zip(rows, raw_groups, relaxed_groups, strict=True):
        for flag, metric in (("stable", "SUN"), ("metastable", "MSUN")):
            row[metric] = bool(row.get(flag) and row.get("relaxed_is_un"))
            row[f"{metric}_raw_UN"] = bool(row.get(flag) and row["raw_is_un"])
            row[f"{metric}_oneshot"] = bool(row.get(f"{flag}_oneshot") and row["raw_is_un"])
        if requested_G is not None:
            row["requested_G"] = int(requested_G[row["index"]])
        for j, symprec in enumerate(SYMPRECS):
            row[f"raw_space_group_{symprec:g}"] = int(raw_row[j])
            row[f"relaxed_space_group_{symprec:g}"] = int(relaxed_row[j])
        if row.get("relax_status") == "ok":
            row["sg_kept"] = row["raw_space_group_0.1"] == row["relaxed_space_group_0.1"] > 0
    metrics.update(joint_metrics(records, {r.sample_index: r.is_un for r in records},
                                 total_attempts=len(items) if enabled else 0))
    if enabled:
        metrics.update(_relax_metrics(rows, tag))
        for key in ("SUN", "MSUN", "SUN_raw_UN", "MSUN_raw_UN"):
            metrics[f"{tag}/{key}"] = sum(row[key] for row in rows) / len(items) if items else float("nan")
        # Use the same observational diagnostics as synchronous training. Only
        # saved endpoints and hull results are needed; no extra MLIP call occurs.
        attempted = {row["index"]: row["relaxed_is_un"] for row in rows if row["constructed"]}
        raw_validity = {row["index"]: {key: row[key] for key in
                        ("raw_struct_valid", "raw_comp_valid", "raw_valid")} for row in rows}
        logger = SimpleNamespace(
            endpoints={record.sample_index: (record.is_un, dict(
                nsteps=by_index[record.sample_index]["nsteps"],
                trajectory=SimpleNamespace(energies=[by_index[record.sample_index]["energy_eV"]]),
            )) for record in records},
            # Initial volumes and failure text do not enter variant_metrics.
            initial_volume_per_atom={}, failures={},
        )
        metrics.update(variant_metrics(
            relaxation_records(records, attempted, raw_validity, logger), checked_count=len(items)))
    metrics.update(references["symmetry_panel"].evaluate(
        raw, records, requested_G=requested_G,
        precomputed_groups=dict(raw=raw_groups, relaxed=relaxed_groups[[r.sample_index for r in records]]),
    ))
    return metrics, rows
