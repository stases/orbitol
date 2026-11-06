"""Observational variants; never alter pinned stability or SUN metrics."""
import itertools
import math

import numpy as np


RELAXATION_FIELDS = (
    "initial_energy_eV", "initial_fmax", "final_fmax", "rms_disp", "max_disp", "strain",
    "volume_change_pct", "energy_drop_meV", "converged", "sg_kept",
    "e_above_hull_oneshot", "stable_oneshot", "metastable_oneshot", "SUN_oneshot", "MSUN_oneshot",
    "SUN_raw_UN", "MSUN_raw_UN",
)


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def minimum_distance(structure):
    """Shortest PBC distance between distinct sites or between a site and its own periodic image.

    Pure numpy over the 27 nearest images of the LLL-reduced cell, with fractional differences
    wrapped first. pymatgen's neighbor-list C code can corrupt the heap on an exploded relaxed cell
    and abort the whole process, so it is not used here. Coincident distinct atoms
    give 0; only the same site in the same image is excluded.
    """
    reduced = structure.get_reduced_structure(reduction_algo="LLL")
    lattice, frac = reduced.lattice.matrix, reduced.frac_coords
    if not (np.isfinite(lattice).all() and np.isfinite(frac).all()):
        return None
    images = np.array(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
    delta = frac[None, :, :] - frac[:, None, :]
    delta -= np.round(delta)
    distances = np.linalg.norm((delta[:, :, None, :] + images) @ lattice, axis=-1)
    same = np.arange(len(frac))
    distances[same, same, 13] = np.inf  # images[13] is the zero shift
    return _finite(distances.min())


def _final_max_force(result):
    for key in ("final_max_force", "max_force"):
        if result.get(key) is not None:
            return _finite(result[key])
    forces = result.get("forces")
    if forces is None:
        history = getattr(result.get("trajectory"), "forces", None)
        if history is not None and len(history):
            forces = history[-1]
    if forces is None:
        return None
    if hasattr(forces, "detach"):
        forces = forces.detach().cpu().numpy()
    return _finite(np.linalg.norm(np.asarray(forces), axis=-1).max())


def relaxation_records(records, attempted, raw_validity, logger):
    """One JSON-safe row per attempted pool index, including failed leaves.

    Missing backend telemetry is null, not reconstructed by another model
    call. ``relaxation_steps`` is the backend's reported nsteps (the pinned
    sequential optimizer can report zero despite manually taking steps).
    """
    by_index = {record.sample_index: record for record in records}
    rows = []
    for index, is_un in sorted(attempted.items()):
        record = by_index.get(index)
        result = logger.endpoints.get(index, (None, {}))[1]
        failure, message = logger.failures.get(index, (None, None))
        if record is not None:
            failure = record.failure_reason
        outcome = "relaxation_failed"
        if record is not None:
            outcome = "success" if failure is None else "hull_failed"
        row = dict(
            pool_index=int(index), is_un=bool(is_un),
            raw_struct_valid=None, raw_comp_valid=None, raw_valid=None,
            relaxation_outcome=outcome,
            failure_reason=failure or ("missing_result" if record is None else None),
            failure_message=message,
            e_above_hull=_finite(record.e_above_hull) if record else None,
            energy_per_atom=None,
            initial_volume_per_atom=_finite(logger.initial_volume_per_atom.get(index)),
            relaxed_volume_per_atom=None, relaxed_min_distance=None,
            relaxation_steps=None, final_max_force=None,
            stable=bool(record.stable) if record else False,
            metastable=bool(record.metastable) if record else False,
            diagnostic_errors=[],
        )
        row.update(raw_validity.get(index, {}))
        row.update({key: result.get(key) for key in RELAXATION_FIELDS})
        final = record.final_structure if record is not None else None
        measurements = {"relaxation_steps": lambda: _finite(result.get("nsteps")),
                        "final_max_force": lambda: _final_max_force(result)}
        if final is not None:
            measurements.update(
                energy_per_atom=lambda: _finite(result["trajectory"].energies[-1] / len(final)),
                relaxed_volume_per_atom=lambda: _finite(final.volume / len(final)),
                relaxed_min_distance=lambda: minimum_distance(final),
            )
        for key, measure in measurements.items():
            try:
                row[key] = measure()
            except Exception as exc:
                # A diagnostic geometry/telemetry error must not discard the
                # evaluator's existing result. Unknown values stay null.
                row["diagnostic_errors"].append(f"{key}: {type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def variant_metrics(rows, *, checked_count):
    """Variants use the unchanged joint checked-count denominator.

    Criterion counts are over counted-stable OR counted-metastable records;
    they can overlap. Suspect counts include non-UN records, while every SUN
    numerator still requires is_un. Missing measurements do not imply suspect.
    """
    criteria = {"e_above_hull": ("e_above_hull", -0.5),
                "min_distance": ("relaxed_min_distance", 0.7),
                "volume_per_atom": ("relaxed_volume_per_atom", 3.0)}
    counts = dict(stable_suspect_count=0, metastable_suspect_count=0,
                  valid_checked_count=0, SUN_plausible=0, MSUN_plausible=0,
                  SUN_valid=0, MSUN_valid=0)
    counts.update({f"suspect_{name}_count": 0 for name in criteria})
    hulls = []
    for row in rows:
        eligible = row["stable"] or row["metastable"]
        suspect = False
        for name, (key, threshold) in criteria.items():
            hit = bool(eligible and row.get(key) is not None and row[key] < threshold)
            counts[f"suspect_{name}_count"] += hit
            suspect |= hit
        valid = row.get("raw_valid") is True
        counts["valid_checked_count"] += valid
        for flag, metric in (("stable", "SUN"), ("metastable", "MSUN")):
            counts[f"{flag}_suspect_count"] += bool(row[flag] and suspect)
            counts[f"{metric}_plausible"] += bool(row["is_un"] and row[flag] and not suspect)
            counts[f"{metric}_valid"] += bool(row["is_un"] and row[flag] and valid)
        hull = row.get("e_above_hull")
        if row["relaxation_outcome"] == "success" and hull is not None and math.isfinite(hull):
            hulls.append(hull)
    for key in ("SUN_plausible", "MSUN_plausible", "SUN_valid", "MSUN_valid"):
        counts[key] = counts[key] / checked_count if checked_count else float("nan")
    counts["e_above_hull_median"] = float(np.median(hulls)) if hulls else float("nan")
    counts["e_above_hull_p05"] = float(np.percentile(hulls, 5)) if hulls else float("nan")
    return {f"diag/{key}": value for key, value in counts.items()}
