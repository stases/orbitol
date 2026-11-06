"""Names at the logging boundary, without changing metric producers.

Logged data keys have exactly one ``/`` (``step`` and ``train_step`` are axes).
W&B and metrics.jsonl use these names; evaluation artifacts keep internal names.
Use ``canonical()`` when reading old runs. Unknown keys are flattened under
``other/`` with a warning once per key. Collisions keep the first value and warn.
"""
from __future__ import annotations

import logging
import re

LOGGER = logging.getLogger(__name__)

# Fixed producer names. None means a verified duplicate in the all-attempt protocol.
FIXED_NAMES = {
    "step": "step", "train_step": "train_step",
    "loss/total": "train/loss_total", "loss/type": "train/loss_type",
    "loss/coord": "train/loss_coord", "loss/lattice": "train/loss_lattice",
    "lr": "train/lr", "sigma/mean": "train/sigma_mean", "sigma/std": "train/sigma_std",
    "stats/type_acc": "train/type_acc", "stats/coord_l2": "train/coord_l2",
    "stats/lattice_l2": "train/lattice_l2", "perf/grad_scale": "train/grad_scale",
    "val/loss_total": "val/loss_total", "val/loss_type": "val/loss_type",
    "val/loss_coord": "val/loss_coord", "val/loss_lattice": "val/loss_lattice",
    "sample_ema/SUN": "eval/SUN", "sample_ema/MSUN": "eval/MSUN",
    "sample_ema/SUN_raw_UN": "eval/SUN_raw_UN", "sample_ema/MSUN_raw_UN": "eval/MSUN_raw_UN",
    "sample_ema/SUN_raw": "eval/SUN_raw_UN", "sample_ema/MSUN_raw": "eval/MSUN_raw_UN",
    "sample_ema/SUN_oneshot": "eval/SUN_oneshot", "sample_ema/MSUN_oneshot": "eval/MSUN_oneshot",
    "diag/SUN_valid": "eval/SUN_valid", "diag/MSUN_valid": "eval/MSUN_valid",
    "diag/SUN_plausible": "eval/SUN_plausible", "diag/MSUN_plausible": "eval/MSUN_plausible",
    "orbit/best_msun": "eval/best_MSUN",
    "sample_ema/stability/stable_rate": "eval/stable_rate",
    "sample_ema/stability/metastable_rate": "eval/metastable_rate",
    "sample_ema/valid_rate": "eval/valid_rate", "sample_ema/comp_valid_rate": "eval/comp_valid_rate",
    "sample_ema/struct_valid_rate": "eval/struct_valid_rate", "sample_ema/unique_rate": "eval/unique_rate",
    "sample_ema/novel_rate": "eval/novel_rate", "sample_ema/un_rate": "eval/un_rate",
    "sample_ema/wdist_nary": "eval/wdist_nary", "sample_ema/wdist_density_mass": "eval/wdist_density_mass",
    "sample_ema/stability/e_above_hull_median": "eval_hull/e_above_hull_median",
    "sample_ema/stability/e_above_hull_p10": "eval_hull/e_above_hull_p10",
    "sample_ema/stability/e_above_hull_p90": "eval_hull/e_above_hull_p90",
    "diag/e_above_hull_p05": "eval_hull/e_above_hull_p05",
    "sample_ema/all_relaxed/thermo_e_above_hull_mean": "eval_hull/e_above_hull_mean",
    "sample_ema/stability/stable_rate_relaxed_ok": "eval_hull/stable_rate_relaxed_ok",
    "sample_ema/stability/metastable_rate_relaxed_ok": "eval_hull/metastable_rate_relaxed_ok",
    "sample_ema/stability/e_above_hull_oneshot_median": "eval_hull/e_above_hull_oneshot_median",
    "sample_ema/stability/e_above_hull_oneshot_p10": "eval_hull/e_above_hull_oneshot_p10",
    "sample_ema/stability/e_above_hull_oneshot_p90": "eval_hull/e_above_hull_oneshot_p90",
    "sample_ema/stability/attempt_count": "eval_count/attempts",
    "sample_ema/stability/constructed_count": "eval_count/constructed",
    "sample_ema/stability/construction_failure_count": "eval_count/construction_failures",
    "sample_ema/stability/relaxed_ok_count": "eval_count/relaxed_ok",
    "sample_ema/stability/relax_failure_count": "eval_count/relax_failures",
    "sample_ema/stability/hull_ok_count": "eval_count/hull_ok",
    "sample_ema/stability/hull_failure_count": "eval_count/hull_failures",
    "joint/SUN_count": "eval_count/SUN", "joint/MSUN_count": "eval_count/MSUN",
    "joint/UN_count": "eval_count/relaxed_UN",
    "diag/valid_checked_count": "eval_count/raw_valid",
    "diag/stable_suspect_count": "eval_count/suspect_stable",
    "diag/metastable_suspect_count": "eval_count/suspect_metastable",
    "diag/suspect_e_above_hull_count": "eval_count/suspect_low_hull",
    "diag/suspect_min_distance_count": "eval_count/suspect_min_distance",
    "diag/suspect_volume_per_atom_count": "eval_count/suspect_volume",
    # _thermo_exception_reason marks BOTH relax_divergence and relax_non_finite.
    "sample_ema/all_relaxed/thermo_divergence_total": "eval_count/relax_divergence_total",
    "orbit/contains_a_rate": "decode/contains_a_rate",
    "orbit/eval_seconds": "perf/eval_seconds", "orbit/nequip_seconds": "perf/nequip_eval_seconds",
    "posthoc/sample_seconds": "perf/posthoc_sample_seconds",
    "posthoc/evaluate_seconds": "perf/posthoc_evaluate_seconds",
    "orbit/nequip_error": "status/eval_error",
    "orbit/eval_snapshot_step": "status/eval_snapshot_step",
    "orbit/eval_job_id": "status/eval_job_id",
    "orbit/eval_submit_failed": "status/eval_submit_failed",
    "orbit/eval_skipped_backlog": "status/eval_skipped_backlog",
    "posthoc/pool_sha256": "status/pool_sha256",
    "joint/SUN": None, "joint/MSUN": None,
    "joint/checked_count": None, "joint/relaxed_count": None,
    "diag/e_above_hull_median": None,
    "sample_ema/all_relaxed/thermo_checked_total": None,
    "sample_ema/all_relaxed/thermo_success_total": None,
    "sample_ema/all_relaxed/thermo_failed_total": None,
    "sample_ema/all_relaxed/thermo_success_rate": None,
    "sample_ema/all_relaxed/thermo_failure_rate": None,
    "sample_ema/all_relaxed/thermo_stable_rate": None,
    "sample_ema/all_relaxed/thermo_metastable_rate": None,
}
LEGACY_JOINT = {
    "joint/SUN": "eval_legacy/joint_SUN", "joint/MSUN": "eval_legacy/joint_MSUN",
    "joint/checked_count": "eval_legacy/joint_checked_count",
    "joint/relaxed_count": "eval_legacy/joint_relaxed_count",
}
# These copies are assigned directly in eval_protocol._thermo_metrics and the core.
THERMO_COPIES = {
    "checked_total": "checked_total", "success_total": "success_total",
    "failed_total": "failed_total", "divergence_total": "divergence_total",
    "success_rate": "success_rate", "failure_rate": "failure_rate",
    "stability": "stable_rate", "metastability": "metastable_rate",
    "e_above_hull_mean": "e_above_hull_mean",
}
RULES = tuple((re.compile(pattern), target) for pattern, target in (
    (r"val/sigma_(?P<s>[^/]+)/coord_ratio/(?P<cls>[^/]+)", "val_coord_class/{cls}_s{s}"),
    (r"val/sigma_(?P<s>[^/]+)/(?P<x>[^/]+)", "val_sigma/{x}_s{s}"),
    (r"sample_ema/all_relaxed/thermo_fail/(?P<reason>.+)", "eval_count/relax_fail_{reason}"),
    (r"sample_ema/all_relaxed/nequip_stability/(?:" + "|".join(THERMO_COPIES) + r")", None),
    (r"sample_ema/relax/(?P<x>.+)", "eval_relax/{x}"),
    (r"sym/(?P<label>raw|relaxed)/(?P<tol>[^/]+)/crystal_system/(?P<cs>[^/]+)", "sym_{label}/cs_{cs}_{tol}"),
    (r"sym/relaxed/(?P<tol>[^/]+)/crystal_system_retention", "sym_relaxed/cs_retention_{tol}"),
    (r"sym/(?P<label>raw|relaxed)/(?P<tol>[^/]+)/(?P<x>p1_share|space_group_jsd|effective_groups|failure_rate|crystal_system_tv|retention)", "sym_{label}/{x}_{tol}"),
    (r"sym/(?:raw|relaxed)/[^/]+/(?:count|detected_count|failure_count|retention_count|retention_total)", None),
    (r"orbit/failures/(?P<reason>.+)", "decode/fail_{reason}"),
    (r"perf/(?P<x>[^/]+)", "perf/{x}"),
    # The historical raw-UN subset has additional core thermo aggregates, with
    # different denominators. Preserve them, including the backend's copies.
    (r"sample_ema/sun/(?P<x>.+)", "eval_legacy/sun_{x}"),
))
LOGGED_SECTIONS = frozenset((
    "train", "val", "val_sigma", "val_coord_class", "eval", "eval_hull", "eval_count",
    "eval_relax", "eval_legacy", "sym_raw", "sym_relaxed", "decode", "perf", "status", "other",
))
_warned_unknown = set()
_warned_collisions = set()


def _name(key, metrics, legacy_subset):
    if legacy_subset and key in LEGACY_JOINT:
        return LEGACY_JOINT[key]
    # Without thermo (or with a legacy subset), these twins do not exist.
    if key == "diag/e_above_hull_median" and "sample_ema/stability/e_above_hull_median" not in metrics:
        return "eval_hull/e_above_hull_median"
    match = re.fullmatch(r"sym/(raw|relaxed)/([^/]+)/count", key)
    if match:
        label, tol = match.groups()
        twin = "attempt_count" if label == "raw" else "relaxed_ok_count"
        if f"sample_ema/stability/{twin}" not in metrics:
            return f"sym_{label}/count_{tol}"
    if key in FIXED_NAMES:
        return FIXED_NAMES[key]
    for pattern, target in RULES:
        match = pattern.fullmatch(key)
        if match:
            return None if target is None else target.format(
                **{k: v.replace("/", "__") for k, v in match.groupdict().items()})
    if key.count("/") == 1 and key.split("/", 1)[0] in LOGGED_SECTIONS:
        return key
    if key not in _warned_unknown:
        _warned_unknown.add(key)
        LOGGER.warning("Unmapped metric %r; logging under other/", key)
    return "other/" + key.replace("/", "__")


def log_view(metrics: dict, *, legacy_subset: bool = False) -> dict:
    """Return the logging view without mutating producers or filtering zero values."""
    result, sources = {}, {}
    for key, value in metrics.items():
        name = _name(key, metrics, legacy_subset)
        if name is None:
            continue
        if name in result:
            pair = (sources[name], key, name)
            if pair not in _warned_collisions:
                _warned_collisions.add(pair)
                LOGGER.warning("Metric collision: %r and %r map to %r; keeping first", *pair)
            continue
        sources[name], result[name] = key, value
    return result


def canonical(row: dict) -> dict:
    """Read either schema; infer historical subsets when full-pool counts are absent.

    Rows carrying joint metrics but no attempt count are conservatively treated
    as legacy. A history row alone cannot recover a missing run configuration.
    """
    legacy = (any(k.startswith("sample_ema/sun/") or k in LEGACY_JOINT for k in row)
              and not any(k in row for k in ("sample_ema/stability/attempt_count", "eval_count/attempts")))
    return log_view(row, legacy_subset=legacy)


# Expected parameter families for the saved view. Producers remain independent;
# coverage tests check their real outputs against the mapping above.
SIGMAS = ("0.02", "0.1", "0.5", "2", "10", "0.2", "0.3")
COORD_CLASSES = ("d1", "d2", "d3", "wrap", "nondiag")
SYMMETRY_TOLERANCES = ("0.01", "0.1")
CRYSTAL_SYSTEMS = ("triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic")
SYMMETRY_METRICS = ("p1_share", "space_group_jsd", "effective_groups", "failure_rate", "crystal_system_tv")
# Exactly the keys eval_protocol._relax_metrics emits (tests compare both ways).
RELAX_QUANTITIES = ("rms_disp", "max_disp", "strain", "volume_change_pct", "energy_drop_meV", "initial_fmax", "final_fmax")
RELAX_STATS = ("median", "mean", "p90")
RELAX_SUBSETS = {"msun": RELAX_QUANTITIES[:5], "sg_kept": ("rms_disp", "strain", "energy_drop_meV"),
                 "sg_changed": ("rms_disp", "strain", "energy_drop_meV")}
RELAX_RATES = ("converged_rate", "sg_kept_rate", "sg_kept_rate_msun", "telemetry_count")


def relax_internal_keys():
    keys = [f"sample_ema/relax/{q}_{stat}" for q in RELAX_QUANTITIES for stat in RELAX_STATS]
    keys += [f"sample_ema/relax/{q}_median_{subset}" for subset, qs in RELAX_SUBSETS.items() for q in qs]
    return keys + [f"sample_ema/relax/{x}" for x in RELAX_RATES]


def expected_internal_keys():
    """Finite catalogue for panels; open-ended reasons still use regex rules."""
    keys = list(FIXED_NAMES)
    keys += [f"val/sigma_{s}/{x}" for s in SIGMAS for x in
             ("loss_type", "loss_coord", "loss_lattice", "coord_do_nothing", "coord_ratio")]
    keys += [f"val/sigma_{s}/coord_ratio/{c}" for s in SIGMAS for c in COORD_CLASSES]
    keys += relax_internal_keys()
    keys += [f"sym/{label}/{tol}/{x}" for label in ("raw", "relaxed") for tol in SYMMETRY_TOLERANCES
             for x in (*SYMMETRY_METRICS, *(f"crystal_system/{cs}" for cs in CRYSTAL_SYSTEMS))]
    keys += [f"sym/relaxed/{tol}/{x}" for tol in SYMMETRY_TOLERANCES
             for x in ("retention", "crystal_system_retention")]
    keys += [f"perf/{x}" for x in ("steps_per_second", "samples_per_second", "gpu_memory_allocated_gb",
                                  "gpu_memory_reserved_gb", "lll_cap_exceedances")]
    return keys


HEADLINE_KEYS = (
    "sample_ema/MSUN", "sample_ema/SUN", "sample_ema/MSUN_raw_UN", "sample_ema/MSUN_oneshot",
    "orbit/best_msun", "diag/MSUN_valid", "diag/MSUN_plausible", "sample_ema/valid_rate",
    "sample_ema/un_rate", "sample_ema/stability/metastable_rate",
)
# The producer includes every reason, even at zero; test against DECODE_FAILURES.
DECODE_REASONS = (
    "invalid_group", "invalid_shape", "empty_structure", "non_finite", "invalid_rows", "reused_fixed_row",
    "invalid_species", "invalid_lattice", "coinciding_orbits", "collapsed_orbit", "containment_failed",
    "failed_decode", "not_emitted", "too_many_atoms", "decode_error",
)
