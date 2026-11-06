"""Build a curated saved W&B view; --dry-run is entirely offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

# Also support direct invocation without an editable install or PYTHONPATH.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from orbitol import contracts as C  # noqa: E402
from orbitol.metric_names import (  # noqa: E402
    DECODE_REASONS, FIXED_NAMES, HEADLINE_KEYS, RELAX_QUANTITIES, RELAX_STATS, RELAX_SUBSETS,
    SYMMETRY_TOLERANCES, expected_internal_keys, log_view,
)


def build_sections():
    """Serializable panel specification shared by dry runs and workspace creation."""
    # Alias spellings belong to different producer versions, never the same row.
    keys = list(dict.fromkeys(name for key in expected_internal_keys()
                             for name in log_view({key: None}) if name not in {"step", "train_step"}))

    def panel(title, y, **kwargs):
        return dict(title=title, y=y, **kwargs)

    def individual(section):
        return [panel(key.split("/", 1)[1], [key]) for key in keys if key.startswith(section + "/")]

    headlines = []
    for internal in HEADLINE_KEYS:
        logged = FIXED_NAMES[internal]
        aliases = [key for key, value in FIXED_NAMES.items() if value == logged and key != logged]
        # Prefer the spelling actually used before the rename, then newer aliases.
        aliases.sort(key=lambda key: key.endswith("_raw_UN"))
        headlines.append(panel(logged.split("/", 1)[1], [logged, *aliases]))

    def relax(*names):
        return [f"eval_relax/{name}" for name in names]

    relaxation = [panel(f"{q} median", relax(f"{q}_median")) for q in RELAX_QUANTITIES]
    relaxation += [panel(x, relax(x)) for x in ("converged_rate", "sg_kept_rate")]
    relaxation.append(panel("MSUN vs one-shot MSUN", ["eval/MSUN", "eval/MSUN_oneshot"]))
    detail = [panel(f"{q}: median / mean / p90", relax(*(f"{q}_{stat}" for stat in RELAX_STATS)))
              for q in RELAX_QUANTITIES]
    detail += [panel(f"{q} median: all vs MSUN", relax(f"{q}_median", f"{q}_median_msun"))
               for q in RELAX_SUBSETS["msun"]]
    detail += [panel(f"{q} median: SG kept vs changed", relax(f"{q}_median_sg_kept", f"{q}_median_sg_changed"))
               for q in RELAX_SUBSETS["sg_kept"]]
    detail += [panel(x, relax(x)) for x in ("sg_kept_rate_msun", "telemetry_count")]

    validation = individual("val")
    families = {}
    for key in keys:
        if key.startswith(("val_sigma/", "val_coord_class/")):
            group = re.sub(r"_s[^_]+$", "", key)
            families.setdefault(group, []).append(key)
    validation += [panel(group.split("/", 1)[1], series) for group, series in families.items()]

    symmetry = []
    for label in ("raw", "relaxed"):
        prefix = f"sym_{label}/"
        families = {}
        shares = {tol: [] for tol in SYMMETRY_TOLERANCES}
        for key in keys:
            if not key.startswith(prefix):
                continue
            metric, tol = key[len(prefix):].rsplit("_", 1)
            if metric.startswith("cs_") and metric != "cs_retention":
                shares[tol].append(key)
            else:
                families.setdefault(metric, []).append(key)
        symmetry += [panel(f"{label}: {metric}", series) for metric, series in families.items()]
        symmetry += [panel(f"{label}: crystal systems ({tol})", series) for tol, series in shares.items()]

    failures = list(log_view({f"orbit/failures/{reason}": 0 for reason in DECODE_REASONS}))
    sections = [
        ("Headline", True, headlines),
        ("Relaxation", True, relaxation),
        ("Relaxation detail", False, detail),
        ("Hull", True, individual("eval_hull")),
        ("Training", True, individual("train")),
        ("Validation", True, validation),
        ("Perf", False, individual("perf")),
        ("Counts", False, individual("eval_count") +
         [panel("Relaxation failure reasons", [], metric_regex=r"^eval_count/relax_fail_")]),
        ("Symmetry", False, symmetry),
        ("Decode", False, [panel("Failure reasons", failures, metric_regex=r"^decode/fail_")]
         + individual("decode")),
        ("Status", False, individual("status")),
    ]
    return [dict(name=name, is_open=is_open, panels=panels) for name, is_open, panels in sections]


def build_workspace(*, entity, project, name="Orbitol"):
    """Construct the wandb_workspaces 0.4.5 objects without saving or using the network."""
    import wandb_workspaces.workspaces as ws
    import wandb_workspaces.reports.v2 as wr

    sections = [ws.Section(name=section["name"], is_open=section["is_open"], panels=[
        # Leave the axis unset so W&B can use the run's defined step metric.
        wr.LinePlot(x=None, **panel)
        for panel in section["panels"]]) for section in build_sections()]
    return ws.Workspace(entity=entity, project=project, name=name, sections=sections,
                        auto_generate_panels=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    entity = C.local_value("ORBITOL_WANDB_ENTITY")
    parser.add_argument("--entity", default=entity, required=entity is None)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", default="Orbitol")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(entity=args.entity, project=args.project, name=args.name,
                              sections=build_sections()), indent=2))
        return
    workspace = build_workspace(entity=args.entity, project=args.project, name=args.name)
    workspace.save()
    print(workspace.url)


if __name__ == "__main__":
    main()
