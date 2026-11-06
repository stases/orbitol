"""Verify release inputs, CPU snapshots, sample tensors and Table 12 metrics."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import json
import math
from pathlib import Path
import sys

from orbitol import contracts as C
from tools.release.fetch import destination, public_inputs

ROOT = Path(__file__).resolve().parents[2]
POOL_KEYS = ("rows", "requested_G", "skeletons", "raw", "contains_a", "failures", "failure_counts")
REQUIRED_POOL_KEYS = ("rows", "requested_G", "failures")  # sweep files keep only these
RELOCATABLE_POOL_KEYS = {"metadata", "evaluation_settings"}  # paths and run identity; never compared
DATA_EXCLUSIONS = {"checkpoint", "paper_config", "attribution"}
# Same seed on the same GPU type: 0.4504 vs 0.4506 across release-check runs; relaxation is not bitwise.
TABLE12_TOLERANCE = 0.01


def manifest_at(path=None):
    path = Path(path) if path else destination("MANIFEST.json")
    return json.loads(path.read_text())


def content_matches(item, path, manifest):
    from orbitol.data import cache_content_sha256
    from orbitol.tables import load_tables
    from tools.build_nequip import weights_digest

    role, digests = item.get("role"), manifest.get("content_digests", {})
    if role == "tables":
        return load_tables(C.local_path("ORBITOL_TABLES_DIR")).sha256 == digests.get("tables")
    if role == "orbit_cache" and path.suffix == ".npz":
        return cache_content_sha256(path) == digests.get(item["path"])
    if role == "relaxer":
        kind = next((kind for kind, key in (("torchscript", "ORBITOL_NEQUIP_TS"), ("oeq", "ORBITOL_NEQUIP_OEQ"))
                     if item["path"] == C.DATA_LAYOUT[key]), None)
        return kind is not None and weights_digest(path) == manifest["nequip"]["weights_digest"][kind]
    if role == "symmetry_reference":
        # Reuse the read-only validator, which calls reference_valid and checks histograms.
        from orbitol.train import verify_symmetry_reference
        dataset = item.get("origin", item["path"].split("/")[0])
        verify_symmetry_reference(C.dataset_spec(dataset).data_root, path)
        return True
    return False


def verify_data(manifest):
    from orbitol.data import file_sha256

    records = []
    for item in manifest["files"]:
        if item.get("role") in DATA_EXCLUSIONS:
            continue
        path = destination(item["path"])
        record = dict(path=str(path), bundle_path=item["path"], status="missing")
        try:
            if path.is_file():
                actual = file_sha256(path)
                record.update(sha256=actual, status="wrong")
                if actual == item["sha256"]:
                    record["status"] = "exact"
                elif content_matches(item, path, manifest):
                    record["status"] = "content"
        except Exception as exc:
            record.update(status="wrong", error=str(exc))
        records.append(record)
    # Every MP-20 recipe needs the public MP-20 inputs (fetch --public mp20-raw,mp20-hull).
    groups = public_inputs()
    for group in ("mp20-raw", "mp20-hull"):
        for item in groups[group]:
            path = destination(item["path"])
            record = dict(path=str(path), group=group, status="missing")
            records.append(record)
            if not path.is_file():
                continue
            expected = item.get("gunzip", item)["sha256"]
            record["status"] = "wrong"
            try:
                actual = file_sha256(path)
                record.update(sha256=actual, status="exact" if actual == expected else "wrong")
            except Exception as exc:
                record["error"] = str(exc)
    return dict(ok=bool(records) and all(r["status"] in ("exact", "content") for r in records), files=records)


def verify_checkpoints(manifest):
    from orbitol.data import file_sha256
    from orbitol.sample import isolated_rng, load_snapshot

    records = []
    for model in manifest["models"]:
        path = destination(model["snapshot"])
        record = dict(name=model["name"], path=str(path), status="missing")
        try:
            if path.is_file():
                record.update(status="wrong", sha256=file_sha256(path))
                if record["sha256"] != model["snapshot_sha256"]:
                    raise ValueError("Snapshot sha256 differs from manifest")
                with isolated_rng():
                    inputs, metadata = load_snapshot(path, device="cpu",
                                                    overrides={"compile_trunk": "none", "sample_token_budget": 0})
                record.update(step=metadata["step"], dataset=inputs["cfg"].dataset)
                del inputs
                if record["step"] != model["step"] or record["dataset"] != model["dataset"]:
                    raise ValueError("Snapshot step/dataset differs from manifest")
                record["status"] = "exact"
        except Exception as exc:
            record.update(status="wrong", error=str(exc))
        records.append(record)
    return dict(ok=bool(records) and all(r["status"] == "exact" for r in records), models=records)


def verify_nequip():
    from orbitol.data import file_sha256
    from tools.build_nequip import EXPECTED, PAPER_SHA256, weights_digest

    records = []
    for kind, key in (("torchscript", "ORBITOL_NEQUIP_TS"), ("oeq", "ORBITOL_NEQUIP_OEQ")):
        path = Path(C.local_path(key)).absolute()
        record = dict(kind=kind, path=str(path), status="missing", paper_identical=False)
        try:
            if path.is_file():
                actual = weights_digest(path)
                sha = file_sha256(path)
                record.update(weights_digest=actual, expected_weights_digest=EXPECTED[kind], sha256=sha,
                              paper_identical=sha == PAPER_SHA256["oeq_sm89" if kind == "oeq" else kind],
                              status="content" if actual == EXPECTED[kind] else "wrong")
        except Exception as exc:
            record.update(status="wrong", error=str(exc))
        records.append(record)
    return dict(ok=all(r["status"] == "content" for r in records), artifacts=records)


def bits(x):
    """A float or bool tensor or array as same-shape integers, so that equality is bitwise."""
    import numpy as np
    import torch

    if torch.is_tensor(x):
        sizes = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        return x.contiguous().view(sizes[x.element_size()]) if x.is_floating_point() or x.dtype == torch.bool else x
    return np.ascontiguousarray(x).view(f"u{x.itemsize}") if x.dtype.kind in "fb" else x


def is_skeleton(entry):
    """Whether entry is a dict of a Skeleton (G, rows and optional A0, all integers)."""
    def integers(values):
        return isinstance(values, (tuple, list)) and all(type(v) is int for v in values)

    return (isinstance(entry, dict) and set(entry) <= {"G", "rows", "A0"} and type(entry.get("G")) is int
            and integers(entry.get("rows")) and (entry.get("A0") is None or integers(entry["A0"])))


def first_difference(left, right, key, index=()):
    """Return the first bitwise difference, including its tensor/list index."""
    import numpy as np
    import torch

    def difference(reason, suffix=()):
        return dict(key=key, index=list(index + tuple(suffix)), reason=reason)

    if torch.is_tensor(left) or torch.is_tensor(right):
        if not (torch.is_tensor(left) and torch.is_tensor(right)):
            return difference("type")
        if left.shape != right.shape or left.dtype != right.dtype:
            return difference("shape or dtype")
        a, b = bits(left), bits(right)
        if not torch.equal(a, b):
            return difference("value", (a != b).nonzero()[0].tolist())
    elif isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)):
            return difference("type")
        if left.shape != right.shape or left.dtype != right.dtype:
            return difference("shape or dtype")
        a, b = bits(left), bits(right)
        if not np.array_equal(a, b):
            return difference("value", np.argwhere(a != b)[0].tolist())
    elif type(left) is not type(right):
        return difference("type")
    elif isinstance(left, dict):
        for name in sorted(left.keys() | right.keys(), key=str):
            subkey = f"{key}.{name}" if isinstance(name, str) else key
            subindex = index if isinstance(name, str) else index + (name,)
            if name not in left or name not in right:
                return dict(key=subkey, index=list(subindex), reason="missing key")
            result = first_difference(left[name], right[name], subkey, subindex)
            if result:
                return result
    elif isinstance(left, (list, tuple)):
        for i, (a, b) in enumerate(zip(left, right)):
            result = first_difference(a, b, key, index + (i,))
            if result:
                return result
        if len(left) != len(right):
            return difference("length", (min(len(left), len(right)),))
    elif left != right:
        return difference("value")
    return None


def pool_problem(pool):
    """Why a pool is malformed, or None."""
    import torch

    if not isinstance(pool, dict):
        return "not a dictionary"
    unknown = sorted(set(pool) - set(POOL_KEYS) - RELOCATABLE_POOL_KEYS, key=str)
    missing = [key for key in REQUIRED_POOL_KEYS if key not in pool]
    if unknown or missing:
        return f"unknown keys {unknown}, missing keys {missing}"
    groups = pool["requested_G"]
    if not torch.is_tensor(groups) or groups.dim() != 1 or not len(groups) \
            or groups.dtype not in (torch.int32, torch.int64):
        return "requested_G must be a nonempty one-dimensional integer tensor"
    n = len(groups)
    for key in ("rows", "raw"):
        if key not in pool:
            continue
        if not isinstance(pool[key], dict) or not pool[key]:
            return f"{key} must be a nonempty dictionary of tensors"
        for name, x in pool[key].items():
            if type(name) is not str:
                return f"{key} keys must be strings"
            if not torch.is_tensor(x) or not x.dim() or len(x) != n or not x.numel() or x.is_complex():
                return f"{key}.{name} must be a real tensor with {n} attempts"
            if x.is_floating_point() and not torch.isfinite(x).all():
                return f"{key}.{name} has non-finite values"
    if "skeletons" in pool and (not isinstance(pool["skeletons"], list) or len(pool["skeletons"]) != n
                                or not all(map(is_skeleton, pool["skeletons"]))):
        return f"skeletons must be a list of {n} skeletons"
    if "contains_a" in pool and not (torch.is_tensor(pool["contains_a"]) and pool["contains_a"].dtype == torch.bool
                                     and pool["contains_a"].shape == (n,)):
        return f"contains_a must be a boolean tensor of {n} attempts"
    failures = pool["failures"]
    if not isinstance(failures, dict) or any(type(i) is not int or not 0 <= i < n or type(reason) is not str
                                             for i, reason in failures.items()):
        return "failures must map attempt indices to reasons"
    if "failure_counts" in pool and pool["failure_counts"] != dict(Counter(failures.values())):
        return "failure_counts disagree with failures"
    return None


def differing_attempts(left, right):
    """For two valid pools, the mask of attempts whose generated tensors (rows, raw) differ bitwise. None
    unless everything else matches except the decode outcomes of those attempts: keys, requested groups,
    skeletons, tensor shapes and dtypes."""
    import torch

    if set(left) - RELOCATABLE_POOL_KEYS != set(right) - RELOCATABLE_POOL_KEYS:
        return None
    if any(first_difference(left.get(key), right.get(key), key) for key in ("requested_G", "skeletons")):
        return None
    n = len(left["requested_G"])
    mask = torch.zeros(n, dtype=torch.bool)
    for key in ("rows", "raw"):
        a, b = left.get(key, {}), right.get(key, {})
        if a.keys() != b.keys():
            return None
        for name in a:
            if a[name].shape != b[name].shape or a[name].dtype != b[name].dtype:
                return None
            mask |= (bits(a[name]) != bits(b[name])).reshape(n, -1).any(1)
    absent = object()
    failures = left["failures"], right["failures"]
    changed = {i for i in failures[0].keys() | failures[1].keys()
               if failures[0].get(i, absent) != failures[1].get(i, absent)}
    if "contains_a" in left:
        changed |= set((bits(left["contains_a"]) != bits(right["contains_a"])).nonzero().flatten().tolist())
    return mask if all(mask[i] for i in changed) else None


def verify_pools(left, right, max_differing=0):
    """Compare two pools. By default everything except relocatable metadata must be bitwise equal. With
    max_differing > 0, up to that many attempts may differ, in their generated tensors and the decode outcomes of
    those attempts only; see differing_attempts. Malformed pools always fail."""
    import torch

    paths = [Path(p) / "samples.pt" if Path(p).is_dir() else Path(p) for p in (left, right)]
    pools = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    invalid = {side: problem for side, problem in zip(("left", "right"), map(pool_problem, pools)) if problem}
    difference = mask = None
    if not invalid:
        for key in sorted((set(pools[0]) | set(pools[1])) - RELOCATABLE_POOL_KEYS, key=POOL_KEYS.index):
            if key not in pools[0] or key not in pools[1]:
                difference = dict(key=key, index=[], reason="missing key")
            else:
                difference = first_difference(pools[0][key], pools[1][key], key)
            if difference:
                break
        mask = differing_attempts(*pools)
    count = None if mask is None else int(mask.sum())
    return dict(ok=not invalid and (difference is None or (count is not None and 0 < count <= max_differing)),
                equal=not invalid and difference is None, invalid=invalid or None,
                left=str(paths[0].absolute()), right=str(paths[1].absolute()), first_difference=difference,
                differing_structures=count, structures=None if mask is None else len(mask))


def verify_table12(directory, tolerance=TABLE12_TOLERANCE, partial=False):
    """Compare sweep JSONs with the paper's Table 12 rates; unless partial, every paper seed of each
    setting found must be present."""
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and nonnegative (absolute rate difference)")
    models = json.loads((ROOT / "configs/paper/models.json").read_text())
    records, found = [], {}
    for path in sorted(Path(directory).rglob("*.json")):
        report = json.loads(path.read_text())
        if not isinstance(report, dict) or not {"logged", "overrides", "seed"} <= report.keys():
            continue
        record = dict(path=str(path.absolute()), seed=report["seed"], status="wrong")
        try:
            matches = []
            sampler = report.get("sampler", report["overrides"])
            for model in models:
                table = model.get("expected", {}).get("table12", {})
                for name in ("chosen", "stored"):
                    setting = table.get(name)
                    if setting and all(sampler.get(k) == v for k, v in setting["overrides"].items()):
                        matches.append((model, table, name, setting))
            if len(matches) != 1:
                raise ValueError("Sweep does not match exactly one Table 12 sampler setting")
            model, table, name, setting = matches[0]
            record["setting"] = name
            found.setdefault(name, (setting, set()))[1].add(report["seed"])
            if report["samples"] != table["n"] or report["checkpoint_step"] != model["step"]:
                raise ValueError("Sample count or checkpoint step differs from Table 12")
            if sampler.get("decode_impl") != table["decode_impl"]:
                raise ValueError("Decoder differs from Table 12")
            seed_index = setting["seeds"].index(report["seed"])
            metrics = {}
            for metric, logged in (("MSUN", "eval/MSUN"), ("UN", "eval/un_rate"),
                                   ("one_shot", "eval/MSUN_oneshot")):
                if metric not in setting:
                    continue  # Stored UN/one-shot were not recorded in the paper registry.
                expected, actual = setting[metric][seed_index], float(report["logged"][logged])
                if not math.isfinite(actual):
                    raise ValueError(f"Non-finite {metric}")
                metrics[metric] = dict(actual=actual, expected=expected, delta=actual - expected,
                                       ok=abs(actual - expected) <= tolerance)
            record.update(metrics=metrics, status="exact" if all(m["ok"] for m in metrics.values()) else "wrong")
        except Exception as exc:
            record["error"] = str(exc)
        records.append(record)
    missing = {name: sorted(set(setting["seeds"]) - seeds) for name, (setting, seeds) in found.items()
               if set(setting["seeds"]) - seeds}
    return dict(ok=bool(records) and all(r["status"] == "exact" for r in records) and (partial or not missing),
                tolerance=tolerance, missing_seeds=missing, sweeps=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "checkpoints"):
        commands.add_parser(name).add_argument("--manifest", type=Path,
                                               help="Default: ORBITOL_DATA_HOME/MANIFEST.json")
    commands.add_parser("nequip")
    pools = commands.add_parser("pools")
    pools.add_argument("left", type=Path)
    pools.add_argument("right", type=Path)
    pools.add_argument("--max-differing", type=int, default=0,
                       help="Accept up to N attempts that differ only in their generated tensors and decode "
                            "outcomes (default: 0, bitwise)")
    table = commands.add_parser("table12")
    table.add_argument("directory", type=Path)
    table.add_argument("--tolerance", type=float, default=TABLE12_TOLERANCE,
                       help=f"Absolute rate tolerance per seed (default: {TABLE12_TOLERANCE})")
    table.add_argument("--partial", action="store_true", help="Accept a subset of the paper seeds")
    args = parser.parse_args(argv)
    try:
        # Keep stdout machine-readable even if a loader prints progress.
        with redirect_stdout(sys.stderr):
            import torch
            torch.set_num_threads(1)
            if args.command == "data":
                result = verify_data(manifest_at(args.manifest))
            elif args.command == "checkpoints":
                result = verify_checkpoints(manifest_at(args.manifest))
            elif args.command == "nequip":
                result = verify_nequip()
            elif args.command == "pools":
                result = verify_pools(args.left, args.right, args.max_differing)
            else:
                result = verify_table12(args.directory, args.tolerance, args.partial)
    except Exception as exc:
        result = dict(ok=False, error=str(exc))
    print(json.dumps(dict(command=args.command, **result), allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
