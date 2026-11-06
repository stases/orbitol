"""CPU-only export and assembly of the release bundle (hosted on Hugging Face)."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from orbitol import contracts as C
from orbitol.data import cache_content_sha256, file_sha256
from orbitol.eval_snapshot import SNAPSHOT_KEYS, ema_only
from orbitol.posthoc_io import atomic_writer, write_json
from orbitol.sample import isolated_rng, load_snapshot
from orbitol.tables import load_tables
from tools.build_nequip import PAPER_SHA256, weights_digest

ROOT = Path(__file__).resolve().parents[2]
RESERVE = 20 * 1024**3
# Cluster placement stored in a training cfg; local configuration, not part of the release.
PLACEMENT_KEYS = ("eval_partition", "eval_account")
# Build records whose JSON holds machine paths (cache manifests, symmetry references).
PATH_RECORD_ROLES = ("orbit_cache", "symmetry_reference")
HOSTING = "https://huggingface.co/stases/orbitol"


def paper_models():
    return json.loads((ROOT / "configs/paper/models.json").read_text())


def require_space(out, size):
    """Check before creating anything, including on a not-yet-existing output path."""
    parent = Path(out).resolve()
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < size + RESERVE:
        raise OSError(f"{parent}: need {size + RESERVE} free bytes (includes 20 GiB reserve)")


def export(source, out):
    """Verify a paper source, export its EMA state, and check the real CPU loader."""
    source, out = Path(source).resolve(), Path(out).resolve()
    require_space(out, source.stat().st_size)
    source_sha = file_sha256(source)
    matches = [m for m in paper_models() if m["source_sha256"] == source_sha]
    if len(matches) != 1 or source.stat().st_size != matches[0]["source_size"]:
        raise ValueError(f"Not an exact source checkpoint from the paper model registry: {source}")
    model = matches[0]
    payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    if (payload["step"] != model["step"] or payload["cfg"].get("dataset", "mp20") != model["dataset"]
            or payload["bank_sha256"] != model["bank_sha256"]):
        raise ValueError(f"Checkpoint metadata differs from paper registry: {source}")
    result = ema_only(payload)
    result["cfg"] = {**result["cfg"], **{key: "" for key in PLACEMENT_KEYS if result["cfg"].get(key)}}
    target = out / model["snapshot"]
    if target == source:
        raise ValueError("Export must not overwrite its source")
    with atomic_writer(target) as stream:
        torch.save(result, stream)
    saved = torch.load(target, map_location="cpu", weights_only=False, mmap=True)
    if set(saved) != set(result):
        raise ValueError("Export changed snapshot keys")
    for key in SNAPSHOT_KEYS:
        if key in result:
            same = (torch.equal(saved[key], result[key]) if torch.is_tensor(result[key])
                    else saved[key] == result[key])
            if not same:
                raise ValueError(f"Export changed metadata: {key}")
    for kind in ("ema", "model"):
        for name, tensor in result[kind].items():
            if not torch.equal(saved[kind][name], tensor):
                raise ValueError(f"Export changed {kind}.{name}")
    for name in saved["ema"]:
        if saved["model"][name].data_ptr() != saved["ema"][name].data_ptr():
            raise ValueError(f"Export duplicated EMA storage: {name}")
    # Tiny test archives are dominated by metadata; real model archives must shrink.
    if source.stat().st_size > 1_000_000 and target.stat().st_size >= .55 * source.stat().st_size:
        raise ValueError("EMA export did not achieve the expected storage reduction")
    with isolated_rng():
        load_snapshot(target, device="cpu", overrides={"compile_trunk": "none", "sample_token_budget": 0})
    return target


def inventory():
    """Explicit allowlist; input paths resolve only through local configuration."""
    files = []

    def add(source, relative, role, origin):
        files.append((Path(source), Path(relative).as_posix(), role, origin))

    for name in ("tables.npz", "cosets.npz", "manifest.json"):
        add(Path(C.local_path("ORBITOL_TABLES_DIR")) / name,
            Path(C.DATA_LAYOUT["ORBITOL_TABLES_DIR"]) / name, "tables", "PyXtal 1.1.2; spglib 2.7.0")
    for dataset in dict.fromkeys(m["dataset"] for m in paper_models()):
        spec = C.dataset_spec(dataset)
        key = f"ORBITOL_{dataset.upper()}_"
        # Only the splits that training and evaluation read; processed test splits are not bundled.
        for split in ("train", "val"):
            for suffix in (".npz", ".manifest.json"):
                add(Path(spec.orbit_cache_dir) / (split + suffix),
                    Path(C.DATA_LAYOUT[key + "ORBIT_CACHE_DIR"]) / (split + suffix), "orbit_cache", dataset)
            name = f"processed/mp20_tokens_{split}.pt"
            add(Path(spec.data_root) / name, Path(C.DATA_LAYOUT[key + "DATA_ROOT"]) / name, "processed_tokens", dataset)
        add(spec.symmetry_reference, C.DATA_LAYOUT[key + "SYMMETRY_REFERENCE"], "symmetry_reference", dataset)
    for key in ("ORBITOL_NEQUIP_TS", "ORBITOL_NEQUIP_OEQ"):
        add(C.local_path(key), C.DATA_LAYOUT[key], "relaxer", "NequIP-OAM-L; doi:10.5281/zenodo.16980200")
    for path in sorted((ROOT / "configs/paper").glob("*.json")):
        add(path, path.relative_to(ROOT), "paper_config", "Orbitol")
    add(ROOT / "tools/release/ATTRIBUTION.md", "ATTRIBUTION.md", "attribution", "Orbitol")
    return files


def relocated(value):
    """Replace absolute paths in a build record by their file names; hashes and other values stay."""
    if isinstance(value, dict):
        return {key: relocated(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relocated(item) for item in value]
    return Path(value).name if isinstance(value, str) and value.startswith("/") else value


@contextmanager
def bundle_paths(out):
    """Check loaders against shipped files, even when local overrides are set."""
    replacements = {key: str(out / relative) for key, relative in C.DATA_LAYOUT.items()}
    old_env = {key: os.environ.get(key) for key in replacements}
    old_datasets = dict(C.DATASETS)
    try:
        os.environ.update(replacements)
        C.DATASETS.clear()
        yield
    finally:
        C.DATASETS.clear()
        C.DATASETS.update(old_datasets)
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assemble(out):
    """Copy exact inputs, re-stamp references, then publish an exhaustive inventory."""
    out = Path(out).resolve()
    models, files = paper_models(), inventory()
    for model in models:
        path = out / model["snapshot"]
        if not path.is_file():
            raise FileNotFoundError(f"Export the checkpoint first: {path}")
    expected = {relative for _, relative, _, _ in files} | {m["snapshot"] for m in models}
    unexpected = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()} - expected - {
        "MANIFEST.json", "SHA256SUMS"}
    if unexpected:
        raise ValueError(f"Unexpected files in bundle: {sorted(unexpected)}")
    for source, relative, _, _ in files:
        if not source.is_file():
            raise FileNotFoundError(source)
        if (out / relative).resolve() == source.resolve():
            raise ValueError(f"Bundle destination overlaps an input: {source}")
    require_space(out, sum(p.stat().st_size for p, _, _, _ in files) + 1024**2)
    # Invalidate a previous inventory before changing payload files.
    for name in ("MANIFEST.json", "SHA256SUMS"):
        (out / name).unlink(missing_ok=True)
    for source, relative, role, origin in files:
        target = out / relative
        with atomic_writer(target) as stream, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, stream)
        if role in PATH_RECORD_ROLES and target.suffix == ".json":
            record = relocated(json.loads(target.read_text()))
            if role == "symmetry_reference":
                tokens = out / C.DATA_LAYOUT[f"ORBITOL_{origin.upper()}_DATA_ROOT"] / "processed/mp20_tokens_train.pt"
                record["metadata"]["train_cache_sha256"] = file_sha256(tokens)
            write_json(target, record)
    with bundle_paths(out), isolated_rng():
        for model in models:
            inputs, metadata = load_snapshot(out / model["snapshot"], device="cpu",
                                            overrides={"compile_trunk": "none", "sample_token_budget": 0})
            if metadata["step"] != model["step"] or inputs["cfg"].dataset != model["dataset"]:
                raise ValueError(f"Wrong paper snapshot: {out / model['snapshot']}")
            del inputs
    entries, content = [], {}
    for model in models:
        path = out / model["snapshot"]
        model.update(snapshot_sha256=file_sha256(path), snapshot_size=path.stat().st_size)
        files.append((path, model["snapshot"], "checkpoint", "Orbitol; source sha256:" + model["source_sha256"]))
    for _, relative, role, origin in sorted(files, key=lambda item: item[1]):
        path = out / relative
        entries.append(dict(path=relative, size=path.stat().st_size, sha256=file_sha256(path),
                            role=role, licence="CC-BY-4.0", origin=origin))
        if role == "orbit_cache" and path.suffix == ".npz":
            content[relative] = cache_content_sha256(path)
    content["tables"] = load_tables(out / C.DATA_LAYOUT["ORBITOL_TABLES_DIR"]).sha256
    manifest = dict(release="orbitol-release-v1", code_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        created_utc=datetime.now(timezone.utc).isoformat(), hosting=HOSTING, licence="CC-BY-4.0",
        code_licence="MIT", files=entries, models=models, content_digests=content,
        nequip=dict(weights_digest={kind: weights_digest(out / C.DATA_LAYOUT[key]) for kind, key in
                    (("torchscript", "ORBITOL_NEQUIP_TS"), ("oeq", "ORBITOL_NEQUIP_OEQ"))},
                    paper_sha256=PAPER_SHA256),
        excluded={"mp20_raw": "public download", "mp20_hull": "public download",
                  "alex_mp20": "model and data not released", "reference_outputs": "not included"})
    public = ROOT / "tools/release/public_inputs.json"
    if public.is_file():
        manifest["public_inputs"] = json.loads(public.read_text())
    write_json(out / "MANIFEST.json", manifest)
    checksums = [(item["sha256"], item["path"]) for item in entries]
    checksums.append((file_sha256(out / "MANIFEST.json"), "MANIFEST.json"))
    with atomic_writer(out / "SHA256SUMS") as stream:
        stream.write("".join(f"{sha}  {path}\n" for sha, path in sorted(checksums, key=lambda x: x[1])).encode())
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    exp = commands.add_parser("export", help="Export one or more exact paper checkpoints to EMA-only snapshots")
    exp.add_argument("sources", type=Path, nargs="+")
    exp.add_argument("--out", type=Path, required=True, help="Bundle root (ORBITOL_DATA_HOME layout)")
    asm = commands.add_parser("assemble", help="Assemble the bundle after exporting the paper checkpoints")
    asm.add_argument("--out", type=Path, required=True, help="Bundle root containing exported checkpoints")
    args = parser.parse_args(argv)
    if args.command == "export":
        for source in args.sources:
            print(export(source, args.out))
    else:
        manifest = assemble(args.out)
        print(f"{args.out.resolve()}: {len(manifest['files'])} files, "
              f"{sum(item['size'] for item in manifest['files'])} payload bytes")


if __name__ == "__main__":
    main()
