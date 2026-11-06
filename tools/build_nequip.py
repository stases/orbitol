"""Build and verify the public NequIP-OAM-L TorchScript and OEQ artifacts."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile

import certifi

SOURCE = dict(
    url="https://zenodo.org/api/records/16980200/files/NequIP-OAM-L-0.1.nequip.zip/content",
    doi="10.5281/zenodo.16980200", file="NequIP-OAM-L-0.1.nequip.zip", size=78473421,
    md5="9700a25ea433aa5cc11aa54cf2e65faf",
    sha256="aee839e4f24d3e7c1af8104ecfee25decbdf98b3ce7e959806cfaacfac637c8d",
)
EXPECTED = {
    "torchscript": "99019472150c41a7d6ff0a189a01f6cd83b369ed422f2d4537d66698f91b2ebd",
    "oeq": "4c56f97dbb4ea3de54d5f77ce8066b317635c5f51247fe0d39370cc89b1804ea",
}
PAPER_SHA256 = {
    "torchscript": "0bf01d4ddc2e53041581a2de38e28af6a3f10df8396a5bd481be8131a417fdc7",
    "oeq_sm89": "507cbd2de97e86174f038286862b4b0d5bdbeb957315aca898f32600223bbe34",
}


def weights_digest(path) -> str:
    """Hash sorted tensor storage and metadata members, ignoring the zip root."""
    h = hashlib.sha256()
    with zipfile.ZipFile(path) as z:
        members = sorted((n.split("/", 1)[1], n) for n in z.namelist() if "/" in n)
        for rel, name in members:
            if rel.startswith(("data/", "extra/")):
                h.update(rel.encode() + b"\0")
                h.update(z.read(name))
    return h.hexdigest()


def file_hash(path, algorithm="sha256"):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def download_source(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SOURCE["file"]
    candidate = path if path.exists() else path.with_suffix(path.suffix + ".part")
    if candidate != path:
        if shutil.disk_usage(directory).free < SOURCE["size"] + 1024**3:
            raise RuntimeError("Download requires the source size plus 1 GiB of free-space headroom")
        # certifi's CA bundle: uv's standalone Python does not see the system certificates.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(SOURCE["url"], context=context) as response, candidate.open("wb") as out:
            shutil.copyfileobj(response, out)
    if (candidate.stat().st_size != SOURCE["size"] or
            any(file_hash(candidate, k) != SOURCE[k] for k in ("md5", "sha256"))):
        raise ValueError(f"Source size/checksum mismatch: {candidate}")
    if candidate != path:
        candidate.replace(path)
    return path


def energy_check(ts, oeq, reference):
    from pymatgen.core import Lattice, Structure
    from orbitol import nequip_relax as relax
    structures = [
        Structure.from_spacegroup(227, Lattice.cubic(5.43), ["Si"], [[0, 0, 0]]),
        Structure.from_spacegroup(225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [.5, 0, 0]]),
        Structure(Lattice.cubic(4.12), ["Cs", "Cl"], [[0, 0, 0], [.5, .5, .5]]),
        Structure.from_spacegroup(
            186, Lattice.hexagonal(3.25, 5.2), ["Zn", "O"], [[1 / 3, 2 / 3, 0], [1 / 3, 2 / 3, .375]],
        ),
    ]
    backend = relax.make_batch_relaxer(ts, "cuda")
    initial = backend.relax_many(structures, steps=0)
    left = backend.relax_many(structures, steps=1)
    right = relax.make_fast_batch_relaxer(oeq, "cuda").relax_many(structures, steps=1)

    def difference(a, b, initial=False):
        values = [
            abs(x["trajectory"].energies[0] - (y["initial_energy"] if initial else y["trajectory"].energies[0]))
            / len(s)
            for x, y, s in zip(a, b, structures, strict=True)
        ]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite parity energies")
        return max(values)

    report = dict(
        max_abs_dE_per_atom=difference(initial, right, True),
        max_abs_endpoint_dE_per_atom=difference(left, right), steps=1,
    )
    if max(report["max_abs_dE_per_atom"], report["max_abs_endpoint_dE_per_atom"]) > 1e-4:
        raise ValueError(f"OEQ/TorchScript parity failed: {report}")
    if reference:
        ref = relax.make_fast_batch_relaxer(reference, "cuda").relax_many(structures, steps=1)
        report["reference_max_abs_dE_per_atom"] = difference(right, ref)
    return dict(report, passed=True)


def capture(*command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable: {exc}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("ts", "oeq", "source-dir", "reference-oeq"):
        parser.add_argument("--" + flag, type=Path)
    parser.add_argument("--only", choices=tuple(EXPECTED))
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="replace existing artifacts")
    args = parser.parse_args(argv)

    if args.ts is None or args.oeq is None or args.source_dir is None:
        from orbitol import nequip_relax as relax
        args.ts = args.ts or Path(relax.DEFAULT_TS_PATH)
        args.oeq = args.oeq or Path(relax.DEFAULT_OEQ_PATH)
        args.source_dir = args.source_dir or Path(relax.DEFAULT_OEQ_PATH).parent.parent / "source"
    paths = {"torchscript": args.ts.expanduser().resolve(), "oeq": args.oeq.expanduser().resolve()}
    selected = [args.only] if args.only else list(paths)
    for kind in selected:
        if not str(paths[kind]).endswith(".nequip.pth") or (paths[kind].exists() and not args.force):
            parser.error(f"Expected unused .nequip.pth output (or --force): {paths[kind]}")
    if paths["torchscript"] == paths["oeq"]:
        parser.error("--ts and --oeq must differ")
    if args.check and args.only and not paths[next(k for k in paths if k != args.only)].is_file():
        parser.error("--check needs both artifacts; supply the other artifact or use --no-check")
    if args.reference_oeq and args.reference_oeq.expanduser().resolve() in (paths[k] for k in selected):
        parser.error("Reference artifact must differ from build outputs")

    source = download_source(args.source_dir.expanduser().resolve())
    import torch
    from orbitol import nequip_relax as relax
    if not torch.cuda.is_available():
        parser.error("CUDA GPU required")
    if "oeq" in selected and (
        not shutil.which("ninja") or not (Path(os.getenv("CUDA_HOME", "/missing")) / "include").is_dir()
    ):
        parser.error("OEQ requires ninja on PATH and CUDA_HOME with include/")
    versions = {p: metadata.version(p) for p in ("torch", "nequip", "e3nn", "numpy")}
    versions["openequivariance"] = metadata.version("openequivariance") if "oeq" in selected or args.check else None

    receipts = {}
    for kind in selected:
        out = paths[kind]
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp.nequip.pth")
        modifiers = ["enable_OpenEquivariance"] if kind == "oeq" else []
        command = [
            sys.executable, "-u", "-X", "faulthandler", "-m", "nequip.scripts.compile", str(source), str(tmp),
            "--mode", "torchscript", "--device", "cuda", "--no-tf32",
        ] + (["--modifiers", *modifiers] if modifiers else [])
        subprocess.run(command, check=True, env=dict(os.environ, PYTHONHASHSEED="0"))
        digest = weights_digest(tmp)
        if digest != EXPECTED[kind] or relax._requires_oeq(tmp) != (kind == "oeq"):
            raise ValueError(f"Artifact integrity failure: {tmp}, weights_digest={digest}")
        Path(str(out) + ".build.json").unlink(missing_ok=True)
        tmp.replace(out)
        sha = file_hash(out)
        gpu = dict(
            name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(),
            driver=capture("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        )
        toolchain = dict(
            CUDA_HOME=os.getenv("CUDA_HOME"), nvcc=capture("nvcc", "--version"),
            cxx=capture(os.getenv("CXX", "g++"), "--version"),
        )
        paper_sha = PAPER_SHA256["oeq_sm89" if kind == "oeq" else kind]
        receipts[kind] = dict(
            source=SOURCE, command=command, mode="torchscript", modifiers=modifiers, tf32=False,
            pythonhashseed=0, versions=versions, gpu=gpu, toolchain=toolchain,
            sha256=sha, weights_digest=digest, expected_weights_digest=EXPECTED[kind],
            paper_sha256=PAPER_SHA256, paper_identical=sha == paper_sha,
        )

    check = (
        energy_check(paths["torchscript"], paths["oeq"], args.reference_oeq)
        if args.check else dict(passed=None, skipped=True)
    )
    for kind, receipt in receipts.items():
        Path(str(paths[kind]) + ".build.json").write_text(
            json.dumps(dict(receipt, check=check), indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(dict(artifact=str(paths[kind]), **receipt, check=check)), flush=True)


if __name__ == "__main__":
    main()
