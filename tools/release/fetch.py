"""Install a release bundle and pinned public inputs at local-config paths."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import ssl
import sys
import urllib.parse
import urllib.request

import certifi

from orbitol import contracts as C
from tools.build_nequip import SOURCE

PUBLIC_INPUTS = Path(__file__).with_name("public_inputs.json")
RESERVE = 1024**3


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\\" in value
            or str(path) != value or value == "."):
        raise ValueError(f"Invalid bundle path: {value!r}")
    return path


def destination(relative):
    """The most specific layout key wins over a dataset-root override."""
    path = relative_path(relative)
    for key, prefix in sorted(C.DATA_LAYOUT.items(), key=lambda item: len(item[1]), reverse=True):
        base = PurePosixPath(prefix)
        if path.is_relative_to(base):
            return (Path(C.local_path(key)) / path.relative_to(base)).absolute()
    return (Path(C.local_path("ORBITOL_DATA_HOME")) / path).absolute()


def public_inputs():
    groups = json.loads(PUBLIC_INPUTS.read_text())
    spec = groups["nequip-source"]
    groups["nequip-source"] = [dict(SOURCE, path=spec["directory"] + "/" + SOURCE["file"])]
    return groups


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def matches(path, spec):
    return (path.is_file() and path.stat().st_size == spec["size"]
            and all(digest(path, key) == spec[key] for key in ("sha256", "md5") if key in spec))


def require_space(path, size):
    parent = path.absolute()
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < size + RESERVE:
        raise OSError(f"{parent}: need {size + RESERVE} free bytes including 1 GiB reserve")


def open_url(url, offset=0):
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(request, context=ssl.create_default_context(cafile=certifi.where()))


def is_url(source):
    return urllib.parse.urlsplit(str(source)).scheme in ("https", "http")


def fetch_file(source, target, spec, *, force=False, dry_run=False):
    """Resume verified transfers; never replace the destination before validation."""
    target = Path(target)
    if dry_run:
        print(f"{source} → {target.absolute()}")
        return
    if not any(key in spec for key in ("sha256", "md5")):
        raise ValueError(f"No checksum for {target}")
    if matches(target, spec):
        print(f"present: {target.absolute()}")
        return
    if target.exists() and not force:
        raise ValueError(f"Refusing to overwrite mismatched file: {target.absolute()} (use --force)")
    part = target.with_name(target.name + ".part")
    require_space(target.parent, spec["size"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if not matches(part, spec):
        if is_url(source):
            offset = part.stat().st_size if part.exists() else 0
            if offset >= spec["size"]:
                offset = 0
            with open_url(str(source), offset) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (not match or int(match[1]) != offset or int(match[3]) != spec["size"]
                            or int(match[2]) != spec["size"] - 1):
                        raise ValueError(f"Invalid Content-Range for {source}")
                elif status != 200:
                    raise ValueError(f"Unexpected HTTP status {status}: {source}")
                # Servers may ignore Range; then restart instead of appending a whole file.
                with part.open("ab" if offset and status == 206 else "wb") as stream:
                    shutil.copyfileobj(response, stream)
        else:
            with Path(source).open("rb") as incoming, part.open("wb") as stream:
                shutil.copyfileobj(incoming, stream)
    if not matches(part, spec):
        raise ValueError(f"Size/checksum mismatch: {part.absolute()}")
    part.replace(target)
    print(f"installed: {target.absolute()}")


def fetch_public(item, *, force=False, dry_run=False):
    target = destination(item["path"])
    if "gunzip" not in item:
        return fetch_file(item["url"], target, item, force=force, dry_run=dry_run)
    unpacked = item["gunzip"]
    if dry_run:
        print(f"{item['url']} (gunzip) → {target}")
        return
    if matches(target, unpacked):
        print(f"present: {target}")
        return
    if target.exists() and not force:
        raise ValueError(f"Refusing to overwrite mismatched file: {target} (use --force)")
    require_space(target.parent, item["size"] + unpacked["size"])
    compressed = target.with_name(target.name + ".gz")
    fetch_file(item["url"], compressed, item, force=force)
    part = target.with_name(target.name + ".part")
    with gzip.open(compressed, "rb") as incoming, part.open("wb") as stream:
        shutil.copyfileobj(incoming, stream)
    if not matches(part, unpacked):
        raise ValueError(f"Gunzipped size/checksum mismatch: {part}")
    part.replace(target)
    print(f"installed: {target}")


def bundle_source(base, relative):
    """A bundle file under a local directory or an HTTP(S) base URL."""
    relative = str(relative_path(relative))
    if is_url(base):
        return str(base).rstrip("/") + "/" + urllib.parse.quote(relative)
    return str(Path(base).absolute() / relative)


def fetch_bundle(base, *, force=False, dry_run=False):
    source = bundle_source(base, "MANIFEST.json")
    if is_url(source):
        with open_url(source) as stream:
            raw = stream.read()
    else:
        raw = Path(source).read_bytes()
    manifest = json.loads(raw)
    seen = set()
    for item in manifest["files"]:
        relative_path(item["path"])
        if item["path"] in seen or item["path"] == "MANIFEST.json":
            raise ValueError(f"Duplicate/reserved manifest path: {item['path']}")
        seen.add(item["path"])
        if not re.fullmatch("[a-f0-9]{64}", item["sha256"]) or item["size"] < 0:
            raise ValueError(f"Invalid file metadata: {item['path']}")
    # Preflight collisions before placing any payload.
    targets = [destination(item["path"]) for item in manifest["files"]]
    if len(set(targets)) != len(targets):
        raise ValueError("Local overrides map multiple bundle files to one destination")
    target = destination("MANIFEST.json")
    if target.exists() and target.read_bytes() != raw and not (force or dry_run):
        raise ValueError(f"Refusing to overwrite mismatched file: {target} (use --force)")
    for item, path in zip(manifest["files"], targets):
        fetch_file(bundle_source(base, item["path"]), path, item, force=force, dry_run=dry_run)
    if dry_run:
        print(f"{source} → {target}")
    elif not target.exists() or target.read_bytes() != raw:
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        part.write_bytes(raw)
        part.replace(target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", help="Local bundle directory or HTTP(S) base URL "
                        "(Hugging Face: https://huggingface.co/stases/orbitol/resolve/<revision>)")
    parser.add_argument("--public", default="", metavar="GROUP,...", help="Groups: " + ", ".join(public_inputs()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Replace mismatched destination files")
    args = parser.parse_args(argv)
    if not args.bundle and not args.public:
        parser.error("provide --bundle and/or --public")
    try:
        groups = public_inputs()
        selected = args.public.split(",") if args.public else []
        for group in selected:
            if group not in groups:
                raise ValueError(f"Unknown public group: {group}")
        if args.bundle:
            fetch_bundle(args.bundle, force=args.force, dry_run=args.dry_run)
        for group in selected:
            for item in groups[group]:
                fetch_public(item, force=args.force, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError, C.LocalConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
