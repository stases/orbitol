"""Durable pool I/O, shared with the torch-free queue planner."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

SNAPSHOT_NAME = re.compile(r"^step_(\d{7})\.pt$")
POOL_NAME = re.compile(r"^step_(\d{7})$")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path):
    info = Path(path).lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Expected a regular, non-symlink file: {path}")
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns, inode=info.st_ino)


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def atomic_writer(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        fsync_dir(path.parent)
    finally:
        Path(name).unlink(missing_ok=True)


def write_json(path, payload):
    with atomic_writer(path) as stream:
        stream.write((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def verify_pool(directory, *, hashes=True):
    directory = Path(directory)
    pool = read_json(directory / "pool.json")
    if not pool or pool.get("status") != "complete":
        raise ValueError(f"Incomplete pool: {directory}")
    if pool.get("kind") not in {"crystalite", "orbitol"}:
        raise ValueError(f"Invalid pool kind: {directory}")
    identity(directory / "samples.pt")
    if hashes and sha256(directory / "samples.pt") != pool["samples_sha256"]:
        raise ValueError(f"Sample hash mismatch: {directory}")
    return pool
