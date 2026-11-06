"""Bind the pinned Crystalite core (commit 03603fe) and verify it.

Import this module before anything that imports ``src``. The core is never copied or edited;
Orbitol extends it only by subclassing and by composing its functions.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from orbitol import contracts as C

DEFAULT_CORE_ROOT = str(C.REPO_ROOT / C.REPO_DEFAULTS["ORBITOL_CORE_ROOT"])
CORE_ROOT = os.path.abspath(C.local_path("ORBITOL_CORE_ROOT"))
_MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core_manifest.json")


def _bind() -> None:
    if not os.path.isfile(os.path.join(CORE_ROOT, "src", "__init__.py")):
        raise RuntimeError(f"ORBITOL_CORE_ROOT={CORE_ROOT} has no src/__init__.py")
    loaded = sys.modules.get("src")
    if loaded is not None:
        where = os.path.dirname(os.path.dirname(os.path.abspath(getattr(loaded, "__file__", "") or "")))
        if os.path.abspath(where) != CORE_ROOT:
            raise RuntimeError(f"a foreign 'src' package is already imported from {where}")
    if CORE_ROOT not in sys.path:
        sys.path.insert(0, CORE_ROOT)


def verify_core() -> None:
    """Raise if any file of the bound core differs from the pinned manifest."""
    manifest = json.load(open(_MANIFEST))["files"]
    bad = []
    for rel, sha in manifest.items():
        p = os.path.join(CORE_ROOT, rel)
        got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else "missing"
        if got != sha:
            bad.append(f"{rel}: expected {sha[:12]}, found {got[:12]}")
    if bad:
        raise RuntimeError("Crystalite core does not match the pin:\n" + "\n".join(bad))


_bind()

if __name__ == "__main__":
    verify_core()
    print(f"core OK: {CORE_ROOT}")
