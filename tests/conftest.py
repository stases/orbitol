import os
import sys

import pytest

# Deterministic CUDA tests need a fixed cuBLAS workspace, set before CUDA initializes.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import orbitol.core  # noqa: E402,F401  binds the pinned Crystalite core before any `src` import
from orbitol import contracts as C  # noqa: E402

# Path-only tests work without local data; data tests skip explicitly.
LOCAL_DATA = C.local_value("ORBITOL_DATA_HOME") is not None
for key in C.REQUIRED:
    if C.local_value(key) is None:
        os.environ[key] = f"/nonexistent/orbitol-tests/{key.lower()}"


@pytest.fixture(scope="session")
def frozen_tables():
    from orbitol.tables import load_tables
    from pathlib import Path
    if not Path(C.TABLES_DIR, "manifest.json").is_file():
        pytest.skip(f"Frozen tables unavailable: {C.TABLES_DIR}; set ORBITOL_DATA_HOME in orbitol.local.env")
    return load_tables(C.TABLES_DIR)


@pytest.fixture(autouse=True)
def _no_slurm_submission(tmp_path_factory, monkeypatch):
    """Async evaluation is the default; a test that reaches the real sbatch gets a failing stub instead."""
    stub = tmp_path_factory.mktemp("no_sbatch")
    (stub / "sbatch").write_text("#!/bin/sh\necho 'tests must not submit Slurm jobs' >&2\nexit 1\n")
    (stub / "sbatch").chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub}{os.pathsep}{os.environ.get('PATH', '')}")


@pytest.fixture(autouse=True)
def _deterministic_cpu():
    import torch
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(det)


def pytest_collection_modifyitems(config, items):
    import torch
    if not LOCAL_DATA:
        skip_data = pytest.mark.skip(reason="needs local data: set ORBITOL_DATA_HOME in orbitol.local.env")
        for item in items:
            if "data" in item.keywords:
                item.add_marker(skip_data)
    if not torch.cuda.is_available():
        skip = pytest.mark.skip(reason="needs CUDA")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip)
