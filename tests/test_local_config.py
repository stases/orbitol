"""Local configuration grammar, precedence, lazy imports and shell interface."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

from orbitol import contracts as C


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith("ORBITOL_"):
            monkeypatch.delenv(key)
    path = tmp_path / "local.env"
    monkeypatch.setenv("ORBITOL_LOCAL_ENV", str(path))
    monkeypatch.setattr(C, "DATASETS", C._DatasetRegistry())
    return path


def test_literals_and_environment_precedence(local_env, monkeypatch):
    local_env.write_text("# comment\n\nexport ORBITOL_DATA_HOME=/data\nORBITOL_RUNS_ROOT=/runs\n"
                         "export SBATCH_OUTPUT=/logs/%x-%j.out\nORBITOL_WANDB_ENTITY=\n")
    assert C.local_path("ORBITOL_DATA_HOME") == "/data"
    assert C.local_value("SBATCH_OUTPUT") == "/logs/%x-%j.out"
    assert C.local_value("ORBITOL_WANDB_ENTITY") is None
    monkeypatch.setenv("ORBITOL_DATA_HOME", "/override")
    assert C.local_path("ORBITOL_DATA_HOME") == "/override"
    monkeypatch.setenv("ORBITOL_DATA_HOME", "")
    assert C.local_path("ORBITOL_DATA_HOME") == "/data"


@pytest.mark.parametrize("line", ["bad line", "export ORBITOL_DATA_HOME='/data'",
                                  'ORBITOL_DATA_HOME="$HOME/data"', "ORBITOL_DATA_HOME=/with space",
                                  "ORBITOL_DATA_HOME=/back\\slash", "ORBITOL_DATA_HOME=`pwd`"])
def test_bad_line_names_file_and_line(local_env, line):
    local_env.write_text("# comment\n" + line + "\n")
    with pytest.raises(C.LocalConfigError, match="expected KEY=value with a literal value") as error:
        C.local_value("ORBITOL_DATA_HOME")
    assert f"{local_env}:2:" in str(error.value)


def test_unknown_orbitol_key(local_env):
    local_env.write_text("export ORBITOL_TYPO=/data\n")
    with pytest.raises(C.LocalConfigError, match="unknown key ORBITOL_TYPO"):
        C.local_value("ORBITOL_DATA_HOME")


def test_layout_and_independent_override(local_env, monkeypatch):
    monkeypatch.setenv("ORBITOL_DATA_HOME", "/bundle")
    for key, relative in C.DATA_LAYOUT.items():
        assert C.local_path(key) == str(Path("/bundle") / relative)
    monkeypatch.setenv("ORBITOL_MP20_DATA_ROOT", "/raw-elsewhere")
    spec = C.dataset_spec("mp20")
    assert spec.data_root == "/raw-elsewhere"
    assert spec.orbit_cache_dir == "/bundle/mp20/orbit_cache"
    assert C.DATA_ROOT == spec.data_root
    assert C.TABLES_DIR == "/bundle/tables/v1"
    assert C.NEQUIP_MODEL == C.local_path("ORBITOL_NEQUIP_TS")


def test_missing_key_and_repo_default(local_env):
    assert C.REPO_DEFAULTS == {"ORBITOL_CORE_ROOT": "external/crystalite-03603fe"}
    assert C.local_path("ORBITOL_CORE_ROOT") == str(C.REPO_ROOT / "external/crystalite-03603fe")
    assert C.TrainConfig().dataset == "mp20"
    assert not C.DATASETS
    with pytest.raises(C.LocalConfigError) as error:
        C.local_path("ORBITOL_TABLES_DIR")
    assert "ORBITOL_DATA_HOME (or ORBITOL_TABLES_DIR)" in str(error.value)
    assert str(local_env) in str(error.value)
    with pytest.raises(C.LocalConfigError, match="ORBITOL_RUNS_ROOT"):
        C.local_path("ORBITOL_RUNS_ROOT")
    with pytest.raises(AttributeError):
        C.unknown_attribute


def test_cli_and_lazy_imports(local_env):
    command = [sys.executable, "-m", "orbitol.contracts"]
    result = subprocess.run(command + ["ORBITOL_TABLES_DIR"], text=True, capture_output=True)
    assert result.returncode == 1
    assert "ORBITOL_DATA_HOME" in result.stderr and str(local_env) in result.stderr
    local_env.write_text("ORBITOL_DATA_HOME=/bundle\nORBITOL_RUNS_ROOT=/runs\n")
    result = subprocess.run(command + ["ORBITOL_TABLES_DIR", "ORBITOL_RUNS_ROOT"], text=True, capture_output=True)
    assert result.returncode == 0 and result.stdout == "/bundle/tables/v1\n/runs\n"
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0
    assert "ORBITOL_DATA_HOME=/bundle  # file" in result.stdout
    assert "ORBITOL_TABLES_DIR=/bundle/tables/v1  # derived" in result.stdout
    assert "ORBITOL_WANDB_ENTITY=  # unset" in result.stdout


def test_imports_need_no_data_config(local_env):
    # Binding the core is independent of configuring data and outputs.
    import orbitol.core as core
    env = {**os.environ, "ORBITOL_CORE_ROOT": core.CORE_ROOT}
    result = subprocess.run([sys.executable, "-c", "import orbitol.train, orbitol.eval_adapter, "
                             "orbitol.nequip_relax, orbitol.posthoc_queue; "
                             "from orbitol.contracts import TrainConfig; TrainConfig()"],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
