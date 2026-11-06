"""Shell configuration and submission without contacting a scheduler."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import pytest

from orbitol import contracts as C

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_env(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("ORBITOL_", "SBATCH_")):
            monkeypatch.delenv(key)
    path = tmp_path / "local.env"
    monkeypatch.setenv("ORBITOL_LOCAL_ENV", str(path))
    return path


def test_bash_python_reader_parity(local_env, monkeypatch):
    values = {
        "ORBITOL_DATA_HOME": "/sample/data", "ORBITOL_RUNS_ROOT": "/sample/runs",
        "ORBITOL_CORE_ROOT": "/sample/core", "ORBITOL_WANDB_ENTITY": "test-entity",
        "ORBITOL_PYTHON": "/sample/python", "ORBITOL_BASE_PYTHON": "/sample/base/python",
        "ORBITOL_GCC_ENABLE": "/sample/gcc/enable", "ORBITOL_CUDA_HOME": "/sample/cuda",
        "ORBITOL_TORCH_CUDA_ARCH_LIST": "8.9", "ORBITOL_SLURM_GPU_PARTITION": "gpu-pool",
        "ORBITOL_SLURM_GPU_ACCOUNT": "gpu-account", "ORBITOL_SLURM_CPU_PARTITION": "",
        "ORBITOL_SLURM_CPU_ACCOUNT": "", "ORBITOL_SLURM_LOG_DIR": "/logs/%x",
    }
    local_env.write_text("# comment\n\nORBITOL_DATA_HOME=/superseded\n" + "\n".join(
        f"  export {key}={value}  " for key, value in values.items()))  # no final newline
    monkeypatch.setenv("ORBITOL_RUNS_ROOT", "/environment/runs")
    monkeypatch.setenv("ORBITOL_DATA_HOME", "")  # Empty environment falls back to the file.
    output = subprocess.check_output(
        ["bash", "-eu", "-c", 'source "$1/slurm/common.sh"; env -0', "bash", str(ROOT)])
    loaded = dict(item.split("=", 1) for item in output.decode().split("\0") if item)
    for key in values:
        assert (loaded[key] or None) == C.local_value(key)
    assert loaded["WANDB_ENTITY"] == values["ORBITOL_WANDB_ENTITY"]
    assert loaded["ORBITOL_REPO_ROOT"] == str(ROOT)


@pytest.mark.parametrize("bad", ["bad line", 'ORBITOL_DATA_HOME="$HOME/data"',
                                  "ORBITOL_DATA_HOME=/with space", "ORBITOL_DATA_HOME=`pwd`"])
def test_loader_rejects_nonliteral_file(local_env, bad):
    local_env.write_text("# comment\n" + bad + "\n")
    result = subprocess.run(["bash", "-eu", "-c", 'source "$1/slurm/common.sh"', "bash", str(ROOT)],
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert f"{local_env}:2: expected KEY=value" in result.stderr


def test_defaults_without_file(local_env, monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    output = subprocess.check_output(["bash", "-eu", "-c",
        'source "$1/slurm/common.sh"; printf "%s\\n" "$ORBITOL_PYTHON" "$ORBITOL_BASE_PYTHON" "${WANDB_ENTITY-unset}"',
        "bash", str(ROOT)], text=True)
    assert output.splitlines() == [str(ROOT / ".venv/bin/python")] * 2 + ["unset"]


@pytest.fixture
def fake_sbatch(tmp_path, local_env, monkeypatch):
    binary = tmp_path / "sbatch"
    binary.write_text(f"#!{sys.executable}\n" + '''import json, os, sys
print(json.dumps(dict(argv=sys.argv[1:], env={key: value for key, value in os.environ.items()
    if key.startswith("SBATCH_") or key == "ORBITOL_REPO_ROOT"})))
''')
    binary.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    local_env.write_text("ORBITOL_SLURM_GPU_PARTITION=gpu-pool\n"
                         "ORBITOL_SLURM_GPU_ACCOUNT=gpu-account\n"
                         "ORBITOL_SLURM_LOG_DIR=/sample/logs\n")

    def run(*args, cwd=tmp_path):
        result = subprocess.run([str(ROOT / "slurm/submit"), *map(str, args)], cwd=cwd,
                                check=True, capture_output=True, text=True)
        return json.loads(result.stdout)
    return run


@pytest.mark.parametrize("script,gpu", [("train_orbitol", True), ("eval_gpu", True),
                                       ("cpu", False), ("eval_score", False)])
def test_submit_class_and_inherited_values(fake_sbatch, monkeypatch, script, gpu):
    for key in ("SBATCH_PARTITION", "SBATCH_ACCOUNT", "SBATCH_GRES", "SBATCH_OUTPUT"):
        monkeypatch.setenv(key, "inherited")
    monkeypatch.setenv("ORBITOL_REPO_ROOT", "/wrong/checkout")
    args = ["--parsable", str(ROOT / f"slurm/{script}.sbatch"), "run with spaces", "--help", ""]
    result = fake_sbatch(*args)
    assert result["argv"] == args
    env = result["env"]
    assert env["ORBITOL_REPO_ROOT"] == str(ROOT)
    assert env["SBATCH_OUTPUT"] == "/sample/logs/%x-%j.out"
    if gpu:
        assert env["SBATCH_PARTITION"] == "gpu-pool"
        assert env["SBATCH_ACCOUNT"] == "gpu-account"
    else:
        assert not {"SBATCH_PARTITION", "SBATCH_ACCOUNT", "SBATCH_GRES"} & env.keys()


def test_cpu_placement_can_be_configured(fake_sbatch, local_env):
    with local_env.open("a") as stream:
        stream.write("ORBITOL_SLURM_CPU_PARTITION=cpu-pool\nORBITOL_SLURM_CPU_ACCOUNT=cpu-account\n")
    env = fake_sbatch(ROOT / "slurm/cpu.sbatch")["env"]
    assert env["SBATCH_PARTITION"] == "cpu-pool"
    assert env["SBATCH_ACCOUNT"] == "cpu-account"


def test_unset_placement_and_logs_use_cluster_defaults(fake_sbatch, local_env, monkeypatch):
    local_env.write_text("")
    for key in ("SBATCH_PARTITION", "SBATCH_ACCOUNT", "SBATCH_OUTPUT"):
        monkeypatch.setenv(key, "inherited")
    env = fake_sbatch(ROOT / "slurm/gpu.sbatch")["env"]
    assert not {"SBATCH_PARTITION", "SBATCH_ACCOUNT", "SBATCH_OUTPUT"} & env.keys()


@pytest.mark.parametrize("options", [
    ["--test-only", "--partition=explicit", "--account=explicit-account", "--output=explicit.out"],
    ["-p", "explicit", "-A", "explicit-account", "-o", "explicit.out", "--job-name", "job name"],
    ["--job-name", "name.sbatch", "--output", "log.sbatch", "--chdir", "/sample/checkout"],
    ["--nice", "--propagate", "--exclusive", "-H"],
])
def test_explicit_options_reach_sbatch_unchanged(fake_sbatch, options):
    args = [*options, "slurm/train_orbitol.sbatch", "x", "--partition=script-argument"]
    result = fake_sbatch(*args, cwd=ROOT)
    # Slurm gives these unchanged CLI options precedence over the wrapper's environment.
    assert result["argv"] == args
    assert result["env"]["SBATCH_PARTITION"] == "gpu-pool"
    assert result["env"]["SBATCH_ACCOUNT"] == "gpu-account"


def test_wrapper_does_not_change_caller_environment(fake_sbatch):
    command = '"$1/slurm/submit" "$1/slurm/cpu.sbatch" >/dev/null; printf "%s" "$SBATCH_PARTITION"'
    result = subprocess.check_output(["bash", "-eu", "-c", command, "bash", str(ROOT)],
                                     env={**os.environ, "SBATCH_PARTITION": "parent"}, text=True)
    assert result == "parent"


def test_help_and_missing_script(local_env):
    for args, code in [(["--help"], 0), ([], 2), (["missing.sbatch"], 2)]:
        result = subprocess.run([str(ROOT / "slurm/submit"), *args], capture_output=True, text=True)
        assert result.returncode == code
        assert "Usage: slurm/submit" in result.stdout + result.stderr


@pytest.mark.parametrize("args,reference", [
    ([], None), (["--reference-oeq"], "/sample/reference.pth"),
    (["--reference-oeq", "--seed", "7"], "/sample/reference.pth"),
    (["--reference-oeq", "/explicit/ref.pth"], "/explicit/ref.pth"),
    (["--reference-oeq=/explicit/ref.pth"], "/explicit/ref.pth"),
])
def test_build_reference_is_opt_in(tmp_path, local_env, monkeypatch, args, reference):
    binary = tmp_path / "python"
    binary.write_text(f"#!{sys.executable}\n" + '''import json, sys
if sys.argv[1:3] == ["-m", "orbitol.contracts"]:
    print("/sample/reference.pth")
else:
    print(json.dumps(sys.argv[1:]))
''')
    binary.chmod(0o755)
    ninja = tmp_path / "ninja"
    ninja.write_text("#!/bin/sh\nexit 0\n")
    ninja.chmod(0o755)
    local_env.write_text(f"ORBITOL_PYTHON={binary}\nORBITOL_DATA_HOME={tmp_path}/data\n")
    monkeypatch.setenv("ORBITOL_REPO_ROOT", str(ROOT))
    result = subprocess.run(["bash", str(ROOT / "slurm/build_nequip.sbatch"), *args],
                            check=True, capture_output=True, text=True)
    argv = json.loads(result.stdout)
    expected = ["-m", "tools.build_nequip", *args]
    if args and args[0] == "--reference-oeq" and (len(args) == 1 or args[1].startswith("--")):
        expected.insert(3, reference)
    assert argv == expected
    if reference is None:
        assert "--reference-oeq" not in argv


def test_wrapper_uses_its_own_checkout_from_another_directory(fake_sbatch, tmp_path, monkeypatch):
    checkout = tmp_path / "another checkout"
    (checkout / "slurm").mkdir(parents=True)
    for name in ("submit", "common.sh", "gpu.sbatch"):
        shutil.copy2(ROOT / "slurm" / name, checkout / "slurm" / name)
    (checkout / "orbitol.local.env").write_text("ORBITOL_SLURM_GPU_PARTITION=checkout-pool\n")
    monkeypatch.delenv("ORBITOL_LOCAL_ENV")
    monkeypatch.setenv("ORBITOL_REPO_ROOT", str(ROOT))
    result = subprocess.run([str(checkout / "slurm/submit"), str(checkout / "slurm/gpu.sbatch"), "arg"],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    loaded = json.loads(result.stdout)
    assert loaded["env"]["ORBITOL_REPO_ROOT"] == str(checkout)
    assert loaded["env"]["SBATCH_PARTITION"] == "checkout-pool"
