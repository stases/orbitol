#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# Usage: slurm/submit slurm/release_check.sbatch --bundle DIR --mp20-pool DIR --table12-samples FILE --full-checkpoint FILE [--commit REV] [--report-dir DIR]
# DRY_RUN=1 prints commands without creating files or submitting work.
# Work dir: node-local RELEASE_WORK_ROOT, else TMPDIR (60 GiB free needed).
set -euo pipefail
release_source=${ORBITOL_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}
release_work=
trap 'if [[ -n $release_work ]]; then rm -rf -- "$release_work"; fi' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ ${DRY_RUN:-0} != 1 && ${1:-} != --help && ${1:-} != -h ]]; then
    work_root=${RELEASE_WORK_ROOT:-${TMPDIR:-}}
    : "${work_root:?set RELEASE_WORK_ROOT or TMPDIR to node-local storage}"
    mkdir -p "$work_root"
    release_work=$(mktemp -d "$work_root/orbitol-release.XXXXXXXX")
fi
# The bootstrap interpreter uses only the standard library. Every project command
# uses the cloned checkout's locked environment; -I excludes inherited Python paths.
python3 -I - "$release_source" "$release_work" "$@" <<'PY'
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

source, work_arg = sys.argv[1:3]
p = argparse.ArgumentParser(description="Fresh-clone release reproduction on one Slurm GPU")
for option in ("bundle", "mp20-pool", "table12-samples", "full-checkpoint"):
    p.add_argument("--" + option, type=Path, required=True)
p.add_argument("--commit", default="HEAD")
p.add_argument("--report-dir", type=Path)
a = p.parse_args(sys.argv[3:])
source = Path(source).resolve()
commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", a.commit + "^{commit}"], universal_newlines=True).strip()
inputs = {name: str(getattr(a, name).resolve()) for name in
          ("bundle", "mp20_pool", "table12_samples", "full_checkpoint")}
dry = os.environ.get("DRY_RUN") == "1"
job = os.environ.get("SLURM_JOB_ID", "manual")
report_dir = a.report_dir
if report_dir is None:
    root = os.environ.get("ORBITOL_RUNS_ROOT")
    if not root and not dry:
        p.error("--report-dir or the submitting wrapper's ORBITOL_RUNS_ROOT is required")
    report_dir = Path(root or "<captured-ORBITOL_RUNS_ROOT>") / "release_check" / job
report_dir = report_dir.absolute()
toolchain = {key: os.environ.get(key, "") for key in
             ("ORBITOL_GCC_ENABLE", "ORBITOL_CUDA_HOME", "ORBITOL_TORCH_CUDA_ARCH_LIST")}
uv = shutil.which("uv")
work = Path(work_arg) if work_arg else Path(os.environ.get("TMPDIR", "/tmp")) / "orbitol-release.DRY_RUN"
repo, data, runs = work / "repo", work / "data", work / "runs"
python = repo / ".venv/bin/python"
local_file = repo / "orbitol.local.env"
# Strip all inherited project settings, including those loaded by slurm/submit.
for key in list(os.environ):
    if key.startswith(("ORBITOL_", "WANDB_", "UV_", "PYTHON", "VIRTUAL_ENV", "CONDA_", "GIT_")):
        os.environ.pop(key)
for key in ("CUDA_HOME", "CUDA_PATH", "TORCH_CUDA_ARCH_LIST", "LD_LIBRARY_PATH", "LD_PRELOAD", "CC", "CXX"):
    os.environ.pop(key, None)
os.environ.update(
    PATH=os.pathsep.join([str(python.parent), str(Path(toolchain["ORBITOL_CUDA_HOME"]) / "bin")
                         if toolchain["ORBITOL_CUDA_HOME"] else "/usr/local/bin", os.defpath]),
    ORBITOL_LOCAL_ENV=str(local_file), ORBITOL_NUMBA="0", WANDB_MODE="disabled",
    PYTHONPATH=str(repo), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
    UV=uv or "uv", UV_PROJECT_ENVIRONMENT=str(repo / ".venv"), UV_CACHE_DIR=str(work / "uv-cache"),
    UV_PYTHON_INSTALL_DIR=str(work / "python"), UV_NO_CONFIG="1",
    XDG_CACHE_HOME=str(work / "cache"), XDG_CONFIG_HOME=str(work / "config"),
    XDG_DATA_HOME=str(work / "share"), TORCH_HOME=str(work / "cache/torch"),
    TORCH_EXTENSIONS_DIR=str(work / "cache/extensions"), TORCHINDUCTOR_CACHE_DIR=str(work / "cache/inductor"),
    TRITON_CACHE_DIR=str(work / "cache/triton"), NUMBA_CACHE_DIR=str(work / "cache/numba"),
    CUDA_CACHE_PATH=str(work / "cache/cuda"), MPLCONFIGDIR=str(work / "cache/matplotlib"),
    GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
    NEQUIP_MULTIPROCESSING_SHARING_STRATEGY="file_descriptor",
)
if toolchain["ORBITOL_CUDA_HOME"]:
    os.environ["CUDA_HOME"] = toolchain["ORBITOL_CUDA_HOME"]
if toolchain["ORBITOL_TORCH_CUDA_ARCH_LIST"]:
    os.environ["TORCH_CUDA_ARCH_LIST"] = toolchain["ORBITOL_TORCH_CUDA_ARCH_LIST"]

report = dict(schema_version=1, status="running", commit=commit, source=str(source), inputs=inputs,
              job_id=job, work_dir=str(work), toolchain=toolchain, steps={},
              notes=["ckpt-every=0 disables periodic checkpoints; train.py still saves at exit.",
                     "Isolation is by fresh configuration and caches, not an OS filesystem sandbox."])
if not dry:
    report_dir.mkdir(parents=True, exist_ok=False)


def save():
    if not dry:
        target = report_dir / "report.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(target)


def command(argv, log):
    argv = [str(x) for x in argv]
    print("+ " + " ".join(shlex.quote(arg) for arg in argv), file=log, flush=True)
    if dry:
        return
    # Only the explicitly captured host compiler setup is sourced.
    wrapper = ['bash', '-c', 'set -e; if [[ -n $1 ]]; then source "$1"; fi; shift; '
               'export CC=$(command -v gcc) CXX=$(command -v g++); exec "$@"',
               'release-toolchain', toolchain["ORBITOL_GCC_ENABLE"]]
    subprocess.run(wrapper + argv, cwd=repo if repo.exists() else work,
                   stdout=log, stderr=subprocess.STDOUT, check=True)


def run(name, action, needs=(), numbers=None):
    entry = dict(status="running", seconds=0, key_numbers={}, log=str(report_dir / (name + ".log")))
    report["steps"][name] = entry
    failed = [n for n in needs if report["steps"][n]["status"] != "passed"]
    if failed and not dry:
        entry.update(status="skipped", reason="Failed dependencies: " + ", ".join(failed))
        Path(entry["log"]).write_text(entry["reason"] + "\n")
        save()
        return
    started = time.monotonic()
    save()
    log = sys.stdout if dry else open(entry["log"], "w")
    print(f"[{name}]", flush=True)
    try:
        action(log)
        if numbers and not dry:
            entry["key_numbers"] = numbers()
        entry["status"] = "passed"
    except Exception as exc:
        entry.update(status="failed", error=str(exc))
        if isinstance(exc, subprocess.CalledProcessError):
            entry["exit_code"] = exc.returncode
        traceback.print_exc(file=log)
    finally:
        entry["seconds"] = round(time.monotonic() - started, 3)
        if not dry:
            log.close()
            # Preserve verifier JSON (including the first difference) on failure too.
            text = Path(entry["log"]).read_text()
            for offset, char in enumerate(text):
                if char == "{" and (offset == 0 or text[offset - 1] == "\n"):
                    try:
                        summary, _ = json.JSONDecoder().raw_decode(text[offset:])
                        entry["summary"] = summary
                    except ValueError:
                        pass
        save()


def invoke(name, args, needs, numbers=None):
    run(name, lambda log: command([python, "-m", *args], log), needs, numbers)


def read_json(path):
    return json.loads(Path(path).read_text())


def metrics(path):
    obj = read_json(path)
    values = obj.get("logged", obj)
    result = {"MSUN": values.get("eval/MSUN", values.get("sample_ema/MSUN")),
              "UN": values.get("eval/un_rate", values.get("sample_ema/un_rate")),
              "one_shot": values.get("eval/MSUN_oneshot", values.get("sample_ema/MSUN_oneshot"))}
    for key in ("MSUN", "UN"):
        if not isinstance(result[key], (float, int)) or not math.isfinite(result[key]):
            raise ValueError(f"Missing or non-finite {key} in {path}")
    return result


def preflight(log):
    print(f"Report: {report_dir}\nWork: {work}\nCommit: {commit}", file=log)
    print("Fresh local file: " + str(local_file), file=log)
    print("Keep 20 GiB free after an estimated 40 GiB for environment/data/caches/results.", file=log)
    if dry:
        return
    free = shutil.disk_usage(work).free
    report["free_bytes_before"] = free
    if free < 60 * 1024**3:
        raise RuntimeError(f"Need 60 GiB free on node-local storage; have {free} bytes")
    if uv is None:
        raise RuntimeError("uv is required on the machine PATH")
    for folder in (data, runs, work / "cache", work / "config", work / "share"):
        folder.mkdir(parents=True, exist_ok=True)


def setup(log):
    command(["git", "clone", "--no-local", "--no-checkout", source, repo], log)
    command(["git", "-C", repo, "checkout", "--detach", commit], log)
    local_values = dict(ORBITOL_DATA_HOME=str(data), ORBITOL_RUNS_ROOT=str(runs), **toolchain)
    if not dry:
        for value in local_values.values():
            if any(c.isspace() or c in "'\"`$\\" for c in value):
                raise ValueError("Local-file values must be literal paths without whitespace or shell quoting")
        local_file.write_text("".join(f"{k}={v}\n" for k, v in local_values.items()))
    command(["bash", repo / "tools/setup_env.sh"], log)
    command([uv or "uv", "--version"], log)
    command([python, "--version"], log)
    command([python, "-c", "import orbitol, pathlib; "
             "assert pathlib.Path(orbitol.__file__).resolve().parent.parent == pathlib.Path.cwd(); "
             "print('Imported:', orbitol.__file__)"], log)
    if not dry:
        report["versions"] = {
            "uv": subprocess.check_output([uv, "--version"], universal_newlines=True).strip(),
            "python": subprocess.check_output([python, "--version"], universal_newlines=True).strip()}


run("00_isolation", preflight)
run("01_environment", setup, ("00_isolation",))
invoke("02_fetch", ["tools.release.fetch", "--bundle", inputs["bundle"], "--public", "mp20-raw,mp20-hull"],
       ("01_environment",))
for item in ("data", "checkpoints", "nequip"):
    invoke("02_verify_" + item, ["tools.release.verify", item], ("02_fetch",))

# The fetch interface owns checkpoint placement. Discover only staged files,
# by their registry suffix, and reject ambiguity or symlinks escaping the work dir.
snapshots = {"mp20": work / "<fetch-destination>/checkpoints/mp20_x_postper_2500k/step_1500000.pt"}
samplers = {"mp20": '{"sample_num_steps":25,"S_churn":10.0,"sigma_min":0.01,"sigma_max":3.0,"decode_impl":"reference"}'}


def locate(log):
    if dry:
        print("Resolve the staged checkpoint and sampler config from the cloned paper registry.", file=log)
        return
    models = read_json(repo / "configs/paper/models.json")
    if len(models) != 1 or {m["dataset"] for m in models} != set(snapshots):
        raise ValueError("Expected exactly the MP-20 paper model")
    for model in models:
        suffix = Path(model["snapshot"])
        candidates = [f for f in work.rglob(suffix.name) if f.parts[-len(suffix.parts):] == suffix.parts]
        if len(candidates) != 1 or work.resolve() not in candidates[0].resolve().parents:
            raise ValueError(f"Expected one staged {suffix}, found {candidates}")
        snapshots[model["dataset"]] = candidates[0]
        samplers[model["dataset"]] = json.dumps(read_json(repo / model["sampler"]))
        print(f"{model['dataset']}: {candidates[0]}", file=log)


run("02_locate_snapshots", locate, ("02_fetch",))
ready = ("02_locate_snapshots", "02_verify_data", "02_verify_checkpoints", "02_verify_nequip")


def sample(name, dataset, count, seed, snapshot=None):
    out = runs / name
    invoke(name, ["orbitol.posthoc_eval", "--kind", "orbitol", "--stage", "sample", "--snapshot",
                  snapshot or snapshots[dataset], "--out", out, "--sample-count", str(count), "--seed", str(seed),
                  "--sampler-overrides", samplers[dataset]], ready,
           lambda: {k: read_json(out / "pool.json")[k] for k in ("n", "seed", "sample_seconds")})
    return out


mp20 = sample("03_mp20_sample", "mp20", 2500, 3001)
# Release checks on one RTX 6000 Ada reproduced each stored pool bitwise or with one differing
# attempt. Allow at most 0.1% under the verifier's compatibility rules (verify.differing_attempts).
invoke("03_mp20_compare", ["tools.release.verify", "pools", mp20, inputs["mp20_pool"], "--max-differing", "2"],
       ("03_mp20_sample",))
invoke("03_mp20_evaluate", ["orbitol.posthoc_eval", "--kind", "orbitol", "--stage", "evaluate", "--out", mp20],
       ("03_mp20_sample",), lambda: metrics(mp20 / "metrics.json"))
ema = sample("05_ema_sample", "mp20", 512, 3001)
# posthoc_eval requires the step_NNNNNNN.pt basename; copy the explicitly supplied
# source to a node-local canonical name, never alter or link the original checkpoint.
full = work / "full/step_1500000.pt"


def stage_full(log):
    print(f"Copy {inputs['full_checkpoint']} -> {full}", file=log)
    if not dry:
        full.parent.mkdir()
        shutil.copyfile(inputs["full_checkpoint"], full)


run("05_stage_full", stage_full, ("00_isolation",))
full_out = runs / "05_full_sample"
invoke("05_full_sample", ["orbitol.posthoc_eval", "--kind", "orbitol", "--stage", "sample", "--snapshot", full,
                          "--out", full_out, "--sample-count", "512", "--seed", "3001",
                          "--sampler-overrides", samplers["mp20"]], (*ready, "05_stage_full"),
       lambda: {k: read_json(full_out / "pool.json")[k] for k in ("n", "seed", "sample_seconds")})
invoke("05_ema_full_compare", ["tools.release.verify", "pools", ema, full_out], ("05_ema_sample", "05_full_sample"))
table = runs / "table12"
chosen = '{"name":"chosen_seed2026","sample_num_steps":25,"S_churn":10,"sigma_min":0.01,"sigma_max":3.0}'
invoke("06_table12", ["tools.sampler_sweep", "--checkpoint", snapshots["mp20"], "--out", table,
                      "--samples", "10000", "--seed", "2026", "--settings", chosen], ready,
       lambda: metrics(table / "chosen_seed2026.json"))
invoke("06_table12_metrics", ["tools.release.verify", "table12", table, "--partial"], ("06_table12",))
invoke("06_table12_pool", ["tools.release.verify", "pools", table / "chosen_seed2026.samples.pt",
                           inputs["table12_samples"], "--max-differing", "10"], ("06_table12",))
def training_numbers():
    records = [json.loads(line) for line in (runs / "training_smoke/metrics.jsonl").read_text().splitlines()]
    final = records[-1]
    if final["step"] != 200:
        raise ValueError(f"Training stopped at step {final['step']}, expected 200")
    return {"completed_steps": final["step"], "loss": final.get("train/loss_total")}


invoke("07_training_smoke", ["orbitol.train", "--config", repo / "configs/paper/mp20_x_postper_2500k.json",
                            "--run-name", "training_smoke", "--out-root", runs, "--max-steps", "200",
                            "--sample-count", "0", "--val-every", "0", "--ckpt-every", "0", "--no-wandb"],
       ("01_environment", "02_verify_data"), training_numbers)
report["status"] = "passed" if all(s["status"] == "passed" for s in report["steps"].values()) else "failed"
save()
print("Dry-run plan complete." if dry else f"{report['status']}: {report_dir / 'report.json'}")
sys.exit(0 if dry or report["status"] == "passed" else 1)
PY
