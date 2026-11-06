"""Shared interfaces for Orbitol.

Every module codes against the names, shapes and dtypes defined here.

Conventions
-----------
* Fractional coordinates are column vectors in the conventional cell of the frozen setting
  (one Hall number per group, pyxtal 1.1.2's default). A space-group operation is x -> R x + t.
* A Wyckoff row w has d = d_w free parameters u in R^d. Its representative chart is
  x = A_w u + b_w (mod 1) and its member maps are x_j = W_{w,j} u + v_{w,j}. Arrays are padded to 3
  free columns: columns d..2 of A_w and W_{w,j} are zero.
* The lattice uses Crystalite's ``ltri`` latent: L = [[e^p0, 0, 0], [p1, e^p2, 0], [p3, p4, e^p5]]
  with rows a, b, c and Gram = L L^T. Families tie entries (FREE_SLOTS below).
* Tokens are unordered. A batch is padded to K tokens; ``pad_mask`` is True on padding.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, get_args

# --------------------------------------------------------------------------------------------
# Local paths: orbitol.local.env (gitignored; template orbitol.local.env.example) or the environment
# --------------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
TYPE_ENCODING_STATE = "orbitol/assets/type_encoding_subatomic_tokenizer_pca_16.pt"  # repo-relative
TYPE_ENCODING_SHA256 = "a1049dde7ccf20da40573574c529d3051f8fbea3315177365a59a8369c888d03"
WANDB_PROJECT = "orbitol"

Dataset = Literal["mp20", "alex_mp20"]
DATASET_NAMES: tuple[str, ...] = get_args(Dataset)
_PATH_FIELDS = ("data_root", "orbit_cache_dir", "atom_cache_dir", "symmetry_reference", "phase_diagram")
_HULLS = {"mp20": "hulls/mp20/2023-02-07-ppd-mp.pkl", "alex_mp20": "hulls/alex_mp2020/variants/v1.pkl"}
# Layout under ORBITOL_DATA_HOME (the release bundle unpacks to it); each entry has its own override key.
DATA_LAYOUT = {
    "ORBITOL_TABLES_DIR": "tables/v1",
    "ORBITOL_NEQUIP_TS": "mlip/nequip_ts/NequIP-OAM-L.nequip.pth",       # TorchScript; sm_89 AOTI builds crash
    "ORBITOL_NEQUIP_OEQ": "mlip/nequip_oeq/NequIP-OAM-L-oeq-ts.nequip.pth",
    **{f"ORBITOL_{name.upper()}_{field.upper()}": path
       for name in DATASET_NAMES
       for field, path in zip(_PATH_FIELDS, (name, f"{name}/orbit_cache", f"{name}/atom_cache",
                                              f"{name}/symmetry_reference.json", _HULLS[name]))},
}
REPO_DEFAULTS = {"ORBITOL_CORE_ROOT": "external/crystalite-03603fe"}  # tools/setup_env.sh clones the pin here
REQUIRED = ("ORBITOL_DATA_HOME", "ORBITOL_RUNS_ROOT")
OPTIONAL = ("ORBITOL_WANDB_ENTITY", "ORBITOL_CUDA_HOME")
SHELL_KEYS = ("ORBITOL_PYTHON", "ORBITOL_BASE_PYTHON", "ORBITOL_GCC_ENABLE", "ORBITOL_TORCH_CUDA_ARCH_LIST",
              "ORBITOL_SLURM_GPU_PARTITION", "ORBITOL_SLURM_GPU_ACCOUNT",
              "ORBITOL_SLURM_CPU_PARTITION", "ORBITOL_SLURM_CPU_ACCOUNT", "ORBITOL_SLURM_LOG_DIR")
_LINE = re.compile(r"(?:export\s+)?([A-Z][A-Z0-9_]*)=([^\s'\"`$\\]*)")


class LocalConfigError(RuntimeError):
    """A machine-specific path is missing or orbitol.local.env is malformed."""


def local_env_file() -> Path:
    return Path(os.environ.get("ORBITOL_LOCAL_ENV") or REPO_ROOT / "orbitol.local.env")


@lru_cache(maxsize=None)
def _read_local_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    known = {*REQUIRED, *DATA_LAYOUT, *REPO_DEFAULTS, *OPTIONAL, *SHELL_KEYS}
    values = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.fullmatch(line)
        if match is None:
            raise LocalConfigError(f"{path}:{number}: expected KEY=value with a literal value "
                                   "(no spaces, quotes, $ or backslashes)")
        if match[1].startswith("ORBITOL_") and match[1] not in known:
            raise LocalConfigError(f"{path}:{number}: unknown key {match[1]} (see orbitol.local.env.example)")
        values[match[1]] = match[2]
    return values


def local_value(key: str) -> str | None:
    """The environment, then orbitol.local.env; empty means unset. No defaults."""
    return os.environ.get(key) or _read_local_env(local_env_file()).get(key) or None


def local_path(key: str) -> str:
    """A configured path: its key, else its ORBITOL_DATA_HOME layout or repo default, else an error."""
    value = local_value(key)
    if value is None and key in REPO_DEFAULTS:
        value = str(REPO_ROOT / REPO_DEFAULTS[key])
    if value is None and key in DATA_LAYOUT and (home := local_value("ORBITOL_DATA_HOME")):
        value = os.path.join(home, DATA_LAYOUT[key])
    if value is None:
        needed = f"ORBITOL_DATA_HOME (or {key})" if key in DATA_LAYOUT else key
        raise LocalConfigError(f"{needed} is not set: add it to {local_env_file()} "
                               "(template: orbitol.local.env.example) or export it")
    return value


@dataclass(frozen=True)
class DatasetSpec:
    """Read-only inputs and separate dataset-derived output paths."""
    name: Dataset
    data_root: str
    csv_format: Literal["mp20", "mattergen"]
    orbit_cache_dir: str
    atom_cache_dir: str
    symmetry_reference: str
    phase_diagram: str
    wandb_project: str  # W&B project for training and post-hoc runs unless one is given


_DATASET_INFO = {"mp20": ("mp20", WANDB_PROJECT), "alex_mp20": ("mattergen", "orbitol_alex_mp20")}


class _DatasetRegistry(dict):
    """DatasetSpec per name, built from the local paths on first lookup; tests may replace entries."""

    def __missing__(self, name):
        if name not in DATASET_NAMES:
            raise KeyError(name)
        csv_format, project = _DATASET_INFO[name]
        self[name] = spec = DatasetSpec(
            name=name, csv_format=csv_format, wandb_project=project,
            **{field: local_path(f"ORBITOL_{name.upper()}_{field.upper()}") for field in _PATH_FIELDS})
        return spec


DATASETS: dict[str, DatasetSpec] = _DatasetRegistry()


def dataset_spec(name: Dataset = "mp20") -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"Unknown dataset {name!r}: choose from {list(DATASET_NAMES)}") from None


def resolve_dataset_paths(cfg=None, *, overrides=None) -> DatasetSpec:
    """Explicit overrides > cfg.extra > dataset spec; old configs mean mp20.

    Accept a TrainConfig, snapshot config mapping, or dataset name. ``cache_dir``
    retains its historical meaning (the cache consumed by training/sampling).
    Overrides are independent: changing data_root does not relocate other paths.
    ``None`` in CLI overrides means the flag was omitted.
    """
    if cfg is None or isinstance(cfg, str):
        cfg = {"dataset": cfg or "mp20"}
    if not isinstance(cfg, Mapping):
        cfg = {"dataset": cfg.dataset, "extra": cfg.extra}
    spec = dataset_spec(cfg.get("dataset", "mp20"))
    names = dict(data_root="data_root", cache_dir="orbit_cache_dir",
                 atom_cache_dir="atom_cache_dir", symmetry_reference="symmetry_reference",
                 phase_diagram="phase_diagram")
    paths = {}
    for source in (cfg.get("extra", {}), overrides or {}):
        paths.update({field: str(source[key]) for key, field in names.items()
                      if key in source and source[key] is not None})
    return replace(spec, **paths)


_ALIASES = {  # Historical names, resolved when used (PEP 562); MP-20 for the dataset aliases.
    "TABLES_DIR": lambda: local_path("ORBITOL_TABLES_DIR"),
    "RUNS_ROOT": lambda: local_path("ORBITOL_RUNS_ROOT"),
    "NEQUIP_MODEL": lambda: local_path("ORBITOL_NEQUIP_TS"),
    "DATA_ROOT": lambda: dataset_spec("mp20").data_root,
    "ORBIT_CACHE_DIR": lambda: dataset_spec("mp20").orbit_cache_dir,
    "SYMMETRY_REFERENCE": lambda: dataset_spec("mp20").symmetry_reference,
    "PHASE_DIAGRAM": lambda: dataset_spec("mp20").phase_diagram,
}


def __getattr__(name):
    if name in _ALIASES:
        return _ALIASES[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------------------------
# Symmetry-table sizes and enumerations
# --------------------------------------------------------------------------------------------
N_GROUPS = 230
N_ROWS = 1731          # real rows over all groups, P1 1a included: ids 1..1731
R_TOTAL = N_ROWS + 1   # row arrays have one extra entry: id 0 is padding and uses the P1 chart
G_TOTAL = N_GROUPS + 1  # group arrays: index g is space group g; index 0 is unused (P1-like)
MAX_MULT = 192         # largest multiplicity (Fm-3m general position)
MAX_OPS = 192          # largest |G / Z^3| in the conventional cell
MAX_VRV = 14           # Voronoi-relevant vectors of a lattice of dimension <= 3
MAX_IMAGES = 192       # chart images (S, s) per row, padded; a general position has |G/Z^3| of them
ROW_DESC_DIM = 19      # Wyckoff-row descriptor width (orbitol.descriptors)
SG_DESC_DIM = 62       # space-group descriptor width (orbitol.descriptors)

FAMILIES = ("triclinic", "monoclinic", "orthorhombic", "tetragonal", "hexagonal", "cubic")
# "hexagonal" includes the trigonal groups (R groups in hexagonal axes, obverse).
CENTRINGS = ("P", "A", "B", "C", "I", "F", "R")
FREE_SLOTS = {  # ltri slots that carry free parameters, per family
    "triclinic": (0, 1, 2, 3, 4, 5),
    "monoclinic": (0, 2, 3, 5),
    "orthorhombic": (0, 2, 5),
    "tetragonal": (0, 5),
    "hexagonal": (0, 5),
    "cubic": (0,),
}

# Pinned row ids: reversed pyxtal Wyckoff order within each group.
ROW_ID_PINS = {("P1", "1a"): 1, ("P-1", "1a"): 2, ("Pmmm", "1a"): 226, ("Pmmm", "8A"): 252}

# --------------------------------------------------------------------------------------------
# Symmetry tables: arrays in TABLES_DIR/tables.npz (built by orbitol.tables)
# name: (shape, dtype, meaning). R = R_TOTAL, G = G_TOTAL.
# --------------------------------------------------------------------------------------------
TABLE_ARRAYS: dict[str, tuple[tuple, str, str]] = {
    # rows
    "row_group": (("R",), "int16", "space group of the row (0 for the pad row)"),
    "row_letter": (("R",), "<U2", "Wyckoff letter as a name, never an ordinal ('' for pad)"),
    "row_mult": (("R",), "int16", "multiplicity m_w in the conventional cell"),
    "row_mprim": (("R",), "int16", "primitive multiplicity m_w / c(G)"),
    "row_dof": (("R",), "int8", "number of free parameters d_w in {0,1,2,3}"),
    "row_A": (("R", 3, 3), "float64", "representative chart matrix, columns >= d zero"),
    "row_b": (("R", 3), "float64", "representative chart offset"),
    "row_W": (("R", "MAX_MULT", 3, 3), "float64", "member maps (padded)"),
    "row_v": (("R", "MAX_MULT", 3), "float64", "member offsets (padded)"),
    "row_member_mask": (("R", "MAX_MULT"), "bool", "True for real members j < m_w"),
    "row_prim_mask": (("R", "MAX_MULT"), "bool", "True for the first member of each centring class"),
    "row_H": (("R", 3, 3), "float64", "chart metric A^T M_G A in the top-left d x d block; identity elsewhere"),
    "row_Hinv": (("R", 3, 3), "float64", "inverse of row_H"),
    "row_Pi": (("R", 3, 3), "float64", "projection H^-1 A^T M_G (rows >= d zero)"),
    "row_N": (("R", 3, 3), "float64", "noise map H^-1 A^T M_G^(1/2) (rows >= d zero); N N^T = H^-1 on the d block"),
    "row_Q": (("R", 6), "float64", "upper triangle of A H^-1 A^T"),
    "row_wrap_fast": (("R",), "bool", "True when Lambda_w = Z^d and H_w is diagonal (wrap = x - round(x))"),
    "row_lambda_basis": (("R", 3, 3), "float64", "LLL-reduced basis of Lambda_w in the H metric, as columns; identity beyond d"),
    "row_vrv": (("R", "MAX_VRV", 3), "float64", "Voronoi-relevant vectors of Lambda_w (padded)"),
    "row_vrv_count": (("R",), "int8", "number of Voronoi-relevant vectors"),
    "row_img_S": (("R", "MAX_IMAGES", 3, 3), "float64", "chart images u -> S u + s (padded)"),
    "row_img_s": (("R", "MAX_IMAGES", 3), "float64", ""),
    "row_img_count": (("R",), "int16", "number of chart images"),
    "row_desc": (("R", "ROW_DESC_DIM"), "float32", "z-scored row descriptor"),
    # groups
    "group_hall": (("G",), "int16", "frozen Hall number (pyxtal 1.1.2 default)"),
    "group_symbol": (("G",), "<U12", "Hermann-Mauguin symbol"),
    "group_q": (("G", 3), "float64", "origin shift: x_spglib = x_frozen + q (mod 1)"),
    "group_family": (("G",), "int8", "index into FAMILIES"),
    "group_centring": (("G",), "int8", "index into CENTRINGS"),
    "group_ncent": (("G",), "int8", "centring factor c(G)"),
    "group_cent_vecs": (("G", 4, 3), "float64", "centring translations (padded)"),
    "group_Pc": (("G", 3, 3), "float64", "standard primitive basis of the centring (columns), conventional coordinates"),
    "group_MG": (("G", 3, 3), "float64", "point-group metric M_G"),
    "group_ops_R": (("G", "MAX_OPS", 3, 3), "int8", "rotation parts of G / Z^3, centring included (padded)"),
    "group_ops_t": (("G", "MAX_OPS", 3), "float64", "translation parts"),
    "group_ops_count": (("G",), "int16", ""),
    "group_row_start": (("G",), "int16", "first global row id of the group; rows are contiguous"),
    "group_row_count": (("G",), "int16", ""),
    "group_polar": (("G", 3), "bool", "polar (free origin) axes"),
    "group_desc": (("G", "SG_DESC_DIM"), "float32", "z-scored space-group descriptor"),
}
# Normaliser cosets live in TABLES_DIR/cosets.npz with a documented layout
# chosen by orbitol.tables; consumers use orbitol.tables.Cosets only.

# --------------------------------------------------------------------------------------------
# Batches (orbitol.data -> model, objective, sampler)
# --------------------------------------------------------------------------------------------
ORBIT_BATCH_KEYS: dict[str, tuple[tuple, str, str]] = {
    "mp_id": (("B",), "list[str]", ""),
    "G": (("B",), "int64", "space group 1..230 (P1 structures are G = 1)"),
    "family": (("B",), "int64", "index into FAMILIES"),
    "row": (("B", "K"), "int64", "global row id; 0 = pad"),
    "A0": (("B", "K"), "int64", "atomic number per orbit; 0 = pad"),
    "U": (("B", "K", 3), "float32", "chart coordinates in [0, 1); zero beyond d"),
    "dof_mask": (("B", "K", 3), "bool", "True on the d free coordinates of real tokens"),
    "pad_mask": (("B", "K"), "bool", "True = padding"),
    "lat": (("B", 6), "float32", "clean ltri latent, dependent entries consistent with the family"),
    "n_prim": (("B",), "int64", "primitive atom count"),
}


@dataclass(frozen=True)
class Skeleton:
    """One requested symmetric structure: a group and an unordered multiset of rows."""
    G: int
    rows: tuple[int, ...]            # global row ids; a d = 0 row appears at most once
    A0: tuple[int, ...] | None = None  # optional fixed atomic numbers, aligned with rows


# --------------------------------------------------------------------------------------------
# Model call (orbitol.model.OrbitolModel)
# --------------------------------------------------------------------------------------------
# raw = model(type_in, x_in, lat_in, pad_mask, c_noise, *, row, G, u=None, z=None, family=None)
#   type_in (B,K,16) float32   c_in-scaled noisy type features
#   x_in    (B,K,3)  float32   representative positions mod1(A(u+1/2)+b), in [0,1)
#   lat_in  (B,6)    float32   c_in * l_f(z_noisy)
#   pad_mask(B,K)    bool
#   c_noise (B,)     float32   1/4 ln sigma
#   row     (B,K)    int64     global row ids (0 = pad)
#   G       (B,)     int64     space group ids
#   u       (B,K,3)  float32   centred noisy chart state (required for GEM)
#   z       (B,6)    float32   unscaled noisy free lattice state (required for GEM)
#   family  (B,)     int64     lattice family; GEM uses the frozen group table
# In gem_mode="none", u, z and family are ignored.
# returns {"type_logits": (B,K,16), "coord_vel": (B,K,3), "lattice_vel": (B,6)}
# With G = 1, row = 1 on every real token and zero-initialised extras, the outputs equal
# CrystaliteModel's on the same weights (use_edge_bias = use_distance_bias = False).

EmbedMode = Literal["categorical", "physical", "hybrid"]
SGMode = Literal["none", "categorical", "physical", "hybrid"]
# Orbitol-X reads the ambient representative mod1(A_w(u + 1/2) + b_w) and projects a 3-D update with Pi_w.
# Orbitol-I reads the chart coordinates u on the cover torus R^d / Z^d; its direct readout writes H_w^{-1/2} f.
CoordInput = Literal["explicit", "internal"]
CoordReadout = Literal["projection", "direct"]
# Coordinate objectives as (target, coordinate skip).
# "wrapped":   Crystalite's wrapped residual to the clean point, EDM c_skip. Bitwise Crystalite at P1 under the
#              matched parity settings.
# "posterior": the periodic posterior-mean target with EDM c_skip. The default.
# "posterior_unit": the periodic posterior target with c_skip = 1. Exactly lift-equivariant.
# "posterior_periodic": the periodic target with the lift-equivariant skip u - (1 - c_skip) sin(2 pi u) / (2 pi).
# The historical names keep their exact meaning: "M" is "wrapped" and "P" is "posterior_unit".
OBJECTIVES = {
    "wrapped": ("wrapped", "edm"),
    "posterior": ("posterior", "edm"),
    "posterior_unit": ("posterior", "unit"),
    "posterior_periodic": ("posterior", "periodic"),
}
OBJECTIVE_ALIASES = {"M": "wrapped", "P": "posterior_unit"}
Objective = Literal["wrapped", "posterior", "posterior_unit", "posterior_periodic", "M", "P"]


def canonical_objective(name):
    """Map a coordinate objective or its historical alias (M, P) to the canonical name."""
    name = OBJECTIVE_ALIASES.get(name, name)
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown coordinate objective {name!r}: choose from {sorted(OBJECTIVES)} "
                         f"or the historical aliases {sorted(OBJECTIVE_ALIASES)}")
    return name

# --------------------------------------------------------------------------------------------
# Emitted samples (orbitol.decode -> evaluation): Crystalite's DNG sample format
# --------------------------------------------------------------------------------------------
# Each valid decode is emitted as its exact primitive cell. Batched tensors for the evaluator:
#   A0 (S, N) int64 atomic numbers (0 = pad), F1 (S, N, 3) float32 fractional coords in [0,1),
#   Y1 (S, 6) float32 (ln a, ln b, ln c, cos alpha, cos beta, cos gamma), pad_mask (S, N) bool,
# plus a float64 pymatgen Structure per sample for contains_ops check (a).


@dataclass
class TrainConfig:
    """Model/optimisation with qualified execution and full-pool evaluation defaults."""
    dataset: Dataset = "mp20"
    # model (Crystalite DNG size, GEM off)
    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 14
    mlp_ratio: float = 4.0
    compile_trunk: Literal["auto", "none", "default", "max-autotune-no-cudagraphs", "reduce-overhead"] = "auto"
    # Training-step speed toggles for host/launch overhead outside the trunk.
    # Defaults keep the original implementation; each toggle switches one part to its faster version.
    periodic_target_impl: Literal["loop", "batched"] = "loop"   # posterior_periodic coordinate target
    gem_compile: bool = False                                    # compile GEM geometry and the eager forward
    fast_update: bool = False                                    # cheaper clip / optimizer / EMA update
    fast_encode: bool = False                                    # sync-free element-type encoding
    compile_step: bool = False                                   # one training forward/loss graph per width
    coord_embed_mode: str = "fourier"
    coord_n_freqs: int = 32
    coord_input: CoordInput = "explicit"
    coord_readout: CoordReadout = "projection"
    lattice_embed_mode: str = "mlp"
    row_embed: EmbedMode = "hybrid"
    sg_embed: SGMode = "hybrid"
    gem_mode: Literal["none", "multiplicity", "distance"] = "none"
    # objective
    objective: Objective = "posterior"
    sigma_data: tuple[float, float, float] = (0.3, 0.3, 0.3)   # type, coordinate, lattice
    loss_weights: tuple[float, float, float] = (1.0, 50.0, 5.0)
    P_mean: float = -1.2
    P_std: float = 1.2
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    # optimisation
    batch_size: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    max_steps: int = 300_000
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    seed: int = 123
    num_workers: int = 4
    K: int = 20                     # maximum orbit tokens per structure (task capacity)
    pad_multiple: int = 4           # 0: fixed K; otherwise round batch maximum up, capped at K
    n_prim_max: int = 20            # emitted primitive atoms (MP-20)
    # sampler (Crystalite's settings, anti-annealing off)
    sample_num_steps: int = 150
    sample_token_budget: int = -1   # -1: model-size auto; 0: padded; positive: packed capacity
    decode_impl: Literal["reference", "fast"] = "fast"  # identical outputs; fast needs Numba to be fast
    # Identical outputs; thread decodes chunk k on one worker while chunk k+1 samples.
    decode_overlap: Literal["none", "thread"] = "none"
    rho: float = 7.0
    S_churn: float = 60.0
    S_min: float = 0.0
    S_max: float = 999.0
    S_noise: float = 1.003
    aa_rho_coords: float = 0.0
    aa_rho_lattice: float = 0.0
    # evaluation (Crystalite's train-time DNG evaluation)
    sample_frequency: int = 20_000
    sample_count: int = 10000
    sun_k: int = 0                  # 0: relax all; positive: explicit legacy raw-UN subset
    nequip_eval: bool = True
    eval_mode: Literal["async", "sync"] = "async"
    eval_backlog_max: int = 2
    eval_partition: str = ""  # SBATCH_PARTITION or the cluster default
    eval_account: str = ""  # SBATCH_ACCOUNT or the cluster default
    eval_gres: str = "gpu:1"
    nequip_backend: Literal["oeq", "torchscript"] = "oeq"
    thermo_relax_steps: int = 500
    nequip_relax_mode: Literal["batch", "sequential"] = "batch"
    nequip_relax_batch_size: int = 128
    val_every: int = 25_000         # each validation pass costs ~13 s (uncompiled, 7 sigmas); 5k cost 4-13% of training
    log_every: int = 50
    ckpt_every: int = 100_000
    ckpt_keep: int = 1              # newest numbered snapshots kept on disk; 0 keeps all
    wandb_project: str = ""         # empty: the dataset's project (DatasetSpec.wandb_project)
    wandb_name: str = "orbitol"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.dataset not in DATASET_NAMES:
            raise ValueError(f"Unknown dataset {self.dataset!r}: choose from {list(DATASET_NAMES)}")
        if type(self.pad_multiple) is not int or self.pad_multiple < 0:
            raise ValueError("pad_multiple must be a nonnegative integer (0 keeps fixed K)")
        if type(self.sample_token_budget) is not int or self.sample_token_budget < -1:
            raise ValueError("sample_token_budget must be -1 (auto), 0 (padded), or a positive integer")
        if self.decode_impl not in ("reference", "fast"):
            raise ValueError("decode_impl must be reference or fast")
        if self.decode_overlap not in ("none", "thread"):
            raise ValueError("decode_overlap must be none or thread")
        if self.periodic_target_impl not in ("loop", "batched"):
            raise ValueError("periodic_target_impl must be loop or batched")
        if self.compile_trunk not in ("auto", "none", "default", "max-autotune-no-cudagraphs", "reduce-overhead"):
            raise ValueError("compile_trunk must be auto, none, default, max-autotune-no-cudagraphs or reduce-overhead")
        # Auto values follow throughput measurements: reduce-overhead and an 8192-token sampling budget
        # up to d512, max-autotune-no-cudagraphs and 6144 tokens above.
        # Resolve before model creation and asdict: checkpoints and W&B record actual values.
        if self.compile_trunk == "auto":
            self.compile_trunk = "reduce-overhead" if self.d_model <= 512 else "max-autotune-no-cudagraphs"
        if self.sample_token_budget == -1:
            self.sample_token_budget = 8192 if self.d_model <= 512 else 6144
        if self.eval_mode not in ("async", "sync"):
            raise ValueError("eval_mode must be async or sync")
        if type(self.eval_backlog_max) is not int or self.eval_backlog_max < 1:
            raise ValueError("eval_backlog_max must be a positive integer")
        for name in ("eval_partition", "eval_account", "eval_gres"):
            value = getattr(self, name)
            if (not isinstance(value, str) or (name == "eval_gres" and not value)
                    or any(c.isspace() or c == "\0" for c in value)):
                raise ValueError(f"{name} must be a Slurm resource string without whitespace "
                                 "(eval_gres must be nonempty)")
        if self.nequip_backend not in ("oeq", "torchscript"):
            raise ValueError("nequip_backend must be oeq or torchscript")
        if self.nequip_relax_mode not in ("batch", "sequential"):
            raise ValueError("nequip_relax_mode must be batch or sequential")
        if self.nequip_backend == "oeq" and self.nequip_relax_mode != "batch":
            raise ValueError("oeq requires batch relaxation; select torchscript for sequential parity")
        for name in ("thermo_relax_steps", "nequip_relax_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("sample_count", "sun_k"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        self.objective = canonical_objective(self.objective)
        if self.compile_step:
            if self.periodic_target_impl != "batched" or not self.fast_encode:
                raise ValueError("compile_step requires periodic_target_impl='batched' and fast_encode=True")
            if self.gem_mode not in ("none", "multiplicity", "distance"):
                raise ValueError("compile_step requires a compilable GEM mode: none, multiplicity or distance")
            if OBJECTIVES[self.objective][0] != "posterior":
                raise ValueError("compile_step requires a posterior objective; wrapped targets use eager loops")
            if type(self.K) is not int or self.K < 1:
                raise ValueError("compile_step requires a positive integer K")
            if self.pad_multiple and (self.K + self.pad_multiple - 1) // self.pad_multiple > 5:
                raise ValueError("compile_step allows at most five padded widths; increase pad_multiple "
                                 "or use pad_multiple=0 for fixed K")
        if self.gem_mode not in ("none", "multiplicity", "distance"):
            raise ValueError("gem_mode must be none, multiplicity or distance")
        if self.coord_input not in ("explicit", "internal"):
            raise ValueError("coord_input must be explicit or internal")
        if self.coord_readout not in ("projection", "direct"):
            raise ValueError("coord_readout must be projection or direct")
        if self.coord_input == "internal" and self.coord_embed_mode != "fourier":
            raise ValueError("coord_input=internal needs the deterministic Fourier embedder (coord_embed_mode=fourier)")
        if self.ckpt_keep < 0:
            raise ValueError("ckpt_keep must be >= 0 (0 keeps every numbered snapshot)")


# Fields that saved configs may still carry but TrainConfig no longer has. use_gem could only be False;
# GEM is selected by gem_mode.
RETIRED_FIELDS = frozenset({"use_gem"})


def drop_retired(cfg_dict: dict) -> dict:
    """Remove the RETIRED_FIELDS of a saved config, so it constructs a TrainConfig."""
    return {key: value for key, value in cfg_dict.items() if key not in RETIRED_FIELDS}


def normalize_historical_config(cfg_dict: dict) -> dict:
    """Fill absent checkpoint fields with legacy meanings and drop retired ones.

    Call only when loading saved configs; new runs use TrainConfig's defaults.
    Stored values, including subset sizes and sequential relaxation, always win.
    """
    return {
        "compile_trunk": "none", "pad_multiple": 0, "sample_token_budget": 0,
        "sample_count": 2048, "sun_k": 512, "nequip_eval": False,
        "eval_mode": "sync", "nequip_backend": "torchscript", "thermo_relax_steps": 200,
        "nequip_relax_mode": "batch", "nequip_relax_batch_size": 128,
        **drop_retired(cfg_dict),
    }


def _main(argv=None) -> int:
    """Resolve all local keys for inspection, or just values for shell consumers."""
    args = sys.argv[1:] if argv is None else argv
    failed = False
    for key in args or (*REQUIRED, *DATA_LAYOUT, *REPO_DEFAULTS, *OPTIONAL):
        try:
            value = local_value(key) if key in OPTIONAL or key in SHELL_KEYS else local_path(key)
            if args:
                print(value or "")
            else:
                source = ("env" if os.environ.get(key) else
                          "file" if _read_local_env(local_env_file()).get(key) else
                          "derived" if value is not None else "unset")
                print(f"{key}={value or ''}  # {source}")
        except LocalConfigError as exc:
            print(str(exc), file=sys.stderr)
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(_main())
