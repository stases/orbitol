"""Crystalite over Wyckoff-orbit tokens, with optional GEM v1 bias."""
from __future__ import annotations

import math

import torch

import orbitol.core  # noqa: F401  bind the pinned core before importing src
from orbitol.contracts import CoordInput, CoordReadout, EmbedMode, SGMode, TrainConfig
from orbitol.embeddings import RowEmbedding, SGConditioning
from orbitol.gem import OrbitGEM
from orbitol.gem_geometry import cell_scale, expand_members, pair_distances, primitive_lattice
from src.crystalite.crystalite import CrystaliteModel, mod1
from src.data.mp20_tokens import VZ


def _run_trunk(trunk, x, t_emb, pad_seq, bias=None):
    """Add padding to the shared bias; mask padded queries."""
    x = x.masked_fill(pad_seq[..., None], 0.0)
    key_padding = pad_seq
    if bias is not None:
        key_padding = None
        padding = torch.zeros_like(pad_seq, dtype=x.dtype).masked_fill(pad_seq, float("-inf"))
        bias = bias.to(dtype=x.dtype) + padding[:, None, None, :]
    for block in trunk.blocks:
        x = block(x, t_emb, pad_mask=key_padding, attn_head_bias=bias)
    return trunk.norm_out(x).masked_fill(pad_seq[..., None], 0.0)


class OrbitolModel(CrystaliteModel):
    """Preserve Crystalite's backbone and add row and group embeddings."""

    def __init__(
        self,
        *,
        row_desc: torch.Tensor,
        group_desc: torch.Tensor,
        row_embed: EmbedMode = "hybrid",
        sg_embed: SGMode = "hybrid",
        coord_input: CoordInput = "explicit",
        coord_readout: CoordReadout = "projection",
        row_dof: torch.Tensor | None = None,
        gem_mode: str = "none",
        gem_tables=None,
        gem_n_max: int = 20,
        gem_compile: bool = False,
        compile_step: bool = False,
        **crystalite_kwargs,
    ) -> None:
        if crystalite_kwargs.get("use_edge_bias", False) or crystalite_kwargs.get(
            "use_distance_bias", False
        ):
            raise ValueError("Crystalite edge and distance biases are not supported; keep them disabled.")
        super().__init__(**crystalite_kwargs)
        d_model = self.segment_embed.embedding_dim
        self.row_embed = RowEmbedding(row_embed, row_desc, d_model)
        self.sg_embed = SGConditioning(sg_embed, group_desc, d_model)
        self.coord_input = coord_input
        self.coord_readout = coord_readout
        if coord_input == "internal":
            if row_dof is None:
                raise ValueError("Internal coordinate input requires frozen tables; pass tables= to build_model")
            self.register_buffer("row_dof", row_dof, persistent=False)
        self.gem_mode = gem_mode
        self.compile_step = compile_step
        # Cumulative LLL cap exceedances of compiled distance batches (this process);
        # read only at logging steps. Eager calls leave it unchanged; not checkpoint state.
        self.register_buffer("lll_exceedances", torch.zeros((), dtype=torch.int64)
                             if (gem_compile or compile_step) and gem_mode.startswith("distance") else None,
                             persistent=False)
        self._run_trunk = _run_trunk
        if gem_mode != "none":
            if gem_tables is None:
                raise ValueError("GEM requires frozen tables")
            self.gem = OrbitGEM(self.trunk.n_heads, gem_mode)
            self.gem_n_max = gem_n_max
            self.gem_tables = torch.nn.Module()
            names = ("row_W", "row_v", "row_member_mask", "row_prim_mask", "group_Pc")
            if gem_mode == "distance":
                names += ("group_family",)
            for name in names:
                self.gem_tables.register_buffer(name, getattr(gem_tables, name), persistent=False)
        # Separate functions (not OptimizedModules) preserve state_dict names.
        # Without compile_step, only the trunk owns CUDA graphs. The smaller
        # regions neither mark steps nor retain outputs across calls.
        self._run_inputs = OrbitolModel._forward_inputs
        self._run_heads = OrbitolModel._forward_heads
        # The training step owns compilation when requested. Keep these Python
        # callables (and the trunk) inline, including when gem_compile is set.
        if gem_compile and not compile_step:
            options = {"triton.cudagraphs": False, "emulate_precision_casts": True}
            self._run_inputs = torch.compile(self._run_inputs, fullgraph=True, dynamic=False, options=options)
            self._run_heads = torch.compile(self._run_heads, fullgraph=True, dynamic=False, options=options)

    def _gem_bias(self, u, z, row, G, pad_mask, seq_len):
        if u is None or z is None:
            raise ValueError("GEM requires the noisy chart state u and unscaled lattice z")
        with torch.autocast(u.device.type, enabled=False):
            # Geometry is data: only the bias MLP needs gradients.
            with torch.no_grad():
                members = expand_members(u.float(), row, G, pad_mask.bool(), self.gem_tables,
                                         n_max=self.gem_n_max)
                d = scale = None
                if self.gem_mode == "distance":
                    L = primitive_lattice(z.float(), G, self.gem_tables)
                    if torch.compiler.is_compiling():
                        d = pair_distances(members.x, members.valid, L,
                                           lll_exceedance_counter=self.lll_exceedances)
                    else:
                        d = pair_distances(members.x, members.valid, L)
                    scale = cell_scale(L)
            return self.gem(members, d, scale, seq_len=seq_len)

    def _internal_coord_embed(self, u, frac_mod, row):
        """Fourier features of the chart coordinates, periodic under integer shifts of u (the cover torus
        R^d/Z^d). Axes beyond d are zeroed; d = 0 rows keep Orbitol-X's fixed position mod1(b_w)."""
        if u is None:
            raise ValueError("Internal coordinate input requires the noisy chart state u")
        dof = self.row_dof[row]
        fixed = (dof == 0)[..., None]
        q = torch.where(fixed, frac_mod, mod1(mod1(u + 0.5)))
        args = 2 * math.pi * q[..., None] * self.coord_embed.freqs[None, None, None, :]
        feats = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        active = fixed | (torch.arange(3, device=u.device) < dof[..., None])
        return self.coord_embed.mlp(feats.masked_fill(~active[..., None], 0.0).reshape(q.shape[0], q.shape[1], -1))

    def forward(
        self,
        type_in: torch.Tensor,
        x_in: torch.Tensor,
        lat_in: torch.Tensor,
        pad_mask: torch.Tensor,
        c_noise: torch.Tensor,
        *,
        row: torch.Tensor,
        G: torch.Tensor,
        u: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        family: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return raw type, coordinate and lattice predictions for orbit tokens."""
        # Sampling, including packed.py's all-padding fallback, stays eager.
        run_inputs = self._run_inputs if torch.is_grad_enabled() else OrbitolModel._forward_inputs
        run_heads = self._run_heads if torch.is_grad_enabled() else OrbitolModel._forward_heads
        x, t_emb, pad_seq, bias, frac_mod, capacity_ok = run_inputs(
            self, type_in, x_in, lat_in, pad_mask, c_noise, row, G, u, z)
        if capacity_ok is not None:
            torch._assert_async(capacity_ok, "primitive member count exceeds n_max")
        h = self._run_trunk(self.trunk, x, t_emb, pad_seq, bias)
        return run_heads(self, h, frac_mod, pad_mask)

    def _forward_inputs(self, type_in, x_in, lat_in, pad_mask, c_noise, row, G, u, z):
        """Embeddings and fp32 GEM, bucketed by padded token width.

        Member capacity stays fixed at gem_n_max; member counts are tensor
        values and never select graphs. For a fixed batch size and execution
        context, K=4,8,12,16,20 needs at most five graphs per compiled region.
        No-grad sampling/validation and packed sampling keep eager GEM.
        """
        # Keep the pinned Crystalite operation order, including its internal mod1.
        frac_mod = mod1(x_in)
        h_type = self.type_proj(type_in) + (
            self._internal_coord_embed(u, frac_mod, row) if self.coord_input == "internal"
            else self.coord_embed(frac_mod)
        ) + self.segment_embed.weight[0]
        h_type = h_type + self.row_embed(row)
        h_lat = self.lattice_embed(lat_in) + self.segment_embed.weight[1]
        h_lat = h_lat[:, None, :]

        x = torch.cat([h_type, h_lat], dim=1)
        pad_seq = torch.cat(
            [
                pad_mask.bool(),
                torch.zeros((pad_mask.shape[0], 1), device=pad_mask.device, dtype=torch.bool),
            ],
            dim=1,
        )
        t_emb = self.time(c_noise, c_noise) + self.sg_embed(G)
        bias = (self._gem_bias(u, z, row, G, pad_mask, x.shape[1])
                if self.gem_mode != "none" else None)
        capacity_ok = None
        if torch.compiler.is_compiling() and self.gem_mode != "none" and not self.compile_step:
            # Reuse expansion's count reduction through compiler CSE. Assert
            # outside Inductor: its _assert_async decomposition is a no-op.
            capacity_ok = self.member_capacity_ok(row, pad_mask)
        return x, t_emb, pad_seq, bias, frac_mod, capacity_ok

    def member_capacity_ok(self, row, pad_mask):
        """Export this predicate for an eager device assert after compilation."""
        if self.gem_mode == "none":
            return None
        keep = (self.gem_tables.row_member_mask[row] & self.gem_tables.row_prim_mask[row]
                & ~pad_mask.bool().unsqueeze(-1))
        return (keep.sum(-1).sum(-1) <= self.gem_n_max).all()

    def _forward_heads(self, h, frac_mod, pad_mask):
        """Readout shared by GEM regions and whole-step compilation."""
        return self.heads(h, frac_coords=frac_mod, pad_mask=pad_mask.bool())


def build_model(cfg: TrainConfig, row_desc: torch.Tensor, group_desc: torch.Tensor, tables=None) -> OrbitolModel:
    """Build the configured model; TrainConfig defaults give the paper model size."""
    if cfg.gem_mode != "none" and tables is None:
        raise ValueError("GEM requires frozen tables; pass tables= to build_model")
    model = OrbitolModel(
        row_desc=row_desc,
        group_desc=group_desc,
        row_embed=cfg.row_embed,
        sg_embed=cfg.sg_embed,
        coord_input=cfg.coord_input,
        coord_readout=cfg.coord_readout,
        row_dof=tables.row_dof if tables is not None else None,
        gem_mode=cfg.gem_mode,
        gem_tables=tables,
        gem_n_max=cfg.n_prim_max,
        gem_compile=cfg.gem_compile,
        compile_step=cfg.compile_step,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        mlp_ratio=cfg.mlp_ratio,
        type_dim=16,
        vz=VZ,
        coord_embed_mode=cfg.coord_embed_mode,
        n_freqs=cfg.coord_n_freqs,
        lattice_embed_mode=cfg.lattice_embed_mode,
        lattice_repr="ltri",
        coord_head_mode="direct",
        use_edge_bias=False,
        use_distance_bias=False,
    )
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"OrbitolModel: {n_params:,} parameters (row={cfg.row_embed}, sg={cfg.sg_embed})")
    if cfg.compile_trunk != "none" and not cfg.compile_step:
        # Static token buckets specialize the kernels. Leave room for train,
        # validation and sampling shapes on the shared _run_trunk code object.
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        # Keep eager bf16 rounding points when fusing modulation and residuals.
        options = {**torch._inductor.list_mode_options(cfg.compile_trunk),
                   "emulate_precision_casts": True}
        model._run_trunk = torch.compile(_run_trunk, dynamic=False, options=options)
    return model
