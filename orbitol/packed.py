"""Parameter-sharing packed inference with fixed token/structure capacity.

Layouts depend only on the skeleton and may be reused during sampling. Build
them once per chunk from CPU padding metadata. The dense
SDPA backend is a small-test reference, not a scalable packed implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

import orbitol.core  # noqa: F401
from src.crystalite.crystalite import mod1


@dataclass(frozen=True)
class PackedLayout:
    batch: int
    width: int
    real_tokens: int
    capacity: int
    structure_capacity: int
    orbit_flat: torch.Tensor
    orbit_positions: torch.Tensor
    lattice_positions: torch.Tensor
    document: torch.Tensor
    column: torch.Tensor

    def to(self, device):
        return PackedLayout(**{name: value.to(device) if isinstance(value, torch.Tensor) else value
                               for name, value in vars(self).items()})


def pack_layout(pad_mask, token_budget=None):
    """Retain original orbit columns, including non-prefix padding holes.

    A budget reserves a fixed trunk length; each unused slot is its own dummy
    document. It never truncates a structure or changes the batch membership.
    """
    if pad_mask.ndim != 2 or not pad_mask.shape[0] or not pad_mask.shape[1]:
        raise ValueError("pad_mask must be a nonempty (B, K) tensor")
    B, K = pad_mask.shape
    keep = ~pad_mask.bool()
    orbit_flat = keep.flatten().nonzero().flatten()
    lengths = keep.sum(1) + 1
    ends = lengths.cumsum(0)
    total = orbit_flat.numel() + B
    capacity = total if token_budget is None else token_budget
    if not isinstance(capacity, int) or capacity < total:
        raise ValueError(f"token budget {capacity} cannot hold {total} real tokens")
    positions = torch.arange(total, device=pad_mask.device)
    document = torch.repeat_interleave(torch.arange(B, device=pad_mask.device), lengths)
    orbit_positions = positions[positions != (ends - 1)[document]]
    column = torch.full((total,), K, device=pad_mask.device, dtype=torch.long)
    column[orbit_positions] = orbit_flat % K
    # Every filler attends only to itself, with a defined one-element softmax.
    document = torch.cat((document, torch.arange(B, B + capacity - total, device=pad_mask.device)))
    column = F.pad(column, (0, capacity - total), value=0)
    # Sampling requires >=1 orbit per structure, hence B <= T//2. The max also
    # supports lattice-only documents in the small network reference tests.
    structures = B if token_budget is None else max(B, capacity // 2)
    return PackedLayout(B, K, total, capacity, structures, orbit_flat, orbit_positions,
                        ends - 1, document, column)


_compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)


def document_mask(layout: PackedLayout, *, compiled=True):
    """Build a head-shared document BlockMask; charge once per new batch."""
    document, size = layout.document, layout.capacity

    def same_document(b, h, q, k):
        # create_block_mask evaluates the final partial tile out to BLOCK_SIZE.
        return ((q < size) & (k < size)
                & (document[q.clamp(max=size - 1)] == document[k.clamp(max=size - 1)]))

    # The deprecated _compile=True path uses automatic dynamic shapes in 2.8;
    # changing capacity through that path produced incorrect attention gradients.
    maker = _compiled_create_block_mask if compiled else create_block_mask
    return maker(same_document, B=1, H=None, Q_LEN=size, KV_LEN=size,
                 device=document.device)


_compiled_flex = torch.compile(flex_attention, dynamic=False)


def _packed_attention(q, k, v, layout, block_mask, bias, backend):
    document, column = layout.document, layout.column
    if backend == "flex":
        score_mod = None
        if bias is not None:
            def score_mod(score, b, head, qi, ki):
                qi = qi.clamp(max=layout.capacity - 1)
                ki = ki.clamp(max=layout.capacity - 1)
                doc = document[qi].clamp(max=bias.shape[0] - 1)
                return score + bias[doc, head, column[qi], column[ki]]
        attend = flex_attention if torch.compiler.is_compiling() else _compiled_flex
        # PyTorch 2.8's short-query decoding kernel miscompiles these document
        # masks inside a compiled inference trunk (T < 128). Use the main kernel.
        out = attend(q, k, v, score_mod=score_mod, block_mask=block_mask,
                     kernel_options={"FORCE_USE_FLEX_ATTENTION": True})
    else:
        allowed = document[:, None] == document[None, :]
        if bias is None:
            mask = allowed
        else:
            doc = document.clamp(max=bias.shape[0] - 1)
            mask = bias[doc[:, None, None], torch.arange(q.shape[1], device=q.device)[None, :, None],
                        column[:, None, None], column[None, None, :]].permute(1, 0, 2)
            mask = mask.masked_fill(~allowed, -torch.inf)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return out


def _attention(block, h, layout, block_mask, bias, backend):
    size, dim = h.shape
    heads = block.n_heads
    qkv = F.linear(h, block.attn.in_proj_weight, block.attn.in_proj_bias)
    q, k, v = qkv.reshape(size, 3, heads, dim // heads).permute(1, 2, 0, 3).unbind(0)
    q, k, v = (value.unsqueeze(0).contiguous() for value in (q, k, v))
    out = _packed_attention(q, k, v, layout, block_mask, bias, backend)
    out = out.squeeze(0).transpose(0, 1).reshape(size, dim)
    return F.linear(out, block.attn.out_proj.weight, block.attn.out_proj.bias)


def _bias_for_scores(bias, dtype):
    """Round the GEM bias to the score dtype (bf16 under autocast), as padded autocast SDPA does."""
    return None if bias is None else bias.to(dtype)


def packed_trunk(trunk, x, t_emb, layout, block_mask=None, bias=None, backend="flex"):
    """The pinned adaLN math, with per-structure modulation gathered once."""
    doc = layout.document.clamp(max=t_emb.shape[0] - 1)
    for block in trunk.blocks:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (
            block.adaLN_mod(t_emb)[doc].chunk(6, dim=-1))
        h = block.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * _attention(block, h, layout, block_mask, bias, backend)
        h = block.norm2(x) * (1 + scale_m) + shift_m
        x = x + gate_m * block.mlp(h)
    return trunk.norm_out(x)


def _scatter_head(head, h, layout):
    values = head(h)
    # The original heads see zero hidden states on padding, hence return bias.
    padded = head.bias.to(values.dtype).expand(layout.batch * layout.width, -1)
    return padded.index_copy(0, layout.orbit_flat, values).reshape(layout.batch, layout.width, -1)


@torch.no_grad()
def packed_forward(model, type_in, x_in, lat_in, pad_mask, c_noise, *, row, G,
                   u=None, z=None, family=None, layout=None, block_mask=None,
                   backend="flex", run_trunk=packed_trunk):
    """Reuse model parameters for real-token embeddings, trunk and heads.

    Inputs and outputs retain Orbitol's padded public interface. Layout and mask
    can be passed explicitly to separate preparation from network timing.
    Inference only. Nonzero dropout is rejected to preserve random draws.
    """
    if backend not in {"flex", "dense"}:
        raise ValueError("backend must be flex or dense")
    if model.heads.coord_head_mode != "direct":
        raise ValueError("packed sampling requires the Orbitol direct coordinate head")
    if any(block.attn.dropout or block.mlp.drop1.p or block.mlp.drop2.p
           for block in model.trunk.blocks):
        raise ValueError("packed sampling requires zero dropout to preserve RNG draws")
    layout = pack_layout(pad_mask) if layout is None else layout
    if (layout.batch, layout.width) != tuple(pad_mask.shape):
        raise ValueError("layout and input shape differ")
    if layout.orbit_flat.numel() == 0:
        # The pinned Fourier embedders cannot reshape an empty token axis.
        # Keep the padded path's lattice-only outputs.
        return model(type_in, x_in, lat_in, pad_mask, c_noise, row=row, G=G,
                     u=u, z=z, family=family)
    if backend == "flex" and block_mask is None:
        block_mask = document_mask(layout)
    indices = layout.orbit_flat
    frac = mod1(x_in.flatten(0, 1)[indices])[None]
    packed_row = row.flatten()[indices][None]
    coords = (model._internal_coord_embed(
        None if u is None else u.flatten(0, 1)[indices][None], frac, packed_row)
        if model.coord_input == "internal" else model.coord_embed(frac))
    orbit = (model.type_proj(type_in.flatten(0, 1)[indices])[None] + coords
             + model.segment_embed.weight[0]) + model.row_embed(packed_row)
    lattice = model.lattice_embed(lat_in) + model.segment_embed.weight[1]
    x = orbit.new_zeros(layout.capacity, orbit.shape[-1])
    x = x.index_copy(0, layout.orbit_positions, orbit[0])
    x = x.index_copy(0, layout.lattice_positions, lattice)
    t_emb = model.time(c_noise, c_noise) + model.sg_embed(G)
    bias = (model._gem_bias(u, z, row, G, pad_mask, layout.width + 1)
            if model.gem_mode != "none" else None)
    # Round the bias outside the compiled trunk: PyTorch 2.8 cannot lower a
    # freshly cast (FlexibleLayout) bias captured by the Flex score modifier.
    dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    bias = _bias_for_scores(bias, dtype)
    extra = layout.structure_capacity - layout.batch
    t_emb = F.pad(t_emb, (0, 0, 0, extra))
    if bias is not None:
        bias = F.pad(bias, (0, 0, 0, 0, 0, 0, 0, extra))
    h = run_trunk(model.trunk, x, t_emb, layout, block_mask, bias, backend)
    atoms = h[layout.orbit_positions]
    return {"type_logits": _scatter_head(model.heads.type_head, atoms, layout),
            "coord_vel": _scatter_head(model.heads.coord_head, atoms, layout),
            "lattice_vel": model.heads.lattice_head(h[layout.lattice_positions])}


def token_chunks(skeletons, token_budget):
    """Greedy contiguous slices, with one lattice token per structure."""
    start, used = 0, 0
    chunks = []
    for i, skeleton in enumerate(skeletons):
        tokens = len(skeleton.rows) + 1
        if tokens < 2:
            raise ValueError("packed sampling requires at least one orbit per structure")
        if tokens > token_budget:
            raise ValueError(f"token budget {token_budget} cannot hold structure {i} ({tokens} tokens)")
        if used + tokens > token_budget:
            chunks.append((start, i))
            start, used = i, 0
        used += tokens
    if used:
        chunks.append((start, len(skeletons)))
    return chunks


@lru_cache(maxsize=None)
def sampling_trunk(mode):
    """Reuse the compiled callable across chunks and sampling pools."""
    if mode == "none":
        return packed_trunk
    return torch.compile(packed_trunk, dynamic=False, fullgraph=True, options={
        **torch._inductor.list_mode_options(mode), "emulate_precision_casts": True})


class PackedDenoiser:
    """Pack only the network call; EDM and all noise stay padded at K."""

    def __init__(self, model, skeleton, *, token_budget, compile_mode="none",
                 autocast_dtype=None, skip_type_scaling=False):
        self.model = model
        pad = skeleton["pad_mask"]
        self.layout = pack_layout(pad.cpu(), token_budget).to(pad.device)
        self.backend = "flex" if pad.is_cuda else "dense"
        self.mask = document_mask(self.layout) if self.backend == "flex" else None
        self.run_trunk = sampling_trunk(compile_mode)
        self.autocast_dtype, self.skip_type_scaling = autocast_dtype, skip_type_scaling
        self.gem_mode = model.gem_mode
        self.coord_input, self.coord_readout = model.coord_input, model.coord_readout

    def __call__(self, *args, **kwargs):
        return packed_forward(self.model, *args, **kwargs, layout=self.layout,
                              block_mask=self.mask, backend=self.backend, run_trunk=self.run_trunk)

    def denoise(self, *args, **kwargs):
        from orbitol.edm import denoise_orbit

        torch.compiler.cudagraph_mark_step_begin()
        return denoise_orbit(self, *args, **kwargs, autocast_dtype=self.autocast_dtype,
                             skip_type_scaling=self.skip_type_scaling)
