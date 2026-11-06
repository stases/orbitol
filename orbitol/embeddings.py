"""Zero-initialised row embeddings and space-group conditioning."""
from __future__ import annotations

import torch
from torch import nn

from orbitol.contracts import EmbedMode, G_TOTAL, R_TOTAL, ROW_DESC_DIM, SG_DESC_DIM, SGMode


def _descriptor_mlp(in_dim: int, d_model: int) -> nn.Sequential:
    mlp = nn.Sequential(
        nn.Linear(in_dim, d_model),
        nn.SiLU(),
        nn.Linear(d_model, d_model),
    )
    nn.init.zeros_(mlp[-1].weight)
    nn.init.zeros_(mlp[-1].bias)
    return mlp


class RowEmbedding(nn.Module):
    """Embed each row by its identity, descriptor, or their sum."""

    def __init__(self, mode: EmbedMode, row_desc: torch.Tensor, d_model: int) -> None:
        super().__init__()
        if mode not in {"categorical", "physical", "hybrid"}:
            raise ValueError(f"Unsupported row embedding mode: {mode}")
        if row_desc.shape != (R_TOTAL, ROW_DESC_DIM):
            raise ValueError(f"row_desc must have shape ({R_TOTAL}, {ROW_DESC_DIM})")
        self.mode = mode
        self.register_buffer("row_desc", row_desc.detach().clone(), persistent=False)
        self.residual = None
        self.mlp = None
        if mode in {"categorical", "hybrid"}:
            self.residual = nn.Embedding(R_TOTAL, d_model)
            nn.init.zeros_(self.residual.weight)
        if mode in {"physical", "hybrid"}:
            self.mlp = _descriptor_mlp(ROW_DESC_DIM, d_model)

    def forward(self, row: torch.Tensor) -> torch.Tensor:
        if self.residual is None:
            return self.mlp(self.row_desc[row])
        embedding = self.residual(row)
        if self.mlp is not None:
            embedding = embedding + self.mlp(self.row_desc[row])
        return embedding


class SGConditioning(nn.Module):
    """Condition AdaLN on group identity, descriptors, both, or neither."""

    def __init__(self, mode: SGMode, group_desc: torch.Tensor, d_model: int) -> None:
        super().__init__()
        if mode not in {"none", "categorical", "physical", "hybrid"}:
            raise ValueError(f"Unsupported space-group conditioning mode: {mode}")
        if group_desc.shape != (G_TOTAL, SG_DESC_DIM):
            raise ValueError(f"group_desc must have shape ({G_TOTAL}, {SG_DESC_DIM})")
        self.mode = mode
        self.d_model = d_model
        self.register_buffer("group_desc", group_desc.detach().clone(), persistent=False)
        self.residual = None
        self.mlp = None
        if mode in {"categorical", "hybrid"}:
            self.residual = nn.Embedding(G_TOTAL, d_model)
            nn.init.zeros_(self.residual.weight)
        if mode in {"physical", "hybrid"}:
            self.mlp = _descriptor_mlp(SG_DESC_DIM, d_model)

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        if self.mode == "none":
            return self.group_desc.new_zeros((*G.shape, self.d_model))
        if self.residual is None:
            return self.mlp(self.group_desc[G])
        embedding = self.residual(G)
        if self.mlp is not None:
            embedding = embedding + self.mlp(self.group_desc[G])
        return embedding
