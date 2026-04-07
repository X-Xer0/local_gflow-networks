from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class GFlowNetModel(nn.Module):
    """Shared trunk with forward and backward policy heads plus logZ."""

    def __init__(self, num_features: int, hidden_dim: int = 128) -> None:
        super().__init__()
        input_dim = num_features + 1  # binary mask + normalized step count
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.forward_head = nn.Linear(hidden_dim, num_features)
        self.backward_head = nn.Linear(hidden_dim, num_features)
        self.logZ = nn.Parameter(torch.tensor(0.0))
        self.num_features = num_features

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        step_fraction = state.sum(dim=-1, keepdim=True) / float(self.num_features)
        x = torch.cat([state, step_fraction], dim=-1)
        return self.trunk(x)

    def forward_logits(self, state: torch.Tensor) -> torch.Tensor:
        h = self.encode_state(state)
        return self.forward_head(h)

    def backward_logits(self, state: torch.Tensor) -> torch.Tensor:
        h = self.encode_state(state)
        return self.backward_head(h)


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    large_neg = torch.finfo(logits.dtype).min / 2.0
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, large_neg))
    return torch.log_softmax(masked_logits, dim=dim)
