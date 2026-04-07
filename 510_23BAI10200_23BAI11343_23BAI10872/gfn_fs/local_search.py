from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .env import FeatureSelectionEnv
from .model import GFlowNetModel
from .reward import RewardModel, SubsetMetrics
from .sampling import sample_action


@dataclass
class LocalSearchResult:
    proposed_order: List[int]
    proposed_metrics: SubsetMetrics
    accepted: bool
    backtrack_steps: int
    removed_features: List[int]


class LocalSearchRefiner:
    """Student-friendly LS-GFN refinement.

    Starting from a sampled terminal subset, we:
    1) backtrack a few steps using the learned backward policy,
    2) reconstruct to a terminal subset using the learned forward policy,
    3) accept the new subset if it is at least as good as the old one.

    This mirrors the paper's core idea while remaining easy to understand.
    """

    def __init__(
        self,
        env: FeatureSelectionEnv,
        scorer: RewardModel,
        device: torch.device,
        rng: np.random.Generator,
        max_backtrack: int = 2,
        temperature: float = 1.0,
    ) -> None:
        self.env = env
        self.scorer = scorer
        self.device = device
        self.rng = rng
        self.max_backtrack = max_backtrack
        self.temperature = temperature

    def refine(
        self,
        model: GFlowNetModel,
        current_order: List[int],
        current_metrics: SubsetMetrics,
    ) -> LocalSearchResult:
        state = self.env.state_from_order(current_order)
        max_steps = min(self.max_backtrack, len(current_order))
        backtrack_steps = int(self.rng.integers(1, max_steps + 1))

        working_state = state.copy()
        removed: List[int] = []
        for _ in range(backtrack_steps):
            valid_backward = self.env.available_backward_actions(working_state)
            action = sample_action(
                model=model,
                state=working_state,
                valid_actions=valid_backward,
                direction="backward",
                device=self.device,
                rng=self.rng,
                epsilon=0.0,
                temperature=self.temperature,
            )
            removed.append(action)
            working_state = self.env.step_backward(working_state, action)

        base_order = [a for a in current_order if a not in set(removed)]
        proposed_order = list(base_order)
        while len(proposed_order) < self.env.subset_size:
            valid_forward = self.env.available_forward_actions(working_state)
            action = sample_action(
                model=model,
                state=working_state,
                valid_actions=valid_forward,
                direction="forward",
                device=self.device,
                rng=self.rng,
                epsilon=0.0,
                temperature=self.temperature,
            )
            proposed_order.append(action)
            working_state = self.env.step_forward(working_state, action)

        proposed_metrics = self.scorer.evaluate(proposed_order)
        accepted = proposed_metrics.reward >= current_metrics.reward
        return LocalSearchResult(
            proposed_order=proposed_order,
            proposed_metrics=proposed_metrics,
            accepted=accepted,
            backtrack_steps=backtrack_steps,
            removed_features=removed,
        )
