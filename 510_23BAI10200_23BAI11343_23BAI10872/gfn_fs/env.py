from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureSelectionConfig:
    num_features: int
    subset_size: int


class FeatureSelectionEnv:
    """Fixed-cardinality feature subset selection.

    State: a binary mask of length D.
    Action: choose one currently unselected feature.
    Terminal: exactly K features have been selected.
    """

    def __init__(self, config: FeatureSelectionConfig) -> None:
        if config.subset_size <= 0 or config.subset_size > config.num_features:
            raise ValueError("subset_size must be between 1 and num_features")
        self.config = config
        self.num_features = config.num_features
        self.subset_size = config.subset_size

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.num_features, dtype=np.float32)

    def is_terminal(self, state: np.ndarray) -> bool:
        return int(state.sum()) >= self.subset_size

    def available_forward_actions(self, state: np.ndarray) -> List[int]:
        if self.is_terminal(state):
            return []
        return np.where(state < 0.5)[0].tolist()

    def available_backward_actions(self, state: np.ndarray) -> List[int]:
        return np.where(state > 0.5)[0].tolist()

    def step_forward(self, state: np.ndarray, action: int) -> np.ndarray:
        if state[action] > 0.5:
            raise ValueError(f"Feature {action} is already selected.")
        next_state = state.copy()
        next_state[action] = 1.0
        return next_state

    def step_backward(self, state: np.ndarray, action: int) -> np.ndarray:
        if state[action] < 0.5:
            raise ValueError(f"Feature {action} is not selected.")
        next_state = state.copy()
        next_state[action] = 0.0
        return next_state

    def state_from_order(self, order: Sequence[int]) -> np.ndarray:
        state = self.initial_state()
        for action in order:
            state = self.step_forward(state, action)
        return state

    def subset_from_state(self, state: np.ndarray) -> List[int]:
        return np.where(state > 0.5)[0].tolist()
