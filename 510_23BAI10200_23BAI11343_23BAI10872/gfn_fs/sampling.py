from __future__ import annotations

from typing import List, Literal, Sequence

import numpy as np
import torch

from .env import FeatureSelectionEnv
from .model import GFlowNetModel

Direction = Literal["forward", "backward"]


def _masked_probs(logits: torch.Tensor, valid_actions: Sequence[int], temperature: float = 1.0) -> np.ndarray:
    logits = logits.detach().cpu().numpy().astype(np.float64).copy()
    mask = np.full_like(logits, -1e9)
    valid_actions = list(valid_actions)
    mask[valid_actions] = logits[valid_actions] / max(temperature, 1e-6)
    mask = mask - mask.max()
    probs = np.exp(mask)
    probs = probs / probs.sum()
    return probs


def sample_action(
    model: GFlowNetModel,
    state: np.ndarray,
    valid_actions: Sequence[int],
    direction: Direction,
    device: torch.device,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    temperature: float = 1.0,
    greedy: bool = False,
) -> int:
    valid_actions = list(valid_actions)
    if not valid_actions:
        raise ValueError("No valid actions available.")

    if (not greedy) and epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.choice(valid_actions))

    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if direction == "forward":
            logits = model.forward_logits(state_tensor)[0]
        elif direction == "backward":
            logits = model.backward_logits(state_tensor)[0]
        else:
            raise ValueError(f"Unknown direction: {direction}")

    probs = _masked_probs(logits, valid_actions=valid_actions, temperature=temperature)
    if greedy:
        return int(np.argmax(probs))
    chosen = int(rng.choice(np.array(valid_actions), p=probs[np.array(valid_actions)] / probs[np.array(valid_actions)].sum()))
    return chosen


def sample_forward_trajectory(
    model: GFlowNetModel,
    env: FeatureSelectionEnv,
    device: torch.device,
    rng: np.random.Generator,
    epsilon: float = 0.0,
    temperature: float = 1.0,
    greedy: bool = False,
) -> List[int]:
    state = env.initial_state()
    order: List[int] = []
    while not env.is_terminal(state):
        valid = env.available_forward_actions(state)
        action = sample_action(
            model=model,
            state=state,
            valid_actions=valid,
            direction="forward",
            device=device,
            rng=rng,
            epsilon=epsilon,
            temperature=temperature,
            greedy=greedy,
        )
        order.append(action)
        state = env.step_forward(state, action)
    return order
