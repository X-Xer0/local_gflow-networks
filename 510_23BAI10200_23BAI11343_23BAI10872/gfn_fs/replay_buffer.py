from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .reward import SubsetMetrics


@dataclass
class TrajectoryRecord:
    order: List[int]
    metrics: SubsetMetrics
    source: str = "sampled"
    accepted: bool = True
    metadata: dict = field(default_factory=dict)


class RewardPrioritizedReplay:
    def __init__(self, capacity: int = 5000, alpha: float = 0.8, seed: int = 42) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.records: List[TrajectoryRecord] = []

    def __len__(self) -> int:
        return len(self.records)

    def add(self, record: TrajectoryRecord) -> None:
        self.records.append(record)
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def extend(self, records: Sequence[TrajectoryRecord]) -> None:
        for record in records:
            self.add(record)

    def sample(self, batch_size: int) -> List[TrajectoryRecord]:
        if len(self.records) == 0:
            raise ValueError("Replay buffer is empty.")
        if len(self.records) <= batch_size:
            return list(self.records)

        rewards = np.array([r.metrics.reward for r in self.records], dtype=np.float64)
        probs = rewards ** self.alpha
        probs = probs / probs.sum()
        idx = self.rng.choice(len(self.records), size=batch_size, replace=False, p=probs)
        return [self.records[i] for i in idx]

    def best(self) -> Optional[TrajectoryRecord]:
        if not self.records:
            return None
        return max(self.records, key=lambda x: x.metrics.reward)

    def topk(self, k: int = 10) -> List[TrajectoryRecord]:
        return sorted(self.records, key=lambda x: x.metrics.reward, reverse=True)[:k]
