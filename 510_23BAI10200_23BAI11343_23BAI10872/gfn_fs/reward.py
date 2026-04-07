from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .data import DataBundle


@dataclass
class SubsetMetrics:
    subset: Tuple[int, ...]
    reward: float
    log_reward: float
    val_balanced_accuracy: float
    val_accuracy: float
    val_f1: float
    test_balanced_accuracy: float
    test_accuracy: float
    test_f1: float


class RewardModel:
    """Scores a subset by fitting a lightweight classifier on the chosen features."""

    def __init__(self, data: DataBundle, reward_scale: float = 5.0, seed: int = 42) -> None:
        self.data = data
        self.reward_scale = reward_scale
        self.seed = seed
        self.cache: Dict[Tuple[int, ...], SubsetMetrics] = {}

    def evaluate(self, subset: Sequence[int]) -> SubsetMetrics:
        subset_key = tuple(sorted(subset))
        if subset_key in self.cache:
            return self.cache[subset_key]

        if len(subset_key) == 0:
            raise ValueError("Subset must contain at least one feature.")

        X_train = self.data.X_train[:, subset_key]
        X_val = self.data.X_val[:, subset_key]
        X_test = self.data.X_test[:, subset_key]

        clf = LogisticRegression(
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=self.seed,
        )
        clf.fit(X_train, self.data.y_train)

        val_pred = clf.predict(X_val)
        test_pred = clf.predict(X_test)

        val_bal_acc = balanced_accuracy_score(self.data.y_val, val_pred)
        val_acc = accuracy_score(self.data.y_val, val_pred)
        val_f1 = f1_score(self.data.y_val, val_pred)

        test_bal_acc = balanced_accuracy_score(self.data.y_test, test_pred)
        test_acc = accuracy_score(self.data.y_test, test_pred)
        test_f1 = f1_score(self.data.y_test, test_pred)

        reward = float(exp(self.reward_scale * val_bal_acc))
        metrics = SubsetMetrics(
            subset=subset_key,
            reward=reward,
            log_reward=float(np.log(reward)),
            val_balanced_accuracy=float(val_bal_acc),
            val_accuracy=float(val_acc),
            val_f1=float(val_f1),
            test_balanced_accuracy=float(test_bal_acc),
            test_accuracy=float(test_acc),
            test_f1=float(test_f1),
        )
        self.cache[subset_key] = metrics
        return metrics

    def exhaustive_search(self, subset_size: int) -> SubsetMetrics:
        best_metrics = None
        all_indices = list(range(len(self.data.feature_names)))
        for subset in combinations(all_indices, subset_size):
            metrics = self.evaluate(subset)
            if best_metrics is None or metrics.reward > best_metrics.reward:
                best_metrics = metrics
        if best_metrics is None:
            raise RuntimeError("Exhaustive search found no subsets.")
        return best_metrics
