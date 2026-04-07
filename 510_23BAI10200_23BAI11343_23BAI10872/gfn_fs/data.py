from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    original_feature_names: List[str]
    selected_feature_indices: np.ndarray


class DatasetBuilder:
    """Prepare a small but realistic feature-selection benchmark.

    We use the breast cancer dataset from scikit-learn, split it into
    train/validation/test, standardize it using the train split only, and then
    keep only a candidate pool of features ranked by mutual information on the
    training split. This keeps the project runnable on a laptop while still
    leaving a combinatorial search space.
    """

    def __init__(
        self,
        candidate_features: int = 15,
        seed: int = 42,
        test_size: float = 0.2,
        val_size: float = 0.2,
    ) -> None:
        self.candidate_features = candidate_features
        self.seed = seed
        self.test_size = test_size
        self.val_size = val_size

    def build(self) -> DataBundle:
        data = load_breast_cancer()
        X = data.data.astype(np.float32)
        y = data.target.astype(np.int64)
        original_feature_names = list(data.feature_names)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=y,
        )

        adjusted_val_size = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=adjusted_val_size,
            random_state=self.seed,
            stratify=y_train_val,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        mi = mutual_info_classif(X_train, y_train, random_state=self.seed)
        top_idx = np.argsort(mi)[::-1][: self.candidate_features]
        top_idx = np.sort(top_idx)

        X_train = X_train[:, top_idx]
        X_val = X_val[:, top_idx]
        X_test = X_test[:, top_idx]
        feature_names = [original_feature_names[i] for i in top_idx]

        return DataBundle(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            original_feature_names=original_feature_names,
            selected_feature_indices=top_idx,
        )


def subset_to_feature_names(subset: List[int], feature_names: List[str]) -> List[str]:
    return [str(feature_names[i]) for i in subset]
