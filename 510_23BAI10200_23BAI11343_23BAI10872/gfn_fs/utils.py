from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def to_numpy_mask(selected: Sequence[int], num_features: int) -> np.ndarray:
    mask = np.zeros(num_features, dtype=np.float32)
    if len(selected) > 0:
        mask[list(selected)] = 1.0
    return mask


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def jaccard_distance(a: Sequence[int], b: Sequence[int]) -> float:
    sa, sb = set(a), set(b)
    union = sa | sb
    inter = sa & sb
    if not union:
        return 0.0
    return 1.0 - len(inter) / len(union)
