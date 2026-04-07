from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def plot_reward_curves(history: List[Dict[str, float]], save_dir: str | Path, method_name: str) -> None:
    save_dir = ensure_dir(save_dir)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["eval_avg_reward"] for row in history], label="Eval avg reward")
    plt.plot(epochs, [row["eval_best_reward"] for row in history], label="Eval best reward")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")
    plt.title(f"Reward curves - {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "reward_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["eval_avg_val_balanced_accuracy"] for row in history], label="Eval avg val bal acc")
    plt.plot(epochs, [row["eval_best_val_balanced_accuracy"] for row in history], label="Eval best val bal acc")
    plt.plot(epochs, [row["eval_avg_test_balanced_accuracy"] for row in history], label="Eval avg test bal acc")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title(f"Accuracy curves - {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "accuracy_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["eval_unique_ratio"] for row in history], label="Unique ratio")
    plt.plot(epochs, [row["eval_avg_pairwise_jaccard_distance"] for row in history], label="Pairwise Jaccard distance")
    plt.xlabel("Epoch")
    plt.ylabel("Diversity")
    plt.title(f"Diversity curves - {method_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "diversity_curves.png", dpi=150)
    plt.close()


def plot_comparison_bars(results_by_method: Dict[str, Dict[str, object]], save_dir: str | Path) -> None:
    save_dir = ensure_dir(save_dir)
    methods = list(results_by_method.keys())
    avg_rewards = [results_by_method[m]["final_eval"]["avg_reward"] for m in methods]
    best_rewards = [results_by_method[m]["final_eval"]["best_reward"] for m in methods]
    diversity = [results_by_method[m]["final_eval"]["unique_ratio"] for m in methods]
    avg_test = [results_by_method[m]["final_eval"]["avg_test_balanced_accuracy"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, avg_rewards, width, label="Avg reward")
    plt.bar(x + width / 2, best_rewards, width, label="Best reward")
    plt.xticks(x, methods)
    plt.ylabel("Reward")
    plt.title("Reward comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "comparison_rewards.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, diversity, width, label="Unique ratio")
    plt.bar(x + width / 2, avg_test, width, label="Avg test balanced accuracy")
    plt.xticks(x, methods)
    plt.ylabel("Value")
    plt.title("Diversity and generalization comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "comparison_diversity_accuracy.png", dpi=150)
    plt.close()
