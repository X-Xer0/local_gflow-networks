from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from .env import FeatureSelectionEnv
from .local_search import LocalSearchRefiner
from .model import GFlowNetModel, masked_log_softmax
from .plotting import plot_comparison_bars, plot_reward_curves
from .replay_buffer import RewardPrioritizedReplay, TrajectoryRecord
from .reward import RewardModel
from .sampling import sample_forward_trajectory
from .utils import ensure_dir, jaccard_distance, save_json


@dataclass
class TrainerConfig:
    method: str = "baseline"
    epochs: int = 40
    warmup_random_trajectories: int = 256
    rollouts_per_epoch: int = 64
    grad_steps_per_epoch: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    hidden_dim: int = 128
    buffer_capacity: int = 5000
    replay_alpha: float = 0.8
    initial_epsilon: float = 0.25
    min_epsilon: float = 0.05
    epsilon_decay: float = 0.97
    grad_clip_norm: float = 5.0
    local_search_steps: int = 3
    local_search_backtrack: int = 2
    sample_temperature: float = 1.0
    eval_samples: int = 200
    seed: int = 42
    device: str = "cpu"
    output_dir: str = "outputs"


def random_order(env: FeatureSelectionEnv, rng: np.random.Generator) -> List[int]:
    return rng.choice(env.num_features, size=env.subset_size, replace=False).tolist()


def trajectory_balance_loss(
    model: GFlowNetModel,
    env: FeatureSelectionEnv,
    batch: List[TrajectoryRecord],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    residuals: List[torch.Tensor] = []

    for record in batch:
        state = env.initial_state()
        log_pf = torch.tensor(0.0, device=device)
        log_pb = torch.tensor(0.0, device=device)

        for action in record.order:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            forward_logits = model.forward_logits(state_tensor)[0]
            forward_mask = torch.tensor((state < 0.5).astype(np.float32), device=device)
            forward_log_probs = masked_log_softmax(forward_logits, forward_mask)
            log_pf = log_pf + forward_log_probs[action]

            next_state = env.step_forward(state, action)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            backward_logits = model.backward_logits(next_state_tensor)[0]
            backward_mask = torch.tensor((next_state > 0.5).astype(np.float32), device=device)
            backward_log_probs = masked_log_softmax(backward_logits, backward_mask)
            log_pb = log_pb + backward_log_probs[action]
            state = next_state

        log_reward = torch.tensor(record.metrics.log_reward, dtype=torch.float32, device=device)
        residual = model.logZ + log_pf - log_reward - log_pb
        residuals.append(residual)

    residual_tensor = torch.stack(residuals)
    loss = (residual_tensor ** 2).mean()
    info = {
        "tb_residual_mean": float(residual_tensor.mean().detach().cpu().item()),
        "tb_residual_abs_mean": float(residual_tensor.abs().mean().detach().cpu().item()),
    }
    return loss, info


@torch.no_grad()
def evaluate_policy(
    model: GFlowNetModel,
    env: FeatureSelectionEnv,
    scorer: RewardModel,
    device: torch.device,
    rng: np.random.Generator,
    num_samples: int = 200,
    greedy: bool = False,
) -> Dict[str, object]:
    records: List[TrajectoryRecord] = []
    for _ in range(num_samples):
        order = sample_forward_trajectory(
            model=model,
            env=env,
            device=device,
            rng=rng,
            epsilon=0.0,
            temperature=1.0,
            greedy=greedy,
        )
        metrics = scorer.evaluate(order)
        records.append(TrajectoryRecord(order=order, metrics=metrics, source="eval"))

    rewards = np.array([r.metrics.reward for r in records], dtype=np.float64)
    val_scores = np.array([r.metrics.val_balanced_accuracy for r in records], dtype=np.float64)
    test_scores = np.array([r.metrics.test_balanced_accuracy for r in records], dtype=np.float64)
    unique_subsets = {tuple(sorted(r.order)) for r in records}

    pairwise = []
    compare_records = records[: min(len(records), 100)]
    for i in range(len(compare_records)):
        for j in range(i + 1, len(compare_records)):
            pairwise.append(jaccard_distance(compare_records[i].order, compare_records[j].order))
    diversity = float(np.mean(pairwise)) if pairwise else 0.0

    best_record = max(records, key=lambda r: r.metrics.reward)

    return {
        "avg_reward": float(rewards.mean()),
        "best_reward": float(rewards.max()),
        "avg_val_balanced_accuracy": float(val_scores.mean()),
        "best_val_balanced_accuracy": float(val_scores.max()),
        "avg_test_balanced_accuracy": float(test_scores.mean()),
        "best_test_balanced_accuracy": float(test_scores.max()),
        "unique_ratio": float(len(unique_subsets) / max(len(records), 1)),
        "avg_pairwise_jaccard_distance": diversity,
        "best_subset": list(best_record.metrics.subset),
        "best_subset_val_balanced_accuracy": float(best_record.metrics.val_balanced_accuracy),
        "best_subset_test_balanced_accuracy": float(best_record.metrics.test_balanced_accuracy),
    }


def save_checkpoint(
    path: Path,
    model: GFlowNetModel,
    config: TrainerConfig,
    env: FeatureSelectionEnv,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "trainer_config": asdict(config),
            "num_features": env.num_features,
            "subset_size": env.subset_size,
        },
        path,
    )


def train_gflownet(
    env: FeatureSelectionEnv,
    scorer: RewardModel,
    config: TrainerConfig,
) -> Dict[str, object]:
    output_dir = ensure_dir(Path(config.output_dir) / config.method)
    device = torch.device(config.device)
    rng = np.random.default_rng(config.seed)

    model = GFlowNetModel(num_features=env.num_features, hidden_dim=config.hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    buffer = RewardPrioritizedReplay(
        capacity=config.buffer_capacity,
        alpha=config.replay_alpha,
        seed=config.seed,
    )

    if config.method == "local_search":
        refiner = LocalSearchRefiner(
            env=env,
            scorer=scorer,
            device=device,
            rng=rng,
            max_backtrack=config.local_search_backtrack,
            temperature=config.sample_temperature,
        )
    else:
        refiner = None

    for _ in range(config.warmup_random_trajectories):
        order = random_order(env, rng)
        metrics = scorer.evaluate(order)
        buffer.add(TrajectoryRecord(order=order, metrics=metrics, source="warmup"))

    history: List[Dict[str, float]] = []
    best_buffer_reward = -float("inf")

    for epoch in range(1, config.epochs + 1):
        model.eval()
        epsilon = max(config.min_epsilon, config.initial_epsilon * (config.epsilon_decay ** (epoch - 1)))
        sampled_rewards = []
        local_rewards = []
        local_accepts = 0
        local_attempts = 0

        for _ in range(config.rollouts_per_epoch):
            order = sample_forward_trajectory(
                model=model,
                env=env,
                device=device,
                rng=rng,
                epsilon=epsilon,
                temperature=config.sample_temperature,
                greedy=False,
            )
            metrics = scorer.evaluate(order)
            buffer.add(TrajectoryRecord(order=order, metrics=metrics, source="sampled"))
            sampled_rewards.append(metrics.reward)

            if refiner is not None:
                current_order = order
                current_metrics = metrics
                for _ in range(config.local_search_steps):
                    result = refiner.refine(model, current_order, current_metrics)
                    local_attempts += 1
                    local_rewards.append(result.proposed_metrics.reward)
                    buffer.add(
                        TrajectoryRecord(
                            order=result.proposed_order,
                            metrics=result.proposed_metrics,
                            source="local_search",
                            accepted=result.accepted,
                            metadata={
                                "backtrack_steps": result.backtrack_steps,
                                "removed_features": result.removed_features,
                            },
                        )
                    )
                    if result.accepted:
                        local_accepts += 1
                        current_order = result.proposed_order
                        current_metrics = result.proposed_metrics

        model.train()
        losses = []
        residual_abs = []
        for _ in range(config.grad_steps_per_epoch):
            batch = buffer.sample(config.batch_size)
            optimizer.zero_grad()
            loss, info = trajectory_balance_loss(model, env, batch, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            residual_abs.append(info["tb_residual_abs_mean"])

        eval_metrics = evaluate_policy(
            model=model,
            env=env,
            scorer=scorer,
            device=device,
            rng=rng,
            num_samples=config.eval_samples,
        )
        best_record = buffer.best()
        assert best_record is not None
        best_buffer_reward = max(best_buffer_reward, best_record.metrics.reward)

        epoch_row = {
            "epoch": epoch,
            "epsilon": epsilon,
            "train_loss": float(np.mean(losses)),
            "tb_residual_abs_mean": float(np.mean(residual_abs)),
            "sampled_avg_reward": float(np.mean(sampled_rewards)) if sampled_rewards else 0.0,
            "sampled_best_reward": float(np.max(sampled_rewards)) if sampled_rewards else 0.0,
            "local_avg_reward": float(np.mean(local_rewards)) if local_rewards else 0.0,
            "local_accept_rate": float(local_accepts / max(local_attempts, 1)),
            "buffer_best_reward": float(best_record.metrics.reward),
            "buffer_best_val_balanced_accuracy": float(best_record.metrics.val_balanced_accuracy),
            "eval_avg_reward": float(eval_metrics["avg_reward"]),
            "eval_best_reward": float(eval_metrics["best_reward"]),
            "eval_unique_ratio": float(eval_metrics["unique_ratio"]),
            "eval_avg_pairwise_jaccard_distance": float(eval_metrics["avg_pairwise_jaccard_distance"]),
            "eval_avg_val_balanced_accuracy": float(eval_metrics["avg_val_balanced_accuracy"]),
            "eval_best_val_balanced_accuracy": float(eval_metrics["best_val_balanced_accuracy"]),
            "eval_avg_test_balanced_accuracy": float(eval_metrics["avg_test_balanced_accuracy"]),
        }
        history.append(epoch_row)

    save_checkpoint(output_dir / "model.pt", model, config, env)
    save_json(output_dir / "history.json", {"history": history})

    plot_reward_curves(history=history, save_dir=output_dir, method_name=config.method)

    top_records = buffer.topk(10)
    final_eval = evaluate_policy(
        model=model,
        env=env,
        scorer=scorer,
        device=device,
        rng=rng,
        num_samples=max(config.eval_samples, 300),
    )
    results = {
        "method": config.method,
        "history": history,
        "final_eval": final_eval,
        "best_buffer_reward": best_buffer_reward,
        "top_subsets": [
            {
                "subset": list(rec.metrics.subset),
                "reward": rec.metrics.reward,
                "val_balanced_accuracy": rec.metrics.val_balanced_accuracy,
                "test_balanced_accuracy": rec.metrics.test_balanced_accuracy,
                "source": rec.source,
            }
            for rec in top_records
        ],
        "checkpoint": str(output_dir / "model.pt"),
    }
    save_json(output_dir / "results.json", results)
    return results


def compare_and_plot(results_by_method: Dict[str, Dict[str, object]], output_root: str | Path) -> None:
    output_root = ensure_dir(output_root)
    plot_comparison_bars(results_by_method, output_root)
    save_json(Path(output_root) / "comparison.json", results_by_method)
