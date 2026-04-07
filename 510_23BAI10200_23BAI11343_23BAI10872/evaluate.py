from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gfn_fs.data import DatasetBuilder, subset_to_feature_names
from gfn_fs.env import FeatureSelectionConfig, FeatureSelectionEnv
from gfn_fs.model import GFlowNetModel
from gfn_fs.reward import RewardModel
from gfn_fs.train import evaluate_policy
from gfn_fs.utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GFlowNet checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--candidate-features", type=int, default=15)
    parser.add_argument("--subset-size", type=int, default=6)
    parser.add_argument("--eval-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-exhaustive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ckpt = torch.load(args.checkpoint, map_location=args.device)

    dataset = DatasetBuilder(candidate_features=args.candidate_features, seed=args.seed).build()
    env = FeatureSelectionEnv(
        FeatureSelectionConfig(
            num_features=len(dataset.feature_names),
            subset_size=args.subset_size,
        )
    )
    scorer = RewardModel(dataset, reward_scale=args.reward_scale, seed=args.seed)

    hidden_dim = ckpt.get("trainer_config", {}).get("hidden_dim", 128)
    model = GFlowNetModel(num_features=env.num_features, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    model.eval()

    rng = torch.Generator()
    del rng

    import numpy as np

    eval_metrics = evaluate_policy(
        model=model,
        env=env,
        scorer=scorer,
        device=torch.device(args.device),
        rng=np.random.default_rng(args.seed),
        num_samples=args.eval_samples,
    )
    eval_metrics["best_subset_feature_names"] = subset_to_feature_names(
        eval_metrics["best_subset"], dataset.feature_names
    )

    payload = {"evaluation": eval_metrics}

    if args.run_exhaustive:
        best_exact = scorer.exhaustive_search(args.subset_size)
        payload["exhaustive_reference"] = {
            "best_subset": list(best_exact.subset),
            "best_subset_feature_names": subset_to_feature_names(best_exact.subset, dataset.feature_names),
            "reward": best_exact.reward,
            "val_balanced_accuracy": best_exact.val_balanced_accuracy,
            "test_balanced_accuracy": best_exact.test_balanced_accuracy,
        }

    out_path = Path(args.checkpoint).parent / "evaluation.json"
    save_json(out_path, payload)

    print("=== EVALUATION ===")
    print(f"Avg reward:       {eval_metrics['avg_reward']:.4f}")
    print(f"Best reward:      {eval_metrics['best_reward']:.4f}")
    print(f"Unique ratio:     {eval_metrics['unique_ratio']:.4f}")
    print(f"Best subset:      {eval_metrics['best_subset_feature_names']}")
    print(f"Val bal acc:      {eval_metrics['best_subset_val_balanced_accuracy']:.4f}")
    print(f"Test bal acc:     {eval_metrics['best_subset_test_balanced_accuracy']:.4f}")
    print(f"Saved to:         {out_path.resolve()}")

    if args.run_exhaustive and "exhaustive_reference" in payload:
        ref = payload["exhaustive_reference"]
        print("\n=== EXHAUSTIVE REFERENCE ===")
        print(f"Best subset:      {ref['best_subset_feature_names']}")
        print(f"Reward:           {ref['reward']:.4f}")
        print(f"Val bal acc:      {ref['val_balanced_accuracy']:.4f}")
        print(f"Test bal acc:     {ref['test_balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()
