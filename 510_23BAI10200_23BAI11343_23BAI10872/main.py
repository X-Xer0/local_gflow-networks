from __future__ import annotations

import argparse
from pathlib import Path

import torch

from gfn_fs.data import DatasetBuilder, subset_to_feature_names
from gfn_fs.env import FeatureSelectionConfig, FeatureSelectionEnv
from gfn_fs.reward import RewardModel
from gfn_fs.train import TrainerConfig, compare_and_plot, train_gflownet
from gfn_fs.utils import ensure_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vanilla GFlowNet and LS-GFlowNet for feature subset selection.")
    parser.add_argument("--method", choices=["baseline", "local_search", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--candidate-features", type=int, default=15)
    parser.add_argument("--subset-size", type=int, default=6)
    parser.add_argument("--rollouts-per-epoch", type=int, default=64)
    parser.add_argument("--grad-steps-per-epoch", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-random-trajectories", type=int, default=256)
    parser.add_argument("--local-search-steps", type=int, default=3)
    parser.add_argument("--local-search-backtrack", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-scale", type=float, default=5.0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-exhaustive", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_root = ensure_dir(args.output_dir)

    dataset = DatasetBuilder(candidate_features=args.candidate_features, seed=args.seed).build()
    env = FeatureSelectionEnv(
        FeatureSelectionConfig(
            num_features=len(dataset.feature_names),
            subset_size=args.subset_size,
        )
    )
    scorer = RewardModel(dataset, reward_scale=args.reward_scale, seed=args.seed)

    methods = [args.method] if args.method != "both" else ["baseline", "local_search"]
    all_results = {}

    for method in methods:
        train_config = TrainerConfig(
            method=method,
            epochs=args.epochs,
            warmup_random_trajectories=args.warmup_random_trajectories,
            rollouts_per_epoch=args.rollouts_per_epoch,
            grad_steps_per_epoch=args.grad_steps_per_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            local_search_steps=args.local_search_steps,
            local_search_backtrack=args.local_search_backtrack,
            eval_samples=args.eval_samples,
            seed=args.seed,
            device=args.device,
            output_dir=str(output_root),
        )
        result = train_gflownet(env=env, scorer=scorer, config=train_config)

        for top in result["top_subsets"]:
            top["feature_names"] = subset_to_feature_names(top["subset"], dataset.feature_names)
        result["final_eval"]["best_subset_feature_names"] = subset_to_feature_names(
            result["final_eval"]["best_subset"], dataset.feature_names
        )
        all_results[method] = result
        print(f"\n=== {method.upper()} ===")
        print(f"Best eval reward: {result['final_eval']['best_reward']:.4f}")
        print(f"Avg eval reward:  {result['final_eval']['avg_reward']:.4f}")
        print(f"Unique ratio:     {result['final_eval']['unique_ratio']:.4f}")
        print(f"Best subset:      {result['final_eval']['best_subset_feature_names']}")
        print(f"Best val bal acc: {result['final_eval']['best_subset_val_balanced_accuracy']:.4f}")
        print(f"Best test bal acc:{result['final_eval']['best_subset_test_balanced_accuracy']:.4f}")

    if args.run_exhaustive:
        best_exact = scorer.exhaustive_search(args.subset_size)
        exhaustive_payload = {
            "best_subset": list(best_exact.subset),
            "best_subset_feature_names": subset_to_feature_names(best_exact.subset, dataset.feature_names),
            "reward": best_exact.reward,
            "val_balanced_accuracy": best_exact.val_balanced_accuracy,
            "test_balanced_accuracy": best_exact.test_balanced_accuracy,
        }
        save_json(Path(output_root) / "exhaustive_search.json", exhaustive_payload)
        print("\n=== EXHAUSTIVE SEARCH REFERENCE ===")
        print(f"Best subset:      {exhaustive_payload['best_subset_feature_names']}")
        print(f"Best reward:      {exhaustive_payload['reward']:.4f}")
        print(f"Val bal acc:      {exhaustive_payload['val_balanced_accuracy']:.4f}")
        print(f"Test bal acc:     {exhaustive_payload['test_balanced_accuracy']:.4f}")

    compare_and_plot(all_results, output_root)
    save_json(Path(output_root) / "all_results.json", all_results)
    print(f"\nSaved outputs to: {Path(output_root).resolve()}")


if __name__ == "__main__":
    main()
