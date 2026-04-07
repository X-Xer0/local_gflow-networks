# Local Search GFlowNets for Feature Subset Selection

A research prototype inspired by **Local Search GFlowNets (ICLR 2024)** and **Trajectory Balance**.

This project turns the paper idea into a complete, runnable machine learning project for **feature subset selection** on a real tabular dataset.

## Project idea

We want to learn a policy that **samples good feature subsets** instead of only finding one greedy best subset.

- **Vanilla GFlowNet** learns to sample subsets with probability roughly proportional to reward.
- **Local Search GFlowNet** improves this by taking sampled subsets, partially undoing them, rebuilding them, and keeping better local refinements.

That gives us a nice balance between:
- **global exploration** (build new subsets from scratch), and
- **local exploitation** (improve promising subsets nearby).

## Chosen application

### Feature subset selection for machine learning

Given a dataset with many input features, choose a subset of exactly `K` features that gives strong predictive performance.

This is a natural fit for GFlowNets because:
- the search space is combinatorial,
- many different subsets can be good,
- diversity matters because multiple good subsets may exist,
- the reward of a complete subset is only known at the end.

## Simplifications used

To keep the project runnable on a laptop while still research-like:

1. We use the **Breast Cancer Wisconsin** dataset from scikit-learn.
2. We first reduce the search to the top `candidate_features` features ranked by mutual information.
3. Then the GFlowNet searches over **subsets of fixed size** `subset_size`.

This avoids variable-length terminal handling and makes Trajectory Balance much easier to understand and implement.

## State, action, reward, terminal condition

### State
A binary mask of length `D`:
- `1` means the feature is selected
- `0` means it is not selected

### Action
Pick one currently unselected feature and add it.

### Terminal condition
The episode ends when exactly `K` features have been selected.

### Reward
For a terminal subset:
1. train a logistic regression classifier using only those features,
2. score it on the validation set using balanced accuracy,
3. convert the score into a positive GFlowNet reward:

```text
R(x) = exp(reward_scale * validation_balanced_accuracy(x))
```

## Folder structure

```text
local_search_gflnet_feature_selection/
├── main.py
├── evaluate.py
├── requirements.txt
├── README.md
└── gfn_fs/
    ├── __init__.py
    ├── data.py
    ├── env.py
    ├── reward.py
    ├── replay_buffer.py
    ├── model.py
    ├── sampling.py
    ├── local_search.py
    ├── plotting.py
    ├── train.py
    └── utils.py
```

## Install

```bash
pip install -r requirements.txt
```

## Train both methods

```bash
python main.py --method both --epochs 30 --candidate-features 15 --subset-size 6 --run-exhaustive
```

## Train only baseline

```bash
python main.py --method baseline --epochs 30
```

## Train only Local Search GFlowNet

```bash
python main.py --method local_search --epochs 30
```

## Evaluate a saved checkpoint

```bash
python evaluate.py --checkpoint outputs/baseline/model.pt --candidate-features 15 --subset-size 6 --eval-samples 500 --run-exhaustive
```

## Outputs created

For each method, the code saves:
- `model.pt`
- `history.json`
- `results.json`
- `reward_curves.png`
- `accuracy_curves.png`
- `diversity_curves.png`

And at the root output directory:
- `comparison.json`
- `comparison_rewards.png`
- `comparison_diversity_accuracy.png`
- optionally `exhaustive_search.json`

## Example runs

These short runs were produced from this codebase.

### Baseline example

```text
=== BASELINE ===
Best eval reward: 121.6339
Avg eval reward:  111.9897
Unique ratio:     0.9533
Best subset:      ['mean radius', 'mean area', 'mean compactness', 'area error', 'worst compactness', 'worst concavity']
Best val bal acc: 0.9602
Best test bal acc:0.9296
```

### Local Search example

```text
=== LOCAL_SEARCH ===
Best eval reward: 123.1371
Avg eval reward:  112.2526
Unique ratio:     0.9533
Best subset:      ['mean area', 'mean compactness', 'mean concave points', 'worst area', 'worst concavity', 'worst concave points']
Best val bal acc: 0.9627
Best test bal acc:0.9464
```

In this short demonstration, the local-search variant found a slightly better best subset.

## How the algorithms work

### Vanilla GFlowNet
1. Start from the empty subset.
2. Add one feature at a time using the learned forward policy.
3. When the subset reaches size `K`, compute reward.
4. Store trajectories in replay.
5. Train with the Trajectory Balance loss.

### Local Search GFlowNet
1. Sample a terminal subset using the forward policy.
2. Use the backward policy to remove a few selected features.
3. Use the forward policy to refill the subset.
4. Accept the new subset if its reward is better or equal.
5. Train on both sampled and refined trajectories.

## Trajectory Balance in one line

For a trajectory `tau = (s0 -> s1 -> ... -> sT)` ending at subset `x`, the model tries to satisfy:

```text
log Z + sum_t log P_F(a_t | s_t) = log R(x) + sum_t log P_B(s_{t-1} | s_t)
```

The code minimizes the squared residual of this equation.

## Suggested report/demo talking points

- Why feature selection is combinatorial.
- Why RL alone is not enough when we want many diverse good solutions.
- Why reward is only available at the terminal subset.
- How GFlowNets differ from standard RL maximization.
- How local search adds exploitation around promising subsets.
- Why diversity metrics matter, not only best reward.

## Quick run order

1. Install requirements.
2. Run `python main.py --method both --epochs 30 --run-exhaustive`.
3. Inspect plots and `results.json` files.
4. Run `evaluate.py` on the saved checkpoints for extra sampling.

