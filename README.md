# Multi-Armed Bandit A/B Testing Framework

Adaptive A/B testing using Thompson Sampling, UCB1, and Epsilon-Greedy algorithms.

## Why Use Bandits?

Traditional A/B testing splits traffic 50/50 and waits for statistical significance. Bandits dynamically shift more traffic to better-performing variants, reducing regret (lost conversions).

## Features

- **Thompson Sampling** - Bayesian approach, best overall
- **UCB1** - Deterministic with theoretical guarantees
- **Epsilon-Greedy** - Simple baseline
- Simulation framework for comparison
- Regret analysis

## Quick Start

```bash
pip install -r requirements.txt
cd examples && python quick_demo.py
```

The demo adds the project root to the import path with a relative path, so run it from `examples/`. An interactive Streamlit version is described in `docs/streamlit_demo.md`.

## Usage

```python
from bandits.algorithms import ThompsonSampling

# Initialize test with 2 variants
bandit = ThompsonSampling(n_arms=2)

# For each user:
variant = bandit.select_arm()  # 0 or 1
# Show variant to user...
# User converts or not:
bandit.update(variant, reward=1)  # or 0

# Get results
best = bandit.get_best_arm()
probs = bandit.get_probabilities()
```

## What I Learned

- Bayesian statistics and Beta distributions
- Exploration vs exploitation tradeoff
- Regret minimization in online learning

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

> _Note: regret curves are most useful when compared across algorithms on identical reward streams. A single curve in isolation can mislead. Averaged over 200 seeds of `simulate_test` (rates 0.10/0.12 and 0.05/0.10/0.15), Thompson Sampling had the lowest or near-lowest mean regret at every horizon from 25 to 2,000 trials, and UCB1 fell further behind as trials grew._
