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
python examples/quick_demo.py
```

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
- Practical production considerations

Contact: Mike Ichikawa - projects.ichikawa@gmail.com

# Updated: 2025-09-10
# Updated: 2025-10-20
# Updated: 2025-10-26
# Updated: 2025-11-01
# Updated: 2025-11-06
# Updated: 2025-11-12
# Updated: 2025-11-20
# Updated: 2025-11-26
# Updated: 2025-12-02
# Updated: 2025-12-08