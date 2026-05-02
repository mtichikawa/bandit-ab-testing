"""
Smoke tests for the multi-armed bandit algorithms.

Coverage:
- Each algorithm constructs without error
- select_arm() returns a valid arm index
- update() does not crash and updates internal state
- get_best_arm() returns a valid index after observations
- simulate_test() runs end-to-end and returns the expected keys
- Reproducibility: same seed produces the same trajectory
- High-true-rate arm wins enough of the time

These are smoke tests, not statistical guarantees. They protect against
regressions in the public API and the simulation harness, not the
algorithms' theoretical properties.
"""

from __future__ import annotations

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bandits.algorithms import EpsilonGreedy, UCB1, simulate_test
from bandits.thompson import ThompsonSampling


# ── Construction ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (UCB1, {"n_arms": 4, "seed": 42}),
        (EpsilonGreedy, {"n_arms": 4, "epsilon": 0.1, "seed": 42}),
        (ThompsonSampling, {"n_arms": 4, "seed": 42}),
    ],
)
def test_constructs(cls, kwargs):
    bandit = cls(**kwargs)
    assert bandit.n_arms == 4


# ── select_arm returns valid index ────────────────────────────────────────────

@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (UCB1, {"n_arms": 4, "seed": 1}),
        (EpsilonGreedy, {"n_arms": 4, "epsilon": 0.1, "seed": 1}),
        (ThompsonSampling, {"n_arms": 4, "seed": 1}),
    ],
)
def test_select_arm_returns_valid_index(cls, kwargs):
    bandit = cls(**kwargs)
    arm = bandit.select_arm()
    assert isinstance(arm, int)
    assert 0 <= arm < kwargs["n_arms"]


# ── update + get_best_arm ─────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (UCB1, {"n_arms": 3, "seed": 1}),
        (EpsilonGreedy, {"n_arms": 3, "epsilon": 0.1, "seed": 1}),
        (ThompsonSampling, {"n_arms": 3, "seed": 1}),
    ],
)
def test_update_and_best_arm(cls, kwargs):
    bandit = cls(**kwargs)
    # Arm 1 always wins
    for _ in range(50):
        bandit.update(0, 0)
        bandit.update(1, 1)
        bandit.update(2, 0)
    best = bandit.get_best_arm()
    assert best == 1


# ── simulate_test end-to-end ──────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", ["thompson", "ucb", "epsilon"])
def test_simulate_test_returns_expected_keys(algorithm):
    result = simulate_test(
        true_rates=[0.1, 0.5, 0.3],
        n_trials=200,
        algorithm=algorithm,
        seed=42,
    )
    expected_keys = {
        "algorithm",
        "best_arm",
        "true_best",
        "pulls",
        "rewards",
        "conversion_rates",
        "regret",
        "regret_pct",
    }
    assert expected_keys.issubset(set(result.keys()))
    assert result["algorithm"] == algorithm
    assert result["true_best"] == 1
    assert len(result["pulls"]) == 3
    assert sum(result["pulls"]) == 200


# ── Reproducibility ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("algorithm", ["thompson", "ucb", "epsilon"])
def test_simulate_test_is_reproducible_with_seed(algorithm):
    a = simulate_test([0.1, 0.5, 0.3], n_trials=100, algorithm=algorithm, seed=7)
    b = simulate_test([0.1, 0.5, 0.3], n_trials=100, algorithm=algorithm, seed=7)
    # pulls and rewards arrays should match exactly with the same seed
    assert list(a["pulls"]) == list(b["pulls"])
    assert list(a["rewards"]) == list(b["rewards"])


# ── Correctness signal: high-rate arm wins enough ─────────────────────────────

@pytest.mark.parametrize("algorithm", ["thompson", "ucb", "epsilon"])
def test_high_rate_arm_gets_majority_of_pulls(algorithm):
    # Strong contrast — should be obvious to all three algorithms
    result = simulate_test(
        true_rates=[0.05, 0.7],
        n_trials=500,
        algorithm=algorithm,
        seed=42,
    )
    pulls = result["pulls"]
    # Best arm should be picked majority of the time
    assert pulls[1] > pulls[0]


def test_simulate_test_unknown_algorithm_falls_back_to_epsilon():
    # Per current implementation, unknown algo falls back to epsilon-greedy
    result = simulate_test(
        true_rates=[0.3, 0.4],
        n_trials=50,
        algorithm="not-a-real-algo",
        seed=1,
    )
    # Just verify it ran and produced sane output
    assert sum(result["pulls"]) == 50
    assert 0 <= result["best_arm"] < 2
