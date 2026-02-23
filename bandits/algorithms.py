"""
bandits/algorithms.py — UCB1, EpsilonGreedy, and simulation utilities.

ThompsonSampling lives in bandits/thompson.py (its natural home) and is
re-exported here for backward compatibility:

    from bandits.algorithms import ThompsonSampling  # works
    from bandits.thompson   import ThompsonSampling  # also works
"""

import numpy as np
from typing import List, Optional

# Re-export ThompsonSampling so any existing code importing it from here
# continues to work without changes.
from bandits.thompson import ThompsonSampling  # noqa: F401  (re-export)


class UCB1:
    """
    UCB1 (Upper Confidence Bound) bandit algorithm.

    Balances exploration and exploitation by adding a confidence bonus to each
    arm's mean reward. The bonus grows for arms that have been pulled rarely
    relative to the total number of pulls, ensuring every arm is eventually
    explored.

    Args:
        n_arms: Number of arms / variants.
        seed:   Optional random seed (used only if random tie-breaking is needed).
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None):
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)
        self.pulls = np.zeros(n_arms, dtype=float)
        self.rewards = np.zeros(n_arms, dtype=float)
        self.total_pulls = 0

    def select_arm(self) -> int:
        """
        Select the arm with the highest UCB score.

        During the warm-up phase (total pulls < n_arms) each arm is pulled
        exactly once in order to get an initial estimate before computing UCBs.

        Returns:
            Index of the selected arm.
        """
        # Warm-up: pull each arm once before computing UCB values
        if self.total_pulls < self.n_arms:
            return self.total_pulls

        means  = self.rewards / np.maximum(self.pulls, 1)
        bonus  = np.sqrt(2 * np.log(self.total_pulls) / np.maximum(self.pulls, 1))
        return int(np.argmax(means + bonus))

    def update(self, arm: int, reward: float) -> None:
        """
        Record the result of pulling an arm.

        Args:
            arm:    Index of the arm that was pulled.
            reward: Observed reward (0 or 1 for Bernoulli).
        """
        self.pulls[arm]   += 1
        self.rewards[arm] += reward
        self.total_pulls  += 1

    def get_best_arm(self) -> int:
        """Return the arm with the highest empirical mean reward."""
        means = self.rewards / np.maximum(self.pulls, 1)
        return int(np.argmax(means))

    def __repr__(self) -> str:
        return f"UCB1(n_arms={self.n_arms}, total_pulls={self.total_pulls})"


class EpsilonGreedy:
    """
    Epsilon-Greedy bandit algorithm.

    With probability epsilon the algorithm explores (pulls a random arm).
    With probability 1 - epsilon it exploits (pulls the current best arm).

    Args:
        n_arms:  Number of arms / variants.
        epsilon: Exploration probability in [0, 1]. Default 0.1 (10% explore).
        seed:    Optional random seed for reproducibility.
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: Optional[int] = None):
        self.n_arms  = n_arms
        self.epsilon = epsilon
        self._rng    = np.random.default_rng(seed)
        self.pulls   = np.zeros(n_arms, dtype=float)
        self.rewards = np.zeros(n_arms, dtype=float)

    def select_arm(self) -> int:
        """
        Select an arm using epsilon-greedy strategy.

        Returns:
            Index of the arm to pull.
        """
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_arms))   # explore
        means = self.rewards / np.maximum(self.pulls, 1)
        return int(np.argmax(means))                       # exploit

    def update(self, arm: int, reward: float) -> None:
        """
        Record the result of pulling an arm.

        Args:
            arm:    Index of the arm that was pulled.
            reward: Observed reward.
        """
        self.pulls[arm]   += 1
        self.rewards[arm] += reward

    def get_best_arm(self) -> int:
        """Return the arm with the highest empirical mean reward."""
        means = self.rewards / np.maximum(self.pulls, 1)
        return int(np.argmax(means))

    def __repr__(self) -> str:
        return f"EpsilonGreedy(n_arms={self.n_arms}, epsilon={self.epsilon})"


# ── Simulation utility ─────────────────────────────────────────────────────────

def simulate_test(
    true_rates: List[float],
    n_trials: int,
    algorithm: str = "thompson",
    seed: Optional[int] = None,
) -> dict:
    """
    Simulate an A/B test using the specified bandit algorithm.

    Args:
        true_rates: True conversion rate for each variant (arm).
        n_trials:   Number of trials (user visits / observations) to simulate.
        algorithm:  One of 'thompson', 'ucb', or 'epsilon'.
        seed:       Optional random seed for reproducibility. Passed to both
                    the bandit and the reward-generation RNG.

    Returns:
        Dictionary with keys:
            algorithm        - algorithm name used
            best_arm         - arm the bandit converged on
            true_best        - arm with the highest true rate
            pulls            - array of pull counts per arm
            rewards          - array of total rewards per arm
            conversion_rates - empirical conversion rate per arm
            regret           - cumulative regret (scalar float)
            regret_pct       - regret as percentage of total trials
    """
    rng = np.random.default_rng(seed)
    n_arms = len(true_rates)
    true_rates_arr = np.array(true_rates)

    # Initialise bandit
    if algorithm == "thompson":
        bandit = ThompsonSampling(n_arms, seed=seed)
    elif algorithm == "ucb":
        bandit = UCB1(n_arms, seed=seed)
    else:
        bandit = EpsilonGreedy(n_arms, seed=seed)

    # Run simulation
    for _ in range(n_trials):
        arm    = bandit.select_arm()
        reward = 1 if rng.random() < true_rates_arr[arm] else 0
        bandit.update(arm, reward)

    # Cumulative regret = rewards we would have earned pulling the best arm
    # every trial, minus the rewards we actually earned.
    best_rate     = float(np.max(true_rates_arr))
    total_rewards = float(np.sum(bandit.rewards))
    regret        = best_rate * n_trials - total_rewards

    return {
        "algorithm":        algorithm,
        "best_arm":         bandit.get_best_arm(),
        "true_best":        int(np.argmax(true_rates_arr)),
        "pulls":            bandit.pulls,
        "rewards":          bandit.rewards,
        "conversion_rates": bandit.rewards / np.maximum(bandit.pulls, 1),
        "regret":           regret,
        "regret_pct":       (regret / n_trials) * 100,
    }


if __name__ == "__main__":
    print("Multi-Armed Bandit Demo")
    print("=" * 50)

    results = simulate_test(
        true_rates=[0.10, 0.12],  # Variant B is 20% better
        n_trials=10_000,
        algorithm="thompson",
        seed=42,
    )

    print(f"\nResults after {10_000:,} trials:")
    print(f"  Detected best arm : Variant {results['best_arm']}")
    print(f"  True best arm     : Variant {results['true_best']}")
    print(f"  Pulls             : {results['pulls']}")
    print(f"  Conversion rates  : {[round(r, 4) for r in results['conversion_rates']]}")
    print(f"  Cumulative regret : {results['regret']:.1f} ({results['regret_pct']:.2f}%)")
