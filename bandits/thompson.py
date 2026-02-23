"""
bandits/thompson.py — Thompson Sampling bandit algorithm.

Thompson Sampling is a Bayesian approach to the exploration-exploitation
trade-off. Each arm's conversion rate is modeled as a Beta distribution
parameterized by observed successes (alpha) and failures (beta). At each
step the algorithm draws one sample per arm and pulls the arm with the
highest sample — arms with genuine uncertainty naturally get explored.

References:
    Thompson, W. R. (1933). On the likelihood that one unknown probability
    exceeds another in view of the evidence of two samples.
    Biometrika, 25(3/4), 285–294.
"""

import numpy as np
from typing import Optional


class ThompsonSampling:
    """
    Thompson Sampling multi-armed bandit using Beta-Bernoulli conjugate model.

    Each arm starts with a uniform Beta(1, 1) prior. After observing a reward
    the posterior is updated analytically: successes increment alpha, failures
    increment beta. Arm selection samples from each posterior and picks the
    maximum — this naturally balances exploration (high-variance posteriors for
    rarely-pulled arms) against exploitation (high-mean posteriors for proven
    arms).

    Args:
        n_arms: Number of variants / arms to test.
        seed: Optional random seed for reproducibility.

    Attributes:
        alpha: Success counts + 1 (Beta distribution alpha parameter).
        beta:  Failure counts + 1 (Beta distribution beta parameter).
        pulls: Total pulls per arm.
        rewards: Total rewards (successes) per arm.

    Example:
        >>> bandit = ThompsonSampling(n_arms=3, seed=42)
        >>> arm = bandit.select_arm()
        >>> bandit.update(arm, reward=1)
        >>> print(bandit.get_means())
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None):
        self.n_arms = n_arms
        self._rng = np.random.default_rng(seed)

        # Beta distribution parameters — start at uniform Beta(1,1)
        self.alpha = np.ones(n_arms, dtype=float)   # alpha = successes + 1
        self.beta  = np.ones(n_arms, dtype=float)   # beta  = failures  + 1

        # Tracking
        self.pulls   = np.zeros(n_arms, dtype=float)
        self.rewards = np.zeros(n_arms, dtype=float)

    # ── Arm selection ──────────────────────────────────────────────────────────

    def select_arm(self) -> int:
        """
        Draw one sample from each arm's Beta posterior and return the argmax.

        Returns:
            Index of the arm to pull.
        """
        samples = self._rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    # ── Posterior update ───────────────────────────────────────────────────────

    def update(self, arm: int, reward: float) -> None:
        """
        Update the Beta posterior for the pulled arm given the observed reward.

        A reward > 0 is treated as a success and increments alpha.
        A reward of 0 is a failure and increments beta.

        Args:
            arm:    Index of the arm that was pulled.
            reward: Observed reward (typically 0 or 1 for Bernoulli outcomes).
        """
        self.pulls[arm]   += 1
        self.rewards[arm] += reward

        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    # ── Posterior summaries ────────────────────────────────────────────────────

    def get_means(self) -> np.ndarray:
        """
        Return the posterior mean conversion rate for each arm.

        The mean of Beta(alpha, beta) is alpha / (alpha + beta).

        Returns:
            Array of shape (n_arms,) with estimated conversion rates.
        """
        return self.alpha / (self.alpha + self.beta)

    def get_best_arm(self) -> int:
        """
        Return the index of the arm with the highest posterior mean.

        Returns:
            Arm index currently estimated to have the highest conversion rate.
        """
        return int(np.argmax(self.get_means()))

    def get_probabilities(self, n_samples: int = 10_000) -> np.ndarray:
        """
        Estimate the probability that each arm is the true best arm via Monte Carlo.

        Draws n_samples from each arm's posterior and counts how often each arm
        produces the maximum sample across all arms.

        Args:
            n_samples: Number of Monte Carlo draws per arm. Higher = more accurate
                       but slower. Default 10,000 gives ~1% standard error.

        Returns:
            Array of shape (n_arms,) summing to ~1.0.
        """
        # Shape: (n_arms, n_samples)
        draws = self._rng.beta(
            self.alpha[:, np.newaxis],
            self.beta[:, np.newaxis],
            size=(self.n_arms, n_samples),
        )
        best_per_draw = np.argmax(draws, axis=0)   # shape: (n_samples,)
        return np.bincount(best_per_draw, minlength=self.n_arms) / n_samples

    def get_regret(self, true_rates: np.ndarray) -> float:
        """
        Calculate cumulative regret given the true arm conversion rates.

        Regret = total reward we would have earned by always pulling the best
        arm  minus  total reward actually earned.

        Args:
            true_rates: Array of true conversion rates, shape (n_arms,).

        Returns:
            Cumulative regret as a scalar float.
        """
        best_rate = float(np.max(true_rates))
        total_pulls = float(np.sum(self.pulls))
        total_rewards = float(np.sum(self.rewards))
        return best_rate * total_pulls - total_rewards

    # ── Dunder ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"ThompsonSampling("
            f"n_arms={self.n_arms}, "
            f"pulls={self.pulls.tolist()}, "
            f"means={[round(m, 3) for m in self.get_means().tolist()]})"
        )
