'''
Thompson Sampling Multi-Armed Bandit
Bayesian approach using Beta distributions
'''

import numpy as np
from typing import List, Tuple


class ThompsonSampling:
    '''Thompson Sampling bandit for A/B testing'''
    
    def __init__(self, n_arms: int):
        '''
        Initialize bandit
        
        Args:
            n_arms: Number of variants to test
        '''
        self.n_arms = n_arms
        # Beta distribution parameters
        self.alpha = np.ones(n_arms)  # Successes + 1
        self.beta = np.ones(n_arms)   # Failures + 1
        
        # Tracking
        self.pulls = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        '''Select arm by sampling from Beta distributions'''
        samples = np.random.beta(self.alpha, self.beta)
        return int(np.argmax(samples))
        
    def update(self, arm: int, reward: float):
        '''Update after observing reward'''
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
            
    def get_means(self) -> np.ndarray:
        '''Get estimated conversion rate for each arm'''
        return self.alpha / (self.alpha + self.beta)
        
    def get_best_arm(self) -> int:
        '''Get arm with highest estimated rate'''
        return int(np.argmax(self.get_means()))
        
    def get_probabilities(self) -> np.ndarray:
        '''Get probability each arm is best (Monte Carlo)'''
        n_samples = 10000
        samples = np.random.beta(
            self.alpha[:, np.newaxis],
            self.beta[:, np.newaxis],
            size=(self.n_arms, n_samples)
        )
        
        best_arm = np.argmax(samples, axis=0)
        probs = np.bincount(best_arm, minlength=self.n_arms) / n_samples
        
        return probs
        
    def get_regret(self, true_rates: np.ndarray) -> float:
        '''Calculate cumulative regret given true conversion rates'''
        best_rate = np.max(true_rates)
        chosen_rates = true_rates[np.arange(len(self.pulls)).astype(int)]
        regret = np.sum(self.pulls * (best_rate - chosen_rates))
        return regret
        
    def __repr__(self):
        return f'ThompsonSampling(n_arms={self.n_arms}, pulls={self.pulls})'


class UCB1:
    '''UCB1 (Upper Confidence Bound) bandit'''
    
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.pulls = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        self.total_pulls = 0
        
    def select_arm(self) -> int:
        '''Select arm with highest UCB'''
        # Pull each arm once first
        if self.total_pulls < self.n_arms:
            return self.total_pulls
            
        means = self.rewards / np.maximum(self.pulls, 1)
        bonus = np.sqrt(2 * np.log(self.total_pulls) / np.maximum(self.pulls, 1))
        ucb = means + bonus
        
        return int(np.argmax(ucb))
        
    def update(self, arm: int, reward: float):
        '''Update statistics'''
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.total_pulls += 1
        
    def get_best_arm(self) -> int:
        '''Get arm with highest mean reward'''
        means = self.rewards / np.maximum(self.pulls, 1)
        return int(np.argmax(means))


class EpsilonGreedy:
    '''Epsilon-Greedy bandit'''
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.pulls = np.zeros(n_arms)
        self.rewards = np.zeros(n_arms)
        
    def select_arm(self) -> int:
        '''Select arm using epsilon-greedy strategy'''
        if np.random.random() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: best arm
            means = self.rewards / np.maximum(self.pulls, 1)
            return int(np.argmax(means))
            
    def update(self, arm: int, reward: float):
        '''Update statistics'''
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        
    def get_best_arm(self) -> int:
        '''Get arm with highest mean reward'''
        means = self.rewards / np.maximum(self.pulls, 1)
        return int(np.argmax(means))


def simulate_test(true_rates: List[float], n_trials: int, algorithm='thompson'):
    '''
    Simulate A/B test with given algorithm
    
    Args:
        true_rates: True conversion rate for each variant
        n_trials: Number of trials to run
        algorithm: 'thompson', 'ucb', or 'epsilon'
        
    Returns:
        Dict with results
    '''
    n_arms = len(true_rates)
    true_rates = np.array(true_rates)
    
    # Initialize bandit
    if algorithm == 'thompson':
        bandit = ThompsonSampling(n_arms)
    elif algorithm == 'ucb':
        bandit = UCB1(n_arms)
    else:
        bandit = EpsilonGreedy(n_arms)
        
    # Run simulation
    for _ in range(n_trials):
        arm = bandit.select_arm()
        reward = 1 if np.random.random() < true_rates[arm] else 0
        bandit.update(arm, reward)
        
    # Calculate results
    best_arm = np.argmax(true_rates)
    regret = n_trials * true_rates[best_arm] - np.sum(bandit.rewards)
    
    results = {
        'algorithm': algorithm,
        'best_arm': bandit.get_best_arm(),
        'true_best': best_arm,
        'pulls': bandit.pulls,
        'rewards': bandit.rewards,
        'conversion_rates': bandit.rewards / np.maximum(bandit.pulls, 1),
        'regret': regret,
        'regret_pct': (regret / n_trials) * 100
    }
    
    return results


if __name__ == '__main__':
    # Example usage
    print('Thompson Sampling Demo')
    print('=' * 50)
    
    # Simulate test with two variants
    results = simulate_test(
        true_rates=[0.10, 0.12],  # Variant B is 20% better
        n_trials=10000,
        algorithm='thompson'
    )
    
    print(f'\nResults:')
    print(f'  Detected best: Variant {results["best_arm"]}')
    print(f'  True best: Variant {results["true_best"]}')
    print(f'  Pulls: {results["pulls"]}')
    print(f'  Conversion rates: {results["conversion_rates"]}')
    print(f'  Regret: {results["regret"]:.0f} ({results["regret_pct"]:.2f}%)')
