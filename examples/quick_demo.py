'''Quick Thompson Sampling Demo'''

import sys
sys.path.append('..')
from bandits.algorithms import ThompsonSampling
import numpy as np

# Simulate A/B test
print('Thompson Sampling A/B Test Demo')
print('=' * 50)

# True conversion rates (unknown to algorithm)
true_rates = [0.10, 0.12]  # Variant B is better
bandit = ThompsonSampling(n_arms=2)

# Run 1000 trials
for trial in range(1000):
    # Select variant
    arm = bandit.select_arm()
    
    # Simulate conversion
    converted = np.random.random() < true_rates[arm]
    reward = 1 if converted else 0
    
    # Update bandit
    bandit.update(arm, reward)

# Results
print(f'\nResults after 1000 trials:')
print(f'Variant A pulls: {bandit.pulls[0]:.0f}')
print(f'Variant B pulls: {bandit.pulls[1]:.0f}')
print(f'Estimated rates: {bandit.get_means()}')
print(f'Best variant: {bandit.get_best_arm()}')
print(f'Probability B is best: {bandit.get_probabilities()[1]:.1%}')
