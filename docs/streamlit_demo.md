# Streamlit Demo — Multi-Armed Bandit

An interactive browser-based demo for exploring the three bandit algorithms
implemented in this project: Thompson Sampling, UCB1, and Epsilon-Greedy.

## How to run

```bash
# From the bandit-ab-testing project directory
source venv/bin/activate
pip install streamlit plotly          # one-time, if not already installed
streamlit run streamlit_app.py
```

The app opens in your default browser at `http://localhost:8501`.

---

## Sidebar controls

| Control | Description |
|---|---|
| **Algorithm** | Selects which bandit policy drives arm selection each round |
| **Number of arms** | How many variants to test (2–10) |
| **Arm win rates** | The true Bernoulli conversion rate for each arm — what you'd set in a real A/B test to know ground truth |
| **Number of rounds** | Total observations simulated; higher values let the algorithm converge more fully |
| **Epsilon** | Exploration probability for Epsilon-Greedy only: fraction of rounds where a random arm is chosen instead of the current best |
| **Random seed** | Fixes the random draws for reproducibility; change it to see a different run with identical settings |
| **Run Simulation** | Executes the simulation and renders all four charts |

---

## Charts

### 1. Arm Selection Over Time
Cumulative pull counts for each arm, plotted against the round number.
A well-performing algorithm rapidly concentrates pulls on the best arm while
still giving the others enough exploration early on.

### 2. Regret Curve
**Cumulative regret** measures the total reward lost compared to an oracle
policy that always pulls the true best arm.

```
regret(t) = sum_{i=1}^{t} [ best_rate − true_rate(arm chosen at step i) ]
```

A flat (zero-slope) curve means the algorithm has converged and is no longer
making suboptimal choices. A steep curve means it is still spending time on
weaker arms. Lower total regret at the end of the run is better.

Thompson Sampling typically achieves near-logarithmic regret (the theoretical
lower bound), UCB1 is similar in theory but often higher in practice on short
horizons, and Epsilon-Greedy accumulates linear regret because it always
explores at rate epsilon regardless of certainty.

### 3. Estimated vs. True Win Rate
A grouped bar chart comparing the algorithm's learned estimate of each arm's
conversion rate against the ground-truth slider values. Smaller gaps between
the orange (estimated) and blue (true) bars indicate better learning.

### 4. Summary Table
Per-arm breakdown of:
- Total pulls and percentage share of all rounds
- Empirical estimated rate
- Absolute estimation error vs. the true rate

The true best arm row is highlighted in green. Below the table, a quick
summary shows whether the algorithm correctly identified the best arm and the
final cumulative regret as a percentage of all rounds.

---

## Algorithm notes

**Thompson Sampling** — Bayesian. Maintains a Beta(alpha, beta) posterior for
each arm and samples from it each round. High uncertainty arms get sampled
more often naturally, which drives efficient exploration without a tuning
parameter.

**UCB1** — Frequentist. Adds an `sqrt(2 log(t) / n_i)` confidence bonus to
each arm's mean. Every arm is pulled at least once during a warm-up phase,
then the algorithm always picks the arm with the highest upper confidence
bound.

**Epsilon-Greedy** — Simplest policy. With probability epsilon a uniformly
random arm is chosen (explore); otherwise the arm with the best empirical mean
is chosen (exploit). Easy to understand but wastes exploration budget even
after convergence.
