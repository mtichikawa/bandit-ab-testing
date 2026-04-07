"""
streamlit_app.py — Multi-Armed Bandit Interactive Demo

Runs a configurable bandit simulation (Thompson Sampling, UCB1, or
Epsilon-Greedy) in the browser and visualises four key charts:

  1. Cumulative arm pull counts over time
  2. Cumulative regret curve vs. the oracle (always-pull-best) policy
  3. Learned estimate vs. true win rate per arm
  4. Summary table with pulls, rewards, and estimated rates

Usage:
    streamlit run streamlit_app.py
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Allow running from the project root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from bandits.algorithms import UCB1, EpsilonGreedy, ThompsonSampling  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Armed Bandit — Interactive Demo",
    layout="wide",
)

PLOTLY_THEME = "plotly_dark"


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(
    algorithm: str,
    true_rates: list[float],
    n_rounds: int,
    epsilon: float,
    seed: int,
) -> dict:
    """
    Run a full bandit simulation and return per-round tracking data.

    Args:
        algorithm:   "Thompson Sampling", "UCB1", or "Epsilon-Greedy"
        true_rates:  True Bernoulli win rate for each arm
        n_rounds:    Number of rounds to simulate
        epsilon:     Exploration probability (only used for Epsilon-Greedy)
        seed:        Random seed for reproducibility

    Returns:
        Dictionary with keys:
            arm_pulls_over_time  - (n_rounds, n_arms) cumulative pull counts
            regret_over_time     - (n_rounds,) cumulative regret
            final_estimates      - estimated rate per arm after all rounds
            pulls                - total pulls per arm
            rewards              - total rewards per arm
            bandit               - the bandit object (for get_probabilities etc.)
    """
    rng = np.random.default_rng(seed)
    n_arms = len(true_rates)
    true_rates_arr = np.asarray(true_rates, dtype=float)
    best_rate = float(np.max(true_rates_arr))

    # Instantiate algorithm
    if algorithm == "Thompson Sampling":
        bandit = ThompsonSampling(n_arms, seed=seed)
    elif algorithm == "UCB1":
        bandit = UCB1(n_arms, seed=seed)
    else:  # Epsilon-Greedy
        bandit = EpsilonGreedy(n_arms, epsilon=epsilon, seed=seed)

    # Per-round tracking arrays
    arm_pulls_over_time = np.zeros((n_rounds, n_arms), dtype=int)
    regret_over_time = np.zeros(n_rounds, dtype=float)

    cumulative_pulls = np.zeros(n_arms, dtype=int)
    cumulative_reward = 0.0
    cumulative_regret = 0.0

    for t in range(n_rounds):
        arm = bandit.select_arm()
        reward = 1 if rng.random() < true_rates_arr[arm] else 0
        bandit.update(arm, reward)

        cumulative_pulls[arm] += 1
        cumulative_reward += reward
        # Per-step regret: best_rate − expected reward from chosen arm
        cumulative_regret += best_rate - true_rates_arr[arm]

        arm_pulls_over_time[t] = cumulative_pulls.copy()
        regret_over_time[t] = cumulative_regret

    # Compute estimated rates depending on algorithm type
    if algorithm == "Thompson Sampling":
        final_estimates = bandit.get_means()
    else:
        final_estimates = bandit.rewards / np.maximum(bandit.pulls, 1)

    return {
        "arm_pulls_over_time": arm_pulls_over_time,
        "regret_over_time": regret_over_time,
        "final_estimates": final_estimates,
        "pulls": bandit.pulls,
        "rewards": bandit.rewards,
        "bandit": bandit,
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Simulation Settings")

algorithm = st.sidebar.selectbox(
    "Algorithm",
    ["Thompson Sampling", "UCB1", "Epsilon-Greedy"],
    index=0,
)

n_arms = st.sidebar.slider("Number of arms", min_value=2, max_value=10, value=4, step=1)

st.sidebar.markdown("**True win rates per arm**")
# Generate stable random defaults keyed to n_arms so they don't jump around
default_rng = np.random.default_rng(99)
default_rates = sorted(default_rng.uniform(0.05, 0.55, n_arms).tolist())

true_rates = []
for i in range(n_arms):
    rate = st.sidebar.slider(
        f"Arm {i} win rate",
        min_value=0.0,
        max_value=1.0,
        value=round(default_rates[i], 2),
        step=0.01,
        key=f"arm_rate_{i}",
    )
    true_rates.append(rate)

n_rounds = st.sidebar.slider(
    "Number of rounds",
    min_value=100,
    max_value=10_000,
    value=2_000,
    step=100,
)

# Epsilon slider — only meaningful for Epsilon-Greedy
if algorithm == "Epsilon-Greedy":
    epsilon = st.sidebar.slider(
        "Epsilon (exploration rate)",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
    )
else:
    epsilon = 0.10  # unused, just needs a value

seed = st.sidebar.number_input("Random seed", value=42, step=1)

run_button = st.sidebar.button("Run Simulation", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------

st.title("Multi-Armed Bandit — Interactive Demo")
st.markdown(
    "Configure the algorithm and arm parameters in the sidebar, then click "
    "**Run Simulation** to see how the bandit learns over time."
)

# ---------------------------------------------------------------------------
# Run and display results
# ---------------------------------------------------------------------------

if run_button:
    # ---- Run simulation ----
    with st.spinner("Running simulation..."):
        results = run_simulation(
            algorithm=algorithm,
            true_rates=true_rates,
            n_rounds=n_rounds,
            epsilon=epsilon,
            seed=int(seed),
        )

    arm_pulls = results["arm_pulls_over_time"]   # (n_rounds, n_arms)
    regret    = results["regret_over_time"]       # (n_rounds,)
    estimates = results["final_estimates"]        # (n_arms,)
    pulls     = results["pulls"]                  # (n_arms,)
    rewards   = results["rewards"]                # (n_arms,)

    arm_labels = [f"Arm {i}" for i in range(n_arms)]
    rounds_idx = np.arange(1, n_rounds + 1)

    # -----------------------------------------------------------------------
    # Chart 1: Arm selection over time (cumulative pull counts)
    # -----------------------------------------------------------------------
    pulls_df = pd.DataFrame(arm_pulls, columns=arm_labels)
    pulls_df["Round"] = rounds_idx

    # Sample to ~500 points for rendering performance
    step = max(1, n_rounds // 500)
    pulls_df_sampled = pulls_df.iloc[::step]

    fig_pulls = px.line(
        pulls_df_sampled,
        x="Round",
        y=arm_labels,
        title="Arm Selection Over Time (cumulative pulls)",
        labels={"value": "Cumulative Pulls", "variable": "Arm"},
        template=PLOTLY_THEME,
    )
    fig_pulls.update_layout(legend_title_text="Arm")

    # -----------------------------------------------------------------------
    # Chart 2: Cumulative regret curve
    # -----------------------------------------------------------------------
    regret_df = pd.DataFrame({"Round": rounds_idx, "Cumulative Regret": regret})
    regret_df_sampled = regret_df.iloc[::step]

    fig_regret = px.line(
        regret_df_sampled,
        x="Round",
        y="Cumulative Regret",
        title="Regret Curve (vs. oracle always-pull-best policy)",
        template=PLOTLY_THEME,
    )
    fig_regret.update_traces(line_color="#f05454")

    # -----------------------------------------------------------------------
    # Chart 3: Estimated vs. true win rates
    # -----------------------------------------------------------------------
    rates_df = pd.DataFrame({
        "Arm": arm_labels * 2,
        "Rate": list(true_rates) + list(estimates),
        "Type": ["True Rate"] * n_arms + ["Estimated Rate"] * n_arms,
    })

    fig_rates = px.bar(
        rates_df,
        x="Arm",
        y="Rate",
        color="Type",
        barmode="group",
        title="Estimated vs. True Win Rate per Arm",
        labels={"Rate": "Win Rate"},
        template=PLOTLY_THEME,
        color_discrete_map={
            "True Rate": "#4c9be8",
            "Estimated Rate": "#f0a500",
        },
    )
    fig_rates.update_layout(yaxis_range=[0, 1])

    # -----------------------------------------------------------------------
    # Layout: 2 × 2 grid
    # -----------------------------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_pulls, use_container_width=True)
    with col2:
        st.plotly_chart(fig_regret, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.plotly_chart(fig_rates, use_container_width=True)

    # -----------------------------------------------------------------------
    # Chart 4 (col4): Summary table
    # -----------------------------------------------------------------------
    with col4:
        best_arm_idx = int(np.argmax(true_rates))
        summary_data = {
            "Arm": arm_labels,
            "True Rate": [f"{r:.3f}" for r in true_rates],
            "Pulls": [int(p) for p in pulls],
            "% of Pulls": [f"{100 * p / n_rounds:.1f}%" for p in pulls],
            "Est. Rate": [f"{e:.3f}" for e in estimates],
            "Est. Error": [f"{abs(e - t):.3f}" for e, t in zip(estimates, true_rates)],
        }
        summary_df = pd.DataFrame(summary_data)

        st.subheader("Summary Table")

        # Highlight best arm row
        def highlight_best(row):
            if row["Arm"] == f"Arm {best_arm_idx}":
                return ["background-color: #1e4d2b"] * len(row)
            return [""] * len(row)

        st.dataframe(
            summary_df.style.apply(highlight_best, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        # Quick stats below the table
        total_reward = int(np.sum(rewards))
        final_regret = float(regret[-1])
        identified_best = int(np.argmax(estimates))

        st.markdown(f"**Best arm (true):** Arm {best_arm_idx} — rate {true_rates[best_arm_idx]:.3f}")
        st.markdown(f"**Best arm (identified):** Arm {identified_best} — "
                    f"{'correct' if identified_best == best_arm_idx else 'incorrect'}")
        st.markdown(f"**Total reward collected:** {total_reward:,}")
        st.markdown(f"**Final cumulative regret:** {final_regret:.1f} "
                    f"({100 * final_regret / n_rounds:.2f}% of rounds)")

else:
    st.info("Configure the simulation in the sidebar and click **Run Simulation** to begin.")
