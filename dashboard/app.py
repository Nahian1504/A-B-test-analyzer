import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPONENTS_DIR = os.path.join(BASE_DIR, "dashboard", "components")
SRC_DIR = os.path.join(BASE_DIR, "src")
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")  # <-- NEW

sys.path.append(SCRIPTS_DIR)  # <-- ADD THIS
sys.path.append(COMPONENTS_DIR)
sys.path.append(SRC_DIR)

from evaluation import calculate_ate, subgroup_lift, uplift_score
from uplift_curve import plot_qini_curve
from bandit_simulation import ThompsonSamplingBandit, simulate_bandit, plot_rewards
from business_impact import calculate_business_impact 

import streamlit as st

# Paths to your test data and uplift prediction CSVs
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "test.csv")
UPLIFT_PRED_PATH = os.path.join(BASE_DIR, "outputs", "uplift_predictions.csv")

st.title("A/B Test Analyzer Dashboard")

@st.cache_data
def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)

@st.cache_data
def load_uplift_preds():
    return pd.read_csv(UPLIFT_PRED_PATH)

test_df = load_test_data()
uplift_df = load_uplift_preds()

st.subheader("Test Data Preview")
st.dataframe(test_df.head())

st.subheader("Uplift Predictions Preview")
st.dataframe(uplift_df.head())

merged_df = pd.merge(test_df, uplift_df, on="user id", how="inner")
st.write(f"Merged data shape: {merged_df.shape}")

treatment = merged_df["test_group_psa"].values
outcome = merged_df["converted"].values

uplift_columns = [col for col in uplift_df.columns if col != "user id"]

selected_uplift = st.selectbox("Select uplift model to visualize", uplift_columns)

if selected_uplift:
    uplift = merged_df[selected_uplift].values
    st.write(f"Showing Qini Curve for: **{selected_uplift}**")
    plot_qini_curve(uplift, treatment, outcome)

if st.checkbox("Show all uplift Qini curves comparison"):
    fig, ax = plt.subplots(figsize=(8, 6))
    cum_n = np.arange(1, len(uplift) + 1)

    for col in uplift_columns:
        uplift_vals = merged_df[col].values
        order = np.argsort(-uplift_vals)
        treatment_sorted = treatment[order]
        outcome_sorted = outcome[order]

        cum_treated = np.cumsum(treatment_sorted)
        cum_control = cum_n - cum_treated

        cum_outcome_treated = np.cumsum(outcome_sorted * treatment_sorted)
        cum_outcome_control = np.cumsum(outcome_sorted * (1 - treatment_sorted))

        with np.errstate(divide='ignore', invalid='ignore'):
            uplift_at_k = (cum_outcome_treated / np.clip(cum_treated, 1, None)) - \
                          (cum_outcome_control / np.clip(cum_control, 1, None))
        uplift_at_k = np.nan_to_num(uplift_at_k)

        ax.plot(cum_n, uplift_at_k, label=col)

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Uplift")
    ax.set_title("Qini Curve Comparison of Uplift Models")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

st.markdown("---")

st.subheader("Quantitative Metrics")

conv_t, conv_c, ate, lift_pct = calculate_ate(merged_df)
st.markdown("### Overall Average Treatment Effect (ATE)")
st.write(f"- Conversion Rate (Treatment): **{conv_t:.4f}**")
st.write(f"- Conversion Rate (Control): **{conv_c:.4f}**")
st.write(f"- ATE (Absolute Lift): **{ate:.4f}**")
st.write(f"- ATE (Relative Lift %): **{lift_pct:.2f}%**")

st.markdown("### Uplift Scores")
for col in uplift_columns:
    score = uplift_score(merged_df[col].values, treatment, outcome)
    st.write(f"- **{col}**: {score:.4f}")

st.markdown("### Subgroup Lift Metrics (Top 20% Responders)")

selected_lift_model = st.selectbox("Select uplift model for subgroup metrics", uplift_columns)

if selected_lift_model:
    size, conv_t_sub, conv_c_sub, uplift_sub, uplift_pct_sub = subgroup_lift(merged_df, selected_lift_model)
    st.write(f"**{selected_lift_model}**:")
    st.write(f"- Target Group Size: {size}")
    st.write(f"- Conversion Rate (Treatment): {conv_t_sub:.4f}")
    st.write(f"- Conversion Rate (Control): {conv_c_sub:.4f}")
    st.write(f"- Uplift (Absolute): {uplift_sub:.4f}")
    st.write(f"- Uplift (Relative %): {uplift_pct_sub:.2f}%")
    st.write("---")

# Business Impact Metrics
st.markdown("### Business Impact Metrics")
try:
    impact_results = calculate_business_impact()
    targeted_gain = impact_results["targeted_gain"]
    net_savings = impact_results["net_savings"]

    st.write(
        f"**Targeted Subgroup Profit (using S-Learner top responders):** "
        f"+${targeted_gain:,.2f} per year"
    )
    st.write(
        f"**Net Savings by Targeting Subgroup Instead of Full Rollout:** "
        f"${net_savings:,.2f} per year"
    )
except Exception as e:
    st.error(f"Could not calculate business impact: {e}")

st.markdown("---")

st.subheader("Thompson Sampling Bandit Simulation")

n_arms = st.slider("Number of arms", min_value=2, max_value=10, value=3)
rounds = st.slider("Number of rounds", min_value=100, max_value=10000, step=100, value=1000)

if st.button("Run Bandit Simulation"):
    bandit = ThompsonSamplingBandit(n_arms)
    rewards, chosen_arms = simulate_bandit(bandit, rounds)
    plot_rewards(rewards)

