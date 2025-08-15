import os
import pandas as pd

def calculate_business_impact():
    # Assumptions
    ARPU = 50
    ANNUAL_USERS = 500_000

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    subgroup_metrics_path = os.path.join(OUTPUTS_DIR, "subgroup_lift_metrics.csv")
    subgroup_df = pd.read_csv(subgroup_metrics_path)

    s_learner_subgroup = subgroup_df[subgroup_df['Model'] == 'uplift_s_learner']

    if s_learner_subgroup.empty:
        raise ValueError("No S-Learner subgroup data found.")

    subgroup_size_test = s_learner_subgroup['Target_Group_Size'].values[0]
    uplift_relative = s_learner_subgroup['Uplift_Relative_Pct'].values[0] / 100
    TEST_FRACTION = 0.2
    overall_ate_relative = -0.2927

    subgroup_annual_users = ANNUAL_USERS * TEST_FRACTION
    full_rollout_loss = ANNUAL_USERS * overall_ate_relative * ARPU
    targeted_gain = subgroup_annual_users * uplift_relative * ARPU
    net_savings = targeted_gain - full_rollout_loss

    summary = f"""
===== Business Impact Summary =====
Assumptions:
- Average Revenue Per User (ARPU): ${ARPU} per year
- Total Annual Users: {ANNUAL_USERS}
- Target Subgroup Size (top 20% responders): {subgroup_annual_users:,.0f} users

Full Rollout Loss (treatment hurts overall):
- {ANNUAL_USERS:,} users × {overall_ate_relative*100:.2f}% lift × ${ARPU} ARPU
= ${full_rollout_loss:,.2f} per year (negative means revenue loss)

Targeted Subgroup Profit (using S-Learner top responders):
- {subgroup_annual_users:,.0f} users × {uplift_relative*100:.2f}% lift × ${ARPU} ARPU
= +${targeted_gain:,.2f} per year

Net Savings by Targeting Subgroup Instead of Full Rollout:
= Targeted Gain - Full Rollout Loss = ${net_savings:,.2f} per year

===== End of Business Impact Summary =====
"""

    summary_path = os.path.join(OUTPUTS_DIR, "business_impact_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary.strip())

    return {
        "full_rollout_loss": full_rollout_loss,
        "targeted_gain": targeted_gain,
        "net_savings": net_savings,
        "summary_path": summary_path
    }


if __name__ == "__main__":
    results = calculate_business_impact()
    print(f"Saved full summary to {results['summary_path']}")
    print(f"Net Savings: ${results['net_savings']:,.2f}")