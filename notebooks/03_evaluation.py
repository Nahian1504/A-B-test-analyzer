import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

import pandas as pd
from evaluation import qini_curve_multiple, uplift_score, calculate_ate, subgroup_lift

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load test data and uplift predictions
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    uplift_df = pd.read_csv(os.path.join(OUTPUTS_DIR, "uplift_predictions.csv"))

    # Merge on user id
    merged_df = pd.merge(test_df, uplift_df, on="user id", how="inner")

    treatment = merged_df['test_group_psa'].values
    outcome = merged_df['converted'].values

    # Get uplift predictions
    uplifts = {
        "T-Learner": merged_df['uplift_t_learner'].values,
        "S-Learner": merged_df['uplift_s_learner'].values,
        "X-Learner": merged_df['uplift_x_learner'].values,
    }

    # Plot combined Qini curves
    qini_plot_path = os.path.join(PLOTS_DIR, "qini_curves_comparison.png")
    qini_curve_multiple(uplifts, treatment, outcome, qini_plot_path)
    print(f"Combined Qini curves plot saved to: {qini_plot_path}")

    # Calculate uplift scores and store results
    results = []
    for label, uplift in uplifts.items():
        score = uplift_score(uplift, treatment, outcome)
        print(f"Uplift score ({label}): {score:.4f}")
        results.append({"Model": label, "Uplift_Score": score})

    # Save uplift scores to CSV
    scores_df = pd.DataFrame(results)
    scores_csv_path = os.path.join(OUTPUTS_DIR, "uplift_scores.csv")
    scores_df.to_csv(scores_csv_path, index=False)
    print(f"Uplift scores saved to: {scores_csv_path}")

    # Calculate and print overall ATE
    conv_t, conv_c, ate, lift_pct = calculate_ate(merged_df)
    print("\n=== Overall Average Treatment Effect (ATE) ===")
    print(f"Conversion Rate (Treatment): {conv_t:.4f}")
    print(f"Conversion Rate (Control): {conv_c:.4f}")
    print(f"ATE (Absolute Lift): {ate:.4f}")
    print(f"ATE (Relative Lift %): {lift_pct:.2f}%")

    # Calculate and print subgroup lifts
    print("\n=== Subgroup Lift Metrics (Top 20% Responders) ===")
    subgroup_results = []
    for model_name in ['uplift_t_learner', 'uplift_s_learner', 'uplift_x_learner']:
        size, conv_t_sub, conv_c_sub, uplift_sub, uplift_pct_sub = subgroup_lift(merged_df, model_name)
        print(f"\nModel: {model_name}")
        print(f"Target Group Size: {size}")
        print(f"Conversion Rate (Treatment): {conv_t_sub:.4f}")
        print(f"Conversion Rate (Control): {conv_c_sub:.4f}")
        print(f"Uplift (Absolute): {uplift_sub:.4f}")
        print(f"Uplift (Relative %): {uplift_pct_sub:.2f}%")
        subgroup_results.append({
            "Model": model_name,
            "Target_Group_Size": size,
            "Conv_Rate_Treatment": conv_t_sub,
            "Conv_Rate_Control": conv_c_sub,
            "Uplift_Absolute": uplift_sub,
            "Uplift_Relative_Pct": uplift_pct_sub
        })

    # Save subgroup lift results
    subgroup_df = pd.DataFrame(subgroup_results)
    subgroup_csv_path = os.path.join(OUTPUTS_DIR, "subgroup_lift_metrics.csv")
    subgroup_df.to_csv(subgroup_csv_path, index=False)
    print(f"\nSubgroup lift metrics saved to: {subgroup_csv_path}")


if __name__ == "__main__":
    main()