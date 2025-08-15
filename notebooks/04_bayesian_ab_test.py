import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
from bayesian_model import run_bayesian_ab_test, plot_posterior

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
    PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    test_path = os.path.join(DATA_DIR, "test.csv")
    test_df = pd.read_csv(test_path)

    # Prepare treatment vector from 'test_group_psa' forcing integer type
    if test_df['test_group_psa'].dtype == 'O':
        treatment = (test_df['test_group_psa'] == 'treatment').astype(int).values
    else:
        treatment = test_df['test_group_psa'].astype(int).values

    # Outcome vector, force integer type
    converted = test_df['converted'].astype(int).values

    # Run Bayesian A/B test, get trace as ArviZ InferenceData
    trace = run_bayesian_ab_test(treatment, converted)

    # Plot posterior distributions and save plot
    plot_posterior(trace)

    print("Bayesian A/B test completed. Posterior plots saved.")

if __name__ == "__main__":
    main()