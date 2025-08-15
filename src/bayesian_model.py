import os
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def run_bayesian_ab_test(treatment, converted):
    with pm.Model() as model:
        # Priors for conversion rates in control and treatment groups
        p_control = pm.Beta('p_control', alpha=1, beta=1)
        p_treatment = pm.Beta('p_treatment', alpha=1, beta=1)

        # Boolean masks for groups
        treat_idx = treatment == 1
        control_idx = treatment == 0

        # Likelihood (observations) with Bernoulli for each group
        pm.Bernoulli('obs_treatment', p=p_treatment, observed=converted[treat_idx])
        pm.Bernoulli('obs_control', p=p_control, observed=converted[control_idx])

        # Deterministic difference
        delta = pm.Deterministic('delta', p_treatment - p_control)

        # Sample with ArviZ InferenceData output
        trace = pm.sample(
            draws=2000,
            tune=1000,
            target_accept=0.95,
            return_inferencedata=True,
            cores=1,
            progressbar=True,
            random_seed=42,
        )
    
    return trace

def plot_posterior(trace):
    fig = az.plot_posterior(trace, var_names=['p_control', 'p_treatment', 'delta'])
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, "posterior_trace.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Posterior plot saved to {save_path}")