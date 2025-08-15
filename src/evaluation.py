import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def plot_qini_curve(uplift, treatment, outcome, save_path=None):
    """
    Plot Qini curve for a single uplift model.
    """
    n = len(uplift)
    order = np.argsort(-uplift)
    treatment_sorted = treatment[order]
    outcome_sorted = outcome[order]

    cum_treated = np.cumsum(treatment_sorted)
    cum_control = np.arange(1, n+1) - cum_treated

    cum_outcome_treated = np.cumsum(outcome_sorted * treatment_sorted)
    cum_outcome_control = np.cumsum(outcome_sorted * (1 - treatment_sorted))

    with np.errstate(divide='ignore', invalid='ignore'):
        rate_treated = np.divide(cum_outcome_treated, cum_treated, out=np.zeros_like(cum_outcome_treated), where=cum_treated!=0)
        rate_control = np.divide(cum_outcome_control, cum_control, out=np.zeros_like(cum_outcome_control), where=cum_control!=0)

    uplift_at_k = rate_treated - rate_control

    plt.figure(figsize=(8,5))
    plt.plot(np.arange(1, n+1), uplift_at_k, label="Qini curve")
    plt.xlabel("Number of Samples")
    plt.ylabel("Uplift")
    plt.title("Qini Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def qini_curve_multiple(uplifts_dict, treatment, outcome, save_path):
    plt.figure(figsize=(8,5))
    cum_n = np.arange(1, len(treatment)+1)

    for label, uplift in uplifts_dict.items():
        order = np.argsort(-uplift)
        treatment_sorted = treatment[order]
        outcome_sorted = outcome[order]

        cum_treated = np.cumsum(treatment_sorted)
        cum_control = cum_n - cum_treated

        cum_outcome_treated = np.cumsum(outcome_sorted * treatment_sorted)
        cum_outcome_control = np.cumsum(outcome_sorted * (1 - treatment_sorted))

        rate_treated = np.divide(cum_outcome_treated, cum_treated, out=np.zeros_like(cum_outcome_treated), where=cum_treated!=0)
        rate_control = np.divide(cum_outcome_control, cum_control, out=np.zeros_like(cum_outcome_control), where=cum_control!=0)

        uplift_at_k = rate_treated - rate_control

        plt.plot(cum_n, uplift_at_k, label=f"Qini curve ({label})")

    plt.xlabel("Number of Samples")
    plt.ylabel("Uplift")
    plt.title("Qini Curves Comparison")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def uplift_score(uplift, treatment, outcome):
    # Simplified uplift score using ROC-AUC as proxy
    try:
        score = roc_auc_score(treatment, uplift)
    except ValueError:
        score = float('nan')
    return score

def calculate_ate(df, treatment_col='test_group_psa', outcome_col='converted'):
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]
    
    conv_rate_treated = treated[outcome_col].mean() if len(treated) > 0 else float('nan')
    conv_rate_control = control[outcome_col].mean() if len(control) > 0 else float('nan')
    
    ate = conv_rate_treated - conv_rate_control
    lift_pct = (ate / conv_rate_control) * 100 if conv_rate_control and conv_rate_control > 0 else float('nan')
    
    return conv_rate_treated, conv_rate_control, ate, lift_pct

def subgroup_lift(df, uplift_col, treatment_col='test_group_psa', outcome_col='converted', top_pct=0.2):
    threshold = df[uplift_col].quantile(1 - top_pct)
    top_group = df[df[uplift_col] >= threshold]
    
    treated = top_group[top_group[treatment_col] == 1]
    control = top_group[top_group[treatment_col] == 0]

    
    # If no treated samples in top group, adjust selection to include top treated and control samples separately
    if len(treated) == 0:
        n_top = int(len(df) * top_pct)
        treated = df[df[treatment_col] == 1].nlargest(n_top, uplift_col)
        control = df[df[treatment_col] == 0].nlargest(n_top, uplift_col)
        top_group = pd.concat([treated, control])


    conv_rate_treated = treated[outcome_col].mean() if len(treated) > 0 else float('nan')
    conv_rate_control = control[outcome_col].mean() if len(control) > 0 else float('nan')
    
    uplift = conv_rate_treated - conv_rate_control
    lift_pct = (uplift / conv_rate_control) * 100 if conv_rate_control and conv_rate_control > 0 else float('nan')
    
    return len(top_group), conv_rate_treated, conv_rate_control, uplift, lift_pct