import matplotlib.pyplot as plt
import numpy as np
import os
import streamlit as st

def plot_qini_curve(uplift, treatment, outcome):
    order = np.argsort(-uplift)
    treatment_sorted = treatment[order]
    outcome_sorted = outcome[order]
    cum_n = np.arange(1, len(uplift)+1)

    cum_treated = np.cumsum(treatment_sorted)
    cum_control = cum_n - cum_treated

    cum_outcome_treated = np.cumsum(outcome_sorted * treatment_sorted)
    cum_outcome_control = np.cumsum(outcome_sorted * (1 - treatment_sorted))

    with np.errstate(divide='ignore', invalid='ignore'):
        uplift_at_k = (cum_outcome_treated / np.clip(cum_treated, 1, None)) - (cum_outcome_control / np.clip(cum_control, 1, None))
    uplift_at_k = np.nan_to_num(uplift_at_k)

    fig, ax = plt.subplots()
    ax.plot(cum_n, uplift_at_k, label="Qini Curve")
    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Uplift")
    ax.set_title("Qini Curve")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)