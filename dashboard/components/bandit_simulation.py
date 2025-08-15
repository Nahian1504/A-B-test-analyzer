import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

class ThompsonSamplingBandit:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.successes = np.zeros(n_arms)
        self.failures = np.zeros(n_arms)

    def select_arm(self):
        samples = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        if reward == 1:
            self.successes[chosen_arm] += 1
        else:
            self.failures[chosen_arm] += 1

def simulate_bandit(bandit, rounds=1000):
    rewards = []
    chosen_arms = []
    for _ in range(rounds):
        arm = bandit.select_arm()
        # Simulate reward (here randomly for demo)
        reward = np.random.binomial(1, 0.5)
        bandit.update(arm, reward)
        rewards.append(reward)
        chosen_arms.append(arm)
    return rewards, chosen_arms

def plot_rewards(rewards):
    plt.plot(np.cumsum(rewards))
    plt.xlabel("Rounds")
    plt.ylabel("Cumulative Rewards")
    plt.title("Bandit Cumulative Rewards")
    st.pyplot(plt)