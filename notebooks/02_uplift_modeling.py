import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))
from uplift_models import TLearner, SLearner, XLearner
from sklearn.preprocessing import StandardScaler

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def get_features_targets(df):
    drop_cols = ['index', 'user id', 'converted']
    features = [col for col in df.columns if col not in drop_cols]

    if 'test_group_psa' not in df.columns:
        raise ValueError("Treatment column 'test_group_psa' missing in dataframe")

    X = df[features].values
    treatment = df['test_group_psa'].values
    y = df['converted'].values
    return X, treatment, y

def plot_uplift_hist(uplift, model_name):
    plt.figure(figsize=(8, 4))
    plt.hist(uplift, bins=50, alpha=0.7, color='skyblue')
    plt.title(f'Uplift Distribution - {model_name}')
    plt.xlabel('Predicted Uplift')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(PLOTS_DIR, f'uplift_hist_{model_name.lower().replace("-", "_")}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved uplift histogram plot for {model_name} to {save_path}")

def main():
    train_df, test_df = load_data()

    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")

    # Define feature columns (exclude id, target, treatment)
    drop_cols = ['index', 'user id', 'converted']
    features = [col for col in train_df.columns if col not in drop_cols]

    # Extract raw features for scaling
    X_train_raw = train_df[features].values
    X_test_raw = test_df[features].values

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Extract treatment and target arrays
    treatment_train = train_df['test_group_psa'].values
    y_train = train_df['converted'].values

    treatment_test = test_df['test_group_psa'].values
    y_test = test_df['converted'].values

    # Initialize learners with increased max_iter inside their classes
    t_learner = TLearner()
    s_learner = SLearner()
    x_learner = XLearner()

    # Fit models
    print("Training T-Learner...")
    t_learner.fit(X_train, treatment_train, y_train)

    print("Training S-Learner...")
    s_learner.fit(X_train, treatment_train, y_train)

    print("Training X-Learner...")
    x_learner.fit(X_train, treatment_train, y_train)

    # Predict uplift on test data
    print("Predicting uplift on test data...")
    uplift_t = t_learner.predict(X_test)
    uplift_s = s_learner.predict(X_test)
    uplift_x = x_learner.predict(X_test)

    # Plot and save uplift distributions
    plot_uplift_hist(uplift_t, "T-Learner")
    plot_uplift_hist(uplift_s, "S-Learner")
    plot_uplift_hist(uplift_x, "X-Learner")

    # Save uplift predictions with user ids
    results_df = test_df[['user id']].copy()
    results_df['uplift_t_learner'] = uplift_t
    results_df['uplift_s_learner'] = uplift_s
    results_df['uplift_x_learner'] = uplift_x

    uplift_csv_path = os.path.join(OUTPUT_DIR, "uplift_predictions.csv")
    results_df.to_csv(uplift_csv_path, index=False)
    print(f"Saved uplift predictions to {uplift_csv_path}")

    # Save scaler as well for future use (important!)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    # Save trained models for future use (SHAP, evaluation, etc.)
    print("Saving trained models...")
    joblib.dump(t_learner, os.path.join(OUTPUT_DIR, "t_learner_model.pkl"))
    joblib.dump(s_learner, os.path.join(OUTPUT_DIR, "s_learner_model.pkl"))
    joblib.dump(x_learner, os.path.join(OUTPUT_DIR, "x_learner_model.pkl"))
    print("Models saved to outputs directory.")

if __name__ == "__main__":
    main() 