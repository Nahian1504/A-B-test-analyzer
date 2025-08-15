import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    classification_report,
)
from sklearn.dummy import DummyClassifier

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src"))

# Baseline logistic regression (for sanity check only - imbalanced data limits usefulness)  
# Focus: Causal inference (ATE/uplift) is primary goal.  
def bootstrap_ate(data, treatment_col, outcome_col, n_bootstrap=1000, seed=42):
    np.random.seed(seed)

    data = data.dropna(subset=[treatment_col, outcome_col])

    # Map numeric treatment encoding 0.0->'control', 1.0->'treatment'
    mapping = {0: 'control', 0.0: 'control', 1: 'treatment', 1.0: 'treatment'}
    data['treatment_str'] = data[treatment_col].map(mapping).astype(str)

    value_counts = data['treatment_str'].value_counts()
    print("Group counts:\n", value_counts)

    if 'treatment' not in value_counts or 'control' not in value_counts:
        raise ValueError(f"Missing treatment or control group. Found groups: {value_counts.index.tolist()}")

    treated = data[data['treatment_str'] == 'treatment'][outcome_col]
    control = data[data['treatment_str'] == 'control'][outcome_col]

    ates = []
    for _ in range(n_bootstrap):
        treated_sample = treated.sample(frac=1, replace=True)
        control_sample = control.sample(frac=1, replace=True)
        ate = treated_sample.mean() - control_sample.mean()
        ates.append(ate)

    return np.percentile(ates, [2.5, 97.5])


def main():
    # Define paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")

    # Load already split train and test data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Prepare features and labels
    features = [col for col in train_df.columns if col not in ['index', 'user id', 'converted']]
    X_train = train_df[features].values
    y_train = train_df['converted'].values

    X_test = test_df[features].values
    y_test = test_df['converted'].values

    # Train baseline logistic regression
    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    pred_probs = clf.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    acc = accuracy_score(y_test, preds)
    print(f"Baseline logistic regression accuracy: {acc:.4f}")

    # Classification report (handle imbalance)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, target_names=["Non-Converter", "Converter"]))

    # Dummy classifier benchmark
    dummy = DummyClassifier(strategy="stratified").fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)
    print("\n=== Dummy Classifier Benchmark ===")
    print(classification_report(y_test, dummy_preds, target_names=["Non-Converter", "Converter"]))

    # Save accuracy to a text file
    with open(os.path.join(OUTPUT_DIR, "baseline_accuracy.txt"), "w") as f:
        f.write(f"Baseline logistic regression accuracy: {acc:.4f}\n")

    # Plot and save confusion matrix
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix - Baseline Logistic Regression")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
    plt.close()

    # Plot and save ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic - Baseline Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"))
    plt.close()

    # Calculate bootstrap 95% CI for ATE on test data
    test_df_full = pd.read_csv(test_path)
    ci_lower, ci_upper = bootstrap_ate(test_df_full, treatment_col='test_group_psa', outcome_col='converted')
    print(f"ATE 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Save CI results
    with open(os.path.join(OUTPUT_DIR, "ate_bootstrap_ci.txt"), "w") as f:
        f.write(f"ATE 95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]\n")


if __name__ == "__main__":
    main()