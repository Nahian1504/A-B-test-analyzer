import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.base import clone
import joblib
import os

class TLearner:
    """
    T-Learner fits separate models for treated and control groups
    to estimate potential outcomes, then computes uplift as their difference.
    """

    def __init__(self, base_model=None):
        self.base_model = base_model or LogisticRegression(max_iter=200)
        self.model_treated = clone(self.base_model)
        self.model_control = clone(self.base_model)
        self.fitted = False

    def fit(self, X, treatment, y):
        treatment = np.array(treatment)
        # Fit model for treated group
        self.model_treated.fit(X[treatment == 1], y[treatment == 1])
        # Fit model for control group
        self.model_control.fit(X[treatment == 0], y[treatment == 0])
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
        pred_treated = self._predict_proba_safe(self.model_treated, X)
        pred_control = self._predict_proba_safe(self.model_control, X)
        uplift = pred_treated - pred_control
        return uplift

    def _predict_proba_safe(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            # fallback for models without predict_proba
            return model.predict(X)


class SLearner:
    """
    S-Learner fits a single model on features + treatment indicator,
    then predicts outcomes with treatment toggled on/off to get uplift.
    """
    
    def __init__(self, base_model=None):
        self.base_model = base_model or LogisticRegression(max_iter=200)
        self.model = clone(self.base_model)
        self.fitted = False
        self.n_features = None  # Track original feature count
        self.feature_names_ = None  # Store feature names


    def fit(self, X, treatment, y):
        treatment = np.array(treatment).reshape(-1, 1)
        self.n_features = X.shape[1]  # Store original feature count
        if hasattr(X, 'columns'):  
            self.feature_names_ = list(X.columns)
        X_aug = np.hstack((X, treatment))
        self.model.fit(X_aug, y)
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
        
        n_samples = X.shape[0]
        treated_feat = np.hstack((X, np.ones((n_samples, 1))))
        control_feat = np.hstack((X, np.zeros((n_samples, 1))))

        pred_treated = self._predict_proba_safe(treated_feat)
        pred_control = self._predict_proba_safe(control_feat)

        uplift = pred_treated - pred_control
        return uplift

    def predict_proba(self, X):
        """Predict probability for SHAP analysis"""
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")
            
        # Handle both augmented (10 features) and original (9 features) input
        if X.shape[1] == self.n_features + 1:  # Already augmented
            return self._predict_proba_safe(X)
        elif X.shape[1] == self.n_features:    # Need to augment
            # Default to treatment=1 for SHAP (arbitrary choice)
            X_aug = np.hstack((X, np.ones((X.shape[0], 1))))
            return self._predict_proba_safe(X_aug)
        else:
            raise ValueError(f"Input has {X.shape[1]} features, expected {self.n_features} or {self.n_features+1}")

    def _predict_proba_safe(self, X):
        """Safe prediction that works with models that may or may not have predict_proba"""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)

    def save_model(self, filepath):
        if not self.fitted:
            raise RuntimeError("Train the model before saving.")
        joblib.dump(self, filepath)  # Save entire SLearner object

    @staticmethod
    def load_model(filepath):
        return joblib.load(filepath)



class XLearner:
    """
    X-Learner uses outcome models for treated/control and
    regression meta learners to estimate treatment effects.
    """

    def __init__(self, outcome_model=None, meta_model=None):
        self.outcome_model_treated = clone(outcome_model or LogisticRegression(max_iter=200))
        self.outcome_model_control = clone(outcome_model or LogisticRegression(max_iter=200))
        # Meta learners should be regressors for continuous targets
        self.meta_model_treated = clone(meta_model or LinearRegression())
        self.meta_model_control = clone(meta_model or LinearRegression())
        self.fitted = False

    def fit(self, X, treatment, y):
        treatment = np.array(treatment)
        # Stage 1: fit outcome models
        self.outcome_model_treated.fit(X[treatment == 1], y[treatment == 1])
        self.outcome_model_control.fit(X[treatment == 0], y[treatment == 0])

        # Predict counterfactuals for treated and control groups
        mu0 = self._predict_proba_safe(self.outcome_model_control, X[treatment == 1])  # control outcome for treated
        mu1 = self._predict_proba_safe(self.outcome_model_treated, X[treatment == 0])  # treated outcome for control

        # Calculate imputed treatment effects (pseudo outcomes)
        D_treated = y[treatment == 1] - mu0  # observed - predicted control
        D_control = mu1 - y[treatment == 0]  # predicted treated - observed control

        # Stage 2: fit meta learners on pseudo outcomes
        self.meta_model_treated.fit(X[treatment == 1], D_treated)
        self.meta_model_control.fit(X[treatment == 0], D_control)

        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("You must fit the model before prediction.")

        mu0 = self._predict_proba_safe(self.outcome_model_control, X)
        mu1 = self._predict_proba_safe(self.outcome_model_treated, X)

        tau0 = self.meta_model_control.predict(X)
        tau1 = self.meta_model_treated.predict(X)

        # Simple weighting by propensity 0.5 (can be improved with real propensities)
        propensity = 0.5
        uplift = propensity * tau0 + (1 - propensity) * tau1

        return np.array(uplift).flatten()

    def _predict_proba_safe(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    print("Testing uplift models with dummy data...")

    X, y = make_classification(n_samples=500, n_features=5, random_state=42)
    treatment = np.random.binomial(1, 0.5, size=500)

    print("Fitting T-Learner...")
    t_learner = TLearner()
    t_learner.fit(X, treatment, y)
    uplift_t = t_learner.predict(X)
    print(f"T-Learner uplift sample: {uplift_t[:5]}")

    print("Fitting S-Learner...")
    s_learner = SLearner()
    s_learner.fit(X, treatment, y)
    uplift_s = s_learner.predict(X)
    print(f"S-Learner uplift sample: {uplift_s[:5]}")

    print("Fitting X-Learner...")
    x_learner = XLearner()
    x_learner.fit(X, treatment, y)
    uplift_x = x_learner.predict(X)
    print(f"X-Learner uplift sample: {uplift_x[:5]}")

    print("All done.")