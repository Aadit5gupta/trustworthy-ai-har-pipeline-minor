import shap
import numpy as np
import pandas as pd
import joblib

class ExplainerSystem:
    def __init__(self, model, X_bg):
        # We assume model is the calibrated CV's estimator, i.e. xgb_isotonic.estimator
        self.explainer = shap.TreeExplainer(model)
        # Precompute global SHAP on background data (e.g., train set or eval subset)
        self.shap_values_global = self.explainer(X_bg)
        
        # Calculate mean absolute SHAP per class per feature
        # shap_values_global.values is (N, Features, Classes)
        self.global_importance = np.abs(self.shap_values_global.values).mean(axis=0) # (Features, Classes)

    def explain_samples(self, X):
        return self.explainer(X)

    def is_explanation_unusual(self, local_shap_vals, predicted_class_idx, threshold=0.3):
        """
        We quantify explanation stability using an Explanation Stability Score (ESS), 
        based on consistency of top contributing features. Low ESS indicates unstable reasoning 
        and triggers rejection.
        
        Note: Low ESS may also occur due to correlated or redundant features, so it should be 
        treated as a heuristic signal rather than a strict guarantee of explanation validity.
        """
        # SHAP values for the predicted class
        local_class_shap = np.abs(local_shap_vals[:, predicted_class_idx])
        global_class_shap = self.global_importance[:, predicted_class_idx]
        
        k = 5
        top_local = set(np.argsort(local_class_shap)[-k:])
        top_global = set(np.argsort(global_class_shap)[-k:])
        
        intersection = top_local.intersection(top_global)
        ess = len(intersection) / k # Explanation Stability Score
        
        # Low ESS indicates unstable reasoning
        return ess < 0.4 # Less than 2 out of 5 overlapping means ESS < 0.4

    def generate_nl_explanation(self, local_shap_vals, predicted_class_idx, class_name, feature_names, X_sample):
        """
        Convert SHAP into natural language.
        """
        local_class_shap = local_shap_vals[:, predicted_class_idx]
        top_idx = np.argsort(local_class_shap)[-3:][::-1] # Top 3 positive contributors
        
        reasons = []
        for idx in top_idx:
            feat_name = feature_names[idx] if feature_names is not None else f"Feature {idx}"
            feat_val = X_sample.iloc[idx] if isinstance(X_sample, pd.Series) else X_sample[idx]
            reasons.append(f"• High contribution from {feat_name} (Value: {feat_val:.2f})")
            
        nl_text = f"Prediction: {class_name}\nReason:\n" + "\n".join(reasons)
        return nl_text
