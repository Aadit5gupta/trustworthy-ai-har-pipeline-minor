import numpy as np

def compute_ece(y_true, y_proba_mat, predicted_labels=None, n_bins=10):
    confidence = np.max(y_proba_mat, axis=1)
    if predicted_labels is None:
        predicted = np.argmax(y_proba_mat, axis=1) + 1
    else:
        predicted = predicted_labels
    correct = (predicted == np.asarray(y_true))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.
    for b in range(n_bins):
        mask = (confidence >= bin_edges[b]) & (confidence < bin_edges[b+1])
        if mask.sum() > 0:
            ece += mask.sum() * abs(correct[mask].mean() - confidence[mask].mean())
    return ece / len(y_true)

def compute_tps(coverage, sel_acc, ece):
    return coverage * sel_acc / (1 + ece)

def dynamic_thresholds(conf_lr, y_true):
    # Thresholds are computed dynamically using percentiles of the confidence distribution.
    # Percentiles determine relative ranking of predictions.
    tau_high = np.percentile(conf_lr, 20)
    tau_low = np.percentile(conf_lr, 5)
    
    # We enforce τ_high > τ_low to preserve a valid decision boundary. 
    # In degenerate cases where percentile thresholds collapse (e.g., identical confidence values), 
    # a small epsilon ensures strict ordering.
    if tau_low >= tau_high:
        tau_high = min(1.0, tau_low + 1e-5)
        
    return tau_high, tau_low

def selective_engine_with_explain(conf_lr, pred_lr, conf_fallback, pred_fallback, y_true, tau_high, tau_low, explainer_system, shap_values_fallback):
    """
    Three-tier selective prediction.
    ACCEPT -> conf_lr >= tau_high : LR prediction
    DEFER -> tau_low <= conf_lr < tau_high : escalate to fallback
    REJECT -> conf_lr < tau_low : abstain
    
    NEW: Check Explanation Stability Score (ESS).
    If ESS is low (unstable reasoning), we REJECT it even if it was DEFER.
    """
    n = len(conf_lr)
    decision = np.empty(n, dtype=object)
    final_pred = np.empty(n, dtype=int)
    
    for i in range(n):
        if conf_lr[i] >= tau_high:
            decision[i] = 'accept'
            final_pred[i] = pred_lr[i]
        elif conf_lr[i] >= tau_low:
            # Check explanation consistency for the fallback prediction
            unusual = explainer_system.is_explanation_unusual(shap_values_fallback[i].values, int(pred_fallback[i]) - 1)
            if unusual:
                decision[i] = 'reject_explain'
                final_pred[i] = -1
            else:
                decision[i] = 'defer'
                final_pred[i] = pred_fallback[i]
        else:
            decision[i] = 'reject_conf'
            final_pred[i] = -1
            
    y_true_arr = np.asarray(y_true)
    accepted_mask = (decision == 'accept') | (decision == 'defer')
    covered = accepted_mask.sum()
    coverage = covered / n
    
    sel_acc = (final_pred[accepted_mask] == y_true_arr[accepted_mask]).mean() if covered > 0 else 0.
        
    conf_covered = np.where(decision == 'accept', conf_lr, np.where(decision == 'defer', conf_fallback, 0.0))
    correct_cov = (final_pred[accepted_mask] == y_true_arr[accepted_mask])
    conf_cov_vals = conf_covered[accepted_mask]
    
    ece_sel = 0.
    if covered > 0:
        bin_edges = np.linspace(0, 1, 11)
        for b in range(10):
            mask = (conf_cov_vals >= bin_edges[b]) & (conf_cov_vals < bin_edges[b+1])
            if mask.sum() > 0:
                ece_sel += mask.sum() * abs(correct_cov[mask].mean() - conf_cov_vals[mask].mean())
        ece_sel /= covered
        
    tps = compute_tps(coverage, sel_acc, ece_sel) if covered > 0 else 0.
    
    return {
        'decision': decision, 'final_pred': final_pred,
        'n_accept': (decision == 'accept').sum(), 
        'n_defer': (decision == 'defer').sum(),
        'n_reject_conf': (decision == 'reject_conf').sum(),
        'n_reject_explain': (decision == 'reject_explain').sum(),
        'coverage': coverage, 'sel_acc': sel_acc, 'ece_sel': ece_sel, 'tps': tps
    }
