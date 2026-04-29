import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import yaml
import shap
from sklearn.metrics import accuracy_score

# Import modular components
from src.data import get_splits
from src.models import load_all_models
from src.selective_engine import selective_engine_with_explain, compute_ece, dynamic_thresholds, compute_tps
from src.drift import ks_drift, psi_score, shap_drift
from src.explain import ExplainerSystem

st.set_page_config(page_title="Production Trustworthy AI: HAR Pipeline", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f1f3f5 100%);
        border-left: 5px solid #2196F3;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .metric-value {font-size: 28px; font-weight: bold; color: #343a40;}
    .metric-label {font-size: 14px; color: #868e96; text-transform: uppercase; letter-spacing: 1px;}
    .tier-accept {border-left-color: #4CAF50;}
    .tier-defer {border-left-color: #FF9800;}
    .tier-reject {border-left-color: #F44336;}
    h1, h2, h3 {color: #2c3e50;}
    </style>
""", unsafe_allow_html=True)

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Cache data and models
@st.cache_data
def load_and_sample_data():
    X_train, y_train, X_cal, y_cal, X_eval, y_eval = get_splits()
    # Sample 500 records for the dashboard to keep it real-time interactive
    idx = np.random.choice(len(X_eval), size=min(500, len(X_eval)), replace=False)
    X_dash = X_eval.iloc[idx].reset_index(drop=True)
    y_dash = y_eval.iloc[idx].reset_index(drop=True)
    X_bg = joblib.load('artifacts/X_bg.joblib')
    return X_dash, y_dash, X_bg, X_train

@st.cache_resource
def setup_system():
    lr_model, xgb_isotonic, nn_wrapper, le = load_all_models()
    X_dash, y_dash, X_bg, X_train = load_and_sample_data()
    explainer_system = ExplainerSystem(xgb_isotonic.estimator, X_bg)
    return lr_model, xgb_isotonic, nn_wrapper, le, explainer_system, X_dash, y_dash, X_train

with st.spinner("Initializing Production System..."):
    lr_model, xgb_isotonic, nn_wrapper, le, explainer_system, X_dash, y_dash, X_train = setup_system()

# UI Layout
st.title("🛡️ Production Trustworthy AI: HAR System")
st.markdown("""
A research-grade Human Activity Recognition pipeline.

**Pipeline Flow:**
1. Input arrives
2. Drift monitoring runs in parallel
3. Model prediction
4. Calibration / confidence evaluation
5. Selective decision (accept / defer / reject)
6. Explainability checks

We separate prediction confidence from trust using calibration, data stability, and explanation consistency.
""")

# Sidebar Controls
st.sidebar.header("⚙️ Configuration")
threshold_mode = st.sidebar.radio("Threshold Mode", ["Manual", "Dynamic (Adaptive)"], help="We enforce τ_high > τ_low to preserve a valid decision boundary. In degenerate cases where percentile thresholds collapse (e.g., identical confidence values), a small epsilon ensures strict ordering.")

if threshold_mode == "Manual":
    tau_high = st.sidebar.slider("High Confidence (τ_high)", 0.5, 1.0, config['thresholds']['tau_high'], 0.01)
    tau_low = st.sidebar.slider("Fallback Threshold (τ_low)", 0.3, 0.9, config['thresholds']['tau_low'], 0.01)
else:
    # We will compute dynamic thresholds dynamically later
    tau_high, tau_low = 0.8, 0.4 
    st.sidebar.info("Thresholds are computed dynamically using percentiles of the confidence distribution.")
    st.sidebar.markdown("<small>Percentiles define relative ranking of predictions, while calibration ensures that confidence values are meaningful in absolute terms.</small>", unsafe_allow_html=True)

sigma = st.sidebar.slider("Simulate Sensor Noise (σ)", 0.0, 1.0, 0.0, 0.05)
enable_trust_check = st.sidebar.checkbox("Enable Explanation Consistency Check", value=True, help="Prediction rejected due to unstable or inconsistent explanation patterns.")

# Generate Data
np.random.seed(42)
X_noisy = X_dash + np.random.normal(0, sigma, X_dash.shape) if sigma > 0 else X_dash.copy()

# Base Predictions
lr_proba = lr_model.predict_proba(X_noisy)
lr_conf = np.max(lr_proba, axis=1)
lr_pred = lr_model.predict(X_noisy)

xgb_proba = xgb_isotonic.predict_proba(X_noisy)
xgb_conf = np.max(xgb_proba, axis=1)
xgb_pred = le.inverse_transform(np.argmax(xgb_proba, axis=1))

nn_proba = nn_wrapper.predict_proba(X_noisy)
nn_conf = np.max(nn_proba, axis=1)
nn_pred = nn_wrapper.predict(X_noisy)

# Compute dynamic thresholds if enabled
if threshold_mode == "Dynamic (Adaptive)":
    tau_high, tau_low = dynamic_thresholds(lr_conf, y_dash.values)
    st.sidebar.success(f"Adaptive τ_high: {tau_high:.2f}\nAdaptive τ_low: {tau_low:.2f}")

# Precompute SHAP for fallback (used in selective engine and tabs)
shap_values_test = explainer_system.explain_samples(X_noisy)

# Run Selective Engine
res = selective_engine_with_explain(
    lr_conf, lr_pred, xgb_conf, xgb_pred, y_dash, 
    tau_high, tau_low, 
    explainer_system if enable_trust_check else type('Dummy', (), {'is_explanation_unusual': lambda self, a, b: False})(),
    shap_values_test
)

# Render Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Selective Prediction Engine", "Model Comparison", "Explainability (SHAP)", "Drift Detection"])

with tab1:
    st.header("Decision Tiers Breakdown")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card tier-accept"><div class="metric-label">🟢 ACCEPT (Tier 1 - LR)</div><div class="metric-value">{res["n_accept"]}</div><div style="color:gray;">Handled directly</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card tier-defer"><div class="metric-label">🟠 DEFER (Tier 2 - XGBoost)</div><div class="metric-value">{res["n_defer"]}</div><div style="color:gray;">Escalated to robust fallback</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card tier-reject"><div class="metric-label">🔴 REJECT (Low Conf)</div><div class="metric-value">{res["n_reject_conf"]}</div><div style="color:gray;">Uncertainty too high</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card tier-reject" style="border-left-color: purple;"><div class="metric-label">🟣 REJECT (Low Explanation Trust)</div><div class="metric-value">{res["n_reject_explain"]}</div><div style="color:gray;">Explanation trust failed</div></div>', unsafe_allow_html=True)
        
    st.markdown("### Performance & Trust Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Coverage", f"{res['coverage']*100:.1f}%")
    m2.metric("Selective Accuracy", f"{res['sel_acc']*100:.2f}%")
    m3.metric("ECE (Selective)", f"{res['ece_sel']:.4f}")
    m4.metric("TPS Score", f"{res['tps']:.4f}")

    # Coverage vs Accuracy Plot
    st.markdown("### Tradeoff Analysis")
    fig, ax = plt.subplots(figsize=(8, 3))
    # Simple curve generation for current threshold range
    taus = np.linspace(0.4, 0.9, 20)
    covs, accs = [], []
    for t in taus:
        r = selective_engine_with_explain(lr_conf, lr_pred, xgb_conf, xgb_pred, y_dash, t, min(t, 0.4), explainer_system, shap_values_test)
        covs.append(r['coverage']*100)
        accs.append(r['sel_acc']*100)
    ax.plot(covs, accs, marker='o', color='#2196F3')
    ax.axvline(res['coverage']*100, color='red', linestyle='--', label='Current Operating Point')
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Selective Accuracy (%)")
    ax.set_title("Coverage vs Accuracy Tradeoff Curve")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Baseline vs Deep Learning Models")
    st.info("Note on Deep Learning: Neural networks tend to produce overconfident probabilities due to overfitting and lack of proper calibration, which increases ECE. This demonstrates why they are less reliable out-of-the-box compared to calibrated models.")
    
    # Calculate metrics for all 3 models directly (100% coverage baseline)
    lr_acc = accuracy_score(y_dash, lr_pred)
    xgb_acc = accuracy_score(y_dash, xgb_pred)
    nn_acc = accuracy_score(y_dash, nn_pred)
    
    lr_ece = compute_ece(y_dash, lr_proba, lr_pred)
    xgb_ece = compute_ece(y_dash, xgb_proba, xgb_pred)
    nn_ece = compute_ece(y_dash, nn_proba, nn_pred)
    
    lr_tps = compute_tps(1.0, lr_acc, lr_ece)
    xgb_tps = compute_tps(1.0, xgb_acc, xgb_ece)
    nn_tps = compute_tps(1.0, nn_acc, nn_ece)

    comp_df = pd.DataFrame({
        "Model": ["Logistic Regression", "XGBoost", "PyTorch MLP"],
        "Accuracy": [f"{lr_acc*100:.2f}%", f"{xgb_acc*100:.2f}%", f"{nn_acc*100:.2f}%"],
        "ECE": [f"{lr_ece:.4f}", f"{xgb_ece:.4f}", f"{nn_ece:.4f}"],
        "TPS": [f"{lr_tps:.4f}", f"{xgb_tps:.4f}", f"{nn_tps:.4f}"]
    })
    st.table(comp_df)
    
    models = ["Logistic Regression", "XGBoost", "PyTorch MLP"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(models, [lr_acc, xgb_acc, nn_acc], color=['#bbdefb', '#64b5f6', '#1976d2'])
    axes[0].set_title("Accuracy")
    axes[1].bar(models, [lr_ece, xgb_ece, nn_ece], color=['#ffcdd2', '#e57373', '#d32f2f'])
    axes[1].set_title("ECE (Lower is better)")
    axes[2].bar(models, [lr_tps, xgb_tps, nn_tps], color=['#c8e6c9', '#81c784', '#388e3c'])
    axes[2].set_title("TPS (Higher is better)")
    st.pyplot(fig)

with tab3:
    st.header("Global & Local Explainability")
    
    st.subheader("1. Global SHAP Summary")
    st.markdown("Features pushing predictions across all classes. (Bee swarm plot)")
    fig = plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values_test.values[:,:,0], X_noisy, show=False, plot_type="dot", max_display=10)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("2. Error Analysis (Misclassified Samples)")
    correct_mask = (xgb_pred == y_dash.values)
    wrong_idx = np.where(~correct_mask)[0]
    
    if len(wrong_idx) > 0:
        sel_idx = st.selectbox("Select a misclassified sample:", wrong_idx)
        actual_class = y_dash.iloc[sel_idx]
        pred_class = xgb_pred[sel_idx]
        
        st.error(f"**Failed Prediction:** Model predicted **{pred_class}**, but actual was **{actual_class}**.")
        
        # Natural Language Explanation
        nl_text = explainer_system.generate_nl_explanation(
            shap_values_test[sel_idx].values, 
            list(le.classes_).index(pred_class), 
            pred_class, 
            X_noisy.columns, 
            X_noisy.iloc[sel_idx]
        )
        st.info("**Why did it make this mistake?**\n" + nl_text)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap_values_test[sel_idx, :, list(le.classes_).index(pred_class)], show=False)
        st.pyplot(fig)
    else:
        st.success("No misclassified samples in the current subset!")

with tab4:
    st.header("Data Drift & Explanation Stability")
    
    with st.spinner("Calculating Drift..."):
        psi_res = psi_score(X_train.iloc[:, :50], X_noisy.iloc[:, :50]) 
        ks_res = ks_drift(X_train.iloc[:, :50], X_noisy.iloc[:, :50])
        shap_res = shap_drift(explainer_system, shap_values_test)
        
    c1, c2, c3 = st.columns(3)
    v_color = "#4CAF50" if psi_res['verdict'] == "STABLE" else "#FF9800" if psi_res['verdict'] == "MONITOR" else "#F44336"
    c1.markdown(f'<div class="metric-card" style="border-color:{v_color};"><div class="metric-label">Input Drift (PSI)</div><div class="metric-value" style="color:{v_color};">{psi_res["verdict"]}</div></div>', unsafe_allow_html=True)
    
    s_color = "#4CAF50" if shap_res['verdict'] == "STABLE" else "#FF9800" if shap_res['verdict'] == "MONITOR" else "#F44336"
    c2.markdown(f'<div class="metric-card" style="border-color:{s_color};"><div class="metric-label">Explanation Drift</div><div class="metric-value" style="color:{s_color};">{shap_res["verdict"]}</div></div>', unsafe_allow_html=True)
    
    c3.metric("KS Drifted Features (Top 50)", f"{ks_res['n_drifted']} ({ks_res['drift_fraction']*100:.1f}%)")
    
    st.markdown("### Explanation Drift Analysis")
    st.info("Explanation drift indicates that the features influencing model predictions have changed compared to training. This reduces trust in predictions and may trigger rejection or fallback decisions.")
