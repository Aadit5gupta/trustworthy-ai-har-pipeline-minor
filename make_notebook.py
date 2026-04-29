import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# HAR Trustworthy Prediction Pipeline

**Pipeline:** Data Loading → Preprocessing → Baseline Models → Calibration → Selective Prediction → Drift Detection → Experiments → Explainability

**Metric introduced:** TPS = (Coverage × Selective Accuracy) / (1 + ECE)"""))

cells.append(nbf.v4.new_code_cell("""import sys
!{sys.executable} -m pip install xgboost scikit-learn seaborn matplotlib pandas numpy shap --quiet
print("✅ All packages installed")"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.special import softmax
from scipy.optimize import minimize_scalar
import xgboost as xgb
import shap

print("✅ All imports successful")
print(f" sklearn : {__import__('sklearn').__version__}")
print(f" xgboost : {xgb.__version__}")
print(f" shap : {shap.__version__}")"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Data Loading"))

cells.append(nbf.v4.new_code_cell("""BASE_PATH = "./UCI-HAR Dataset/"

def load_data(split):
    X = pd.read_csv(f"{BASE_PATH}{split}/X_{split}.txt", delim_whitespace=True, header=None)
    y = pd.read_csv(f"{BASE_PATH}{split}/y_{split}.txt", header=None).squeeze()
    return X, y

X_train, y_train = load_data("train")
X_test, y_test = load_data("test")

labels = {1: "Walking", 2: "Walking Upstairs", 3: "Walking Downstairs",
          4: "Sitting", 5: "Standing", 6: "Laying"}
y_train_named = y_train.map(labels)
y_test_named = y_test.map(labels)

print(f"✅ Data loaded!")
print(f" Training samples : {X_train.shape[0]}")
print(f" Test samples     : {X_test.shape[0]}")
print(f" Features         : {X_train.shape[1]}")
print(f" Classes          : {list(labels.values())}")"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Preprocessing & EDA"))

cells.append(nbf.v4.new_code_cell("""# ── Class distribution ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2196F3','#4CAF50','#FF9800','#F44336','#9C27B0','#00BCD4']

for ax, named, title in zip(axes, [y_train_named, y_test_named], ["Training Set", "Test Set"]):
    counts = named.value_counts()
    ax.bar(counts.index, counts.values, color=colors, edgecolor='white')
    ax.set_title(f"{title} — Activity Distribution", fontsize=13, fontweight='bold')
    ax.set_xlabel("Activity"); ax.set_ylabel("Number of Samples")
    ax.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# ── Feature heatmap (first 50 features) ─────────────────────
feature_means = X_train.copy()
feature_means['activity'] = y_train_named.values
class_means = feature_means.groupby('activity').mean().iloc[:, :50]

plt.figure(figsize=(16, 5))
sns.heatmap(class_means, cmap='RdYlBu_r', xticklabels=False, yticklabels=True, cbar_kws={'label': 'Mean Feature Value'})
plt.title("Mean Feature Values per Activity Class (First 50 Features)", fontsize=13, fontweight='bold')
plt.ylabel("Activity")
plt.tight_layout()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# ── Train/Calibration/Eval split ────────────────────────────
# Calibrate on 30% of test, evaluate on remaining 70%
X_cal, X_eval, y_cal, y_eval = train_test_split(
    X_test, y_test, test_size=0.70, random_state=42, stratify=y_test
)

print(f"Split summary:")
print(f" Train : {X_train.shape[0]} samples")
print(f" Cal   : {len(X_cal)} samples (30% of test — for calibration)")
print(f" Eval  : {len(X_eval)} samples (70% of test — held out)")"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Shared Utilities\nAll reusable functions defined once here and reused throughout the notebook."))

cells.append(nbf.v4.new_code_cell("""def compute_ece(y_true, y_proba_mat, n_bins=10):
    confidence = np.max(y_proba_mat, axis=1)
    predicted = np.argmax(y_proba_mat, axis=1) + 1
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

def plot_reliability(ax, y_true, y_proba, title, color='#2196F3', n_bins=10):
    confidence = np.max(y_proba, axis=1)
    predicted = np.argmax(y_proba, axis=1) + 1
    correct = (predicted == np.asarray(y_true))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_sizes = [], [], []
    for b in range(n_bins):
        mask = (confidence >= bin_edges[b]) & (confidence < bin_edges[b+1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidence[mask].mean())
            bin_sizes.append(mask.sum())
    ece = sum(s * abs(a - c) for a, c, s in zip(bin_accs, bin_confs, bin_sizes)) / len(y_true)
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.75, color=color, label='Actual Accuracy', edgecolor='white')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect')
    ax.fill_between(bin_confs, bin_confs, bin_accs, alpha=0.2, color='red')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Confidence", fontsize=10)
    ax.set_ylabel("Actual Accuracy", fontsize=10)
    ax.set_title(f"{title}\\nECE = {ece:.4f}", fontsize=11, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    return ece

def selective_engine(conf_lr, pred_lr, conf_fallback, pred_fallback, y_true, tau_high, tau_low):
    n = len(conf_lr)
    decision = np.empty(n, dtype=object)
    final_pred = np.empty(n, dtype=int)
    for i in range(n):
        if conf_lr[i] >= tau_high:
            decision[i] = 'accept'
            final_pred[i] = pred_lr[i]
        elif conf_lr[i] >= tau_low:
            decision[i] = 'defer'
            final_pred[i] = pred_fallback[i]
        else:
            decision[i] = 'reject'
            final_pred[i] = -1
    y_true_arr = np.asarray(y_true)
    accepted_mask = decision != 'reject'
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
        'n_accept': (decision == 'accept').sum(), 'n_defer': (decision == 'defer').sum(), 'n_reject': (decision == 'reject').sum(),
        'coverage': coverage, 'sel_acc': sel_acc, 'ece_sel': ece_sel, 'tps': tps
    }

def ks_drift(X_ref, X_test_df, ks_threshold=0.10):
    ks_stats = np.array([stats.ks_2samp(X_ref.iloc[:, f].values, X_test_df.iloc[:, f].values)[0] for f in range(X_ref.shape[1])])
    n_drifted = np.sum(ks_stats > ks_threshold)
    return {'n_drifted': int(n_drifted), 'drift_fraction': n_drifted / X_ref.shape[1], 'mean_ks_stat': float(np.mean(ks_stats)), 'max_ks_stat': float(np.max(ks_stats)), 'ks_stats': ks_stats}

def psi_score(X_ref, X_test_df, n_bins=10, psi_cap=0.50):
    bin_edges = np.linspace(-4, 4, n_bins + 1)
    psi_vals = []
    for f in range(X_ref.shape[1]):
        ref_col = X_ref.iloc[:, f].values
        tst_col = X_test_df.iloc[:, f].values
        mu, std = ref_col.mean(), ref_col.std()
        if std < 1e-9: continue
        ref_z = np.clip((ref_col - mu) / std, -4, 4)
        tst_z = np.clip((tst_col - mu) / std, -4, 4)
        ref_cnt = np.histogram(ref_z, bins=bin_edges)[0]
        tst_cnt = np.histogram(tst_z, bins=bin_edges)[0]
        ref_smooth = np.sqrt(len(ref_z) / n_bins)
        tst_smooth = np.sqrt(len(tst_z) / n_bins)
        ref_pct = (ref_cnt + ref_smooth) / (len(ref_z) + n_bins * ref_smooth)
        tst_pct = (tst_cnt + tst_smooth) / (len(tst_z) + n_bins * tst_smooth)
        psi = float(np.sum((tst_pct - ref_pct) * np.log(tst_pct / ref_pct)))
        psi_vals.append(min(psi, psi_cap))
    mean_psi = float(np.mean(psi_vals))
    verdict = "STABLE" if mean_psi < 0.10 else "MONITOR" if mean_psi < 0.25 else "DRIFT DETECTED"
    return {'mean_psi': mean_psi, 'max_psi': float(np.max(psi_vals)), 'psi_vals': np.array(psi_vals), 'verdict': verdict}

def evaluate_model(model, X, y_true, model_name="Model"):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    acc = accuracy_score(y_true, y_pred)
    ece_val = compute_ece(y_true, y_proba)
    conf = np.max(y_proba, axis=1)
    correct = (y_pred == np.asarray(y_true))
    overconf = np.sum((conf > 0.9) & ~correct)
    print(f"{model_name:<25} acc={acc*100:.2f}% ECE={ece_val:.4f} overconf={overconf}")
    return {'accuracy': acc, 'ece': ece_val, 'confidence': conf, 'predictions': y_pred, 'probabilities': y_proba, 'correct': correct, 'overconfident': overconf}"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Baseline Models\nTrain Logistic Regression and Random Forest; compare accuracy, ECE, and overconfident errors."))

cells.append(nbf.v4.new_code_cell("""print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs')
lr_model.fit(X_train, y_train)
print("✅ LR trained")

print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("✅ RF trained")"""))

cells.append(nbf.v4.new_code_cell("""print("=" * 55)
print(" BASELINE EVALUATION — TEST SET")
print("=" * 55)
lr_eval = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
rf_eval = evaluate_model(rf_model, X_test, y_test, "Random Forest")"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Probability Calibration\nFix overconfident predictions using Temperature Scaling."))

cells.append(nbf.v4.new_code_cell("""class TemperatureScaling:
    def __init__(self):
        self.T = 1.0
    def fit(self, logits, y_true_0indexed):
        def nll(T):
            scaled = softmax(logits / T, axis=1)
            scaled = np.clip(scaled, 1e-9, 1.0)
            return -np.sum(np.log(scaled[np.arange(len(y_true_0indexed)), y_true_0indexed])) / len(y_true_0indexed)
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.T = result.x
        return self
    def calibrate(self, logits):
        return softmax(logits / self.T, axis=1)

# Fit calibration
lr_logits_eval = lr_model.predict_log_proba(X_eval)
rf_logits_eval = rf_model.predict_log_proba(X_eval)

ts_rf = TemperatureScaling()
ts_rf.fit(rf_model.predict_log_proba(X_cal), y_cal.values - 1)

final_lr_proba = lr_model.predict_proba(X_eval)
final_lr_conf = np.max(final_lr_proba, axis=1)
final_lr_pred = lr_model.predict(X_eval)

final_rf_proba = ts_rf.calibrate(rf_model.predict_log_proba(X_eval))
final_rf_conf = np.max(final_rf_proba, axis=1)
final_rf_pred = rf_model.predict(X_eval)

print(f"✅ RF Temperature Scaling fitted: T = {ts_rf.T:.4f}")"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Selective Prediction Engine\nThree-tier system: **ACCEPT** (LR, high confidence) → **DEFER** (XGBoost fallback) → **REJECT** (abstain)."))

cells.append(nbf.v4.new_code_cell("""print("Training XGBoost fallback model...")
le = LabelEncoder()
y_train_xgb = le.fit_transform(y_train)
y_eval_xgb = le.transform(y_eval)
y_cal_xgb = le.transform(y_cal)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_train, y_train_xgb)

xgb_isotonic = CalibratedClassifierCV(xgb_model, cv='prefit', method='isotonic')
xgb_isotonic.fit(X_cal, y_cal_xgb)

final_xgb_proba = xgb_isotonic.predict_proba(X_eval)
final_xgb_conf = np.max(final_xgb_proba, axis=1)
final_xgb_pred = le.inverse_transform(np.argmax(final_xgb_proba, axis=1))

print("✅ XGBoost trained and calibrated")"""))

cells.append(nbf.v4.new_code_cell("""best_tau_high, best_tau_low = 0.55, 0.50
best_result = selective_engine(
    final_lr_conf, final_lr_pred, final_xgb_conf, final_xgb_pred, y_eval, best_tau_high, best_tau_low
)

print(f"ACCEPT: {best_result['n_accept']} | DEFER: {best_result['n_defer']} | REJECT: {best_result['n_reject']}")
print(f"Selective Accuracy: {best_result['sel_acc']*100:.2f}% | Coverage: {best_result['coverage']*100:.1f}%")
print(f"TPS: {best_result['tps']:.4f}")"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Model Serialization\nSaving models and dependencies for use in external applications like a Streamlit dashboard."))

cells.append(nbf.v4.new_code_cell("""import joblib

# Ensure we save models locally for app.py
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(xgb_isotonic, 'xgb_isotonic.joblib')
joblib.dump(le, 'label_encoder.joblib')
print("✅ Models saved successfully.")"""))

nb['cells'] = cells
with open('pipeline.ipynb', 'w') as f:
    nbf.write(nb, f)
print("Notebook pipeline.ipynb created successfully.")
