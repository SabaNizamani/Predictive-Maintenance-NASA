# =============================================================================
# Predictive Maintenance — Modelling & Evaluation
# Client: Siemens AG / BMW Group Industrial Engines Division
# =============================================================================
# Two modelling approaches:
#
#   APPROACH 1 — REGRESSION (predict exact RUL in cycles)
#     Model  : Random Forest Regressor
#     Target : RUL_clipped (capped at 125 cycles)
#     Metric : RMSE, MAE, R²
#
#   APPROACH 2 — CLASSIFICATION (predict failure within 30 cycles)
#     Model  : Random Forest Classifier
#     Target : will_fail_soon (1 = failure within 30 cycles)
#     Metric : Precision, Recall, F1, ROC-AUC
#     Note   : Recall is the KEY metric — missing a real failure is costly!
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, GroupShuffleSplit
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13})

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")

# ── Load Engineered Data ──────────────────────────────────────────────────────
train    = pd.read_csv(os.path.join(DATA, "train_engineered.csv"))
test     = pd.read_csv(os.path.join(DATA, "test_engineered.csv"))
rul_true = pd.read_csv(os.path.join(DATA, "rul_true.csv"))

print("=" * 65)
print("  PREDICTIVE MAINTENANCE — MODELLING")
print("  Client: Siemens AG / BMW Group")
print("=" * 65)
print(f"  Train: {train.shape}  |  Test: {test.shape}")

# ── Feature / Target Split ────────────────────────────────────────────────────
DROP_COLS = ["engine_id", "cycle", "RUL", "RUL_clipped", "will_fail_soon"]
FEATURE_COLS = [c for c in train.columns if c not in DROP_COLS]

X_train = train[FEATURE_COLS]
X_test  = test[FEATURE_COLS]

y_reg   = train["RUL_clipped"]       # regression target
y_cls   = train["will_fail_soon"]    # classification target

print(f"\n  Features used    : {len(FEATURE_COLS)}")
print(f"  Failure rate     : {y_cls.mean():.1%} of training cycles near failure")

# =============================================================================
# APPROACH 1 — REGRESSION: Predict Exact RUL
# =============================================================================
print("\n" + "=" * 65)
print("  APPROACH 1 — REGRESSION (Predict RUL in cycles)")
print("=" * 65)

# Use GroupShuffleSplit so engines don't leak between train/val
# (all cycles of one engine should stay together)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X_train, y_reg, groups=train["engine_id"]))

X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
y_tr, y_val = y_reg.iloc[train_idx],   y_reg.iloc[val_idx]

reg_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
reg_model.fit(X_tr, y_tr)
print("  ✔ Regression model trained")

# Validate on held-out engines
val_preds = reg_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, val_preds))
mae  = mean_absolute_error(y_val, val_preds)
r2   = r2_score(y_val, val_preds)

print(f"\n  Validation Results:")
print(f"    RMSE : {rmse:.2f} cycles")
print(f"    MAE  : {mae:.2f} cycles")
print(f"    R²   : {r2:.4f}")

# Test set predictions
# For test set: last cycle of each engine + true RUL from RUL_FD001.txt
test_last = test.groupby("engine_id").last().reset_index()
X_test_last = test_last[FEATURE_COLS]
test_rul_pred = reg_model.predict(X_test_last)
rul_true_vals = rul_true["RUL"].values

test_rmse = np.sqrt(mean_squared_error(rul_true_vals, test_rul_pred))
test_mae  = mean_absolute_error(rul_true_vals, test_rul_pred)
test_r2   = r2_score(rul_true_vals, test_rul_pred)

print(f"\n  Test Set Results:")
print(f"    RMSE : {test_rmse:.2f} cycles  ← this is your headline metric")
print(f"    MAE  : {test_mae:.2f} cycles")
print(f"    R²   : {test_r2:.4f}")

# Plot: Predicted vs Actual RUL
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(rul_true_vals, test_rul_pred, alpha=0.6,
                color="#00B89F", s=40, edgecolors="white", linewidth=0.5)
max_val = max(rul_true_vals.max(), test_rul_pred.max())
axes[0].plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect prediction")
axes[0].set_xlabel("True RUL (cycles)", fontweight="bold")
axes[0].set_ylabel("Predicted RUL (cycles)", fontweight="bold")
axes[0].set_title(f"Predicted vs Actual RUL\nRMSE = {test_rmse:.1f} cycles | R² = {test_r2:.3f}",
                  fontweight="bold")
axes[0].legend()

# Error distribution
errors = test_rul_pred - rul_true_vals
axes[1].hist(errors, bins=30, color="#1C1C1C", edgecolor="white", alpha=0.8)
axes[1].axvline(0, color="#e74c3c", linestyle="--", lw=2, label="Zero error")
axes[1].axvline(errors.mean(), color="#00B89F", linestyle="--",
                lw=2, label=f"Mean error = {errors.mean():.1f}")
axes[1].set_xlabel("Prediction Error (cycles)", fontweight="bold")
axes[1].set_ylabel("Frequency", fontweight="bold")
axes[1].set_title("Prediction Error Distribution", fontweight="bold")
axes[1].legend()

plt.suptitle("Regression Results — Remaining Useful Life Prediction",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "model_01_regression_results.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: model_01_regression_results.png")

# =============================================================================
# APPROACH 2 — CLASSIFICATION: Failure Within 30 Cycles?
# =============================================================================
print("\n" + "=" * 65)
print("  APPROACH 2 — CLASSIFICATION (Failure within 30 cycles?)")
print("=" * 65)
print("  ⚠  KEY METRIC IS RECALL — missing a real failure is very costly!")

cls_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",   # handle imbalance (fewer failure cycles)
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
cls_model.fit(X_tr, y_cls.iloc[train_idx])
print("  ✔ Classification model trained")

# Validation
val_cls_preds = cls_model.predict(X_val)
val_cls_proba = cls_model.predict_proba(X_val)[:, 1]
y_val_cls = y_cls.iloc[val_idx]

print(f"\n  Validation Results:")
print(f"    Precision : {precision_score(y_val_cls, val_cls_preds):.4f}")
print(f"    Recall    : {recall_score(y_val_cls, val_cls_preds):.4f}  ← most important!")
print(f"    F1 Score  : {f1_score(y_val_cls, val_cls_preds):.4f}")
print(f"    ROC-AUC   : {roc_auc_score(y_val_cls, val_cls_proba):.4f}")
print(f"\n{classification_report(y_val_cls, val_cls_preds, target_names=['OK', 'Near Failure'])}")

# Plots
fig = plt.figure(figsize=(18, 5))
gs  = gridspec.GridSpec(1, 3)

# Confusion matrix
ax1 = fig.add_subplot(gs[0])
cm  = confusion_matrix(y_val_cls, val_cls_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
            xticklabels=["OK", "Near Failure"],
            yticklabels=["OK", "Near Failure"],
            linewidths=1, annot_kws={"size": 14})
ax1.set_title("Confusion Matrix", fontweight="bold")
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

# ROC Curve
ax2 = fig.add_subplot(gs[1])
fpr, tpr, _ = roc_curve(y_val_cls, val_cls_proba)
auc_val      = roc_auc_score(y_val_cls, val_cls_proba)
ax2.plot(fpr, tpr, color="#e74c3c", lw=2, label=f"AUC = {auc_val:.3f}")
ax2.plot([0,1],[0,1], "k--", lw=1)
ax2.fill_between(fpr, tpr, alpha=0.08, color="#e74c3c")
ax2.set_title("ROC Curve", fontweight="bold")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.legend()

# Feature Importances
ax3 = fig.add_subplot(gs[2])
fi  = pd.Series(cls_model.feature_importances_, index=FEATURE_COLS).nlargest(10).sort_values()
colors_fi = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(fi)))
ax3.barh(fi.index, fi.values, color=colors_fi, edgecolor="white")
ax3.set_title("Top 10 Feature Importances", fontweight="bold")
ax3.set_xlabel("Importance")

plt.suptitle("Classification Results — Will Engine Fail Within 30 Cycles?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "model_02_classification_results.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: model_02_classification_results.png")

# =============================================================================
# MAINTENANCE SCHEDULE SIMULATION
# =============================================================================
print("\n" + "=" * 65)
print("  BUSINESS IMPACT — MAINTENANCE SCHEDULE SIMULATION")
print("=" * 65)

# Simulate what happens if we follow model predictions
alert_threshold = 30  # send alert when predicted RUL <= 30 cycles

# Apply to test set (last reading per engine)
results = pd.DataFrame({
    "engine_id":    test_last["engine_id"].values,
    "true_rul":     rul_true_vals,
    "predicted_rul": test_rul_pred.round(0).astype(int),
    "will_fail_soon": (test_rul_pred <= alert_threshold).astype(int),
    "needs_alert":  (rul_true_vals <= alert_threshold).astype(int)
})

correctly_flagged = ((results["will_fail_soon"] == 1) &
                     (results["needs_alert"] == 1)).sum()
missed_failures   = ((results["will_fail_soon"] == 0) &
                     (results["needs_alert"] == 1)).sum()
false_alarms      = ((results["will_fail_soon"] == 1) &
                     (results["needs_alert"] == 0)).sum()
total_near_fail   = results["needs_alert"].sum()

print(f"  Engines truly near failure (<30 cycles) : {total_near_fail}")
print(f"  Correctly flagged by model              : {correctly_flagged}")
print(f"  Missed failures (DANGEROUS)             : {missed_failures}")
print(f"  False alarms (wasteful but safe)        : {false_alarms}")
print(f"\n  Catch rate: {correctly_flagged/max(total_near_fail,1):.1%}")

# Save predictions
results.to_csv(os.path.join(OUTPUTS, "maintenance_predictions.csv"), index=False)

# Visualise maintenance schedule
fig, ax = plt.subplots(figsize=(14, 6))
colors_map = {
    (1, 1): "#27AE60",  # correctly flagged → green
    (0, 1): "#E74C3C",  # missed failure → red
    (1, 0): "#F39C12",  # false alarm → orange
    (0, 0): "#95A5A6",  # correctly ok → grey
}
for _, row in results.iterrows():
    key   = (int(row["will_fail_soon"]), int(row["needs_alert"]))
    color = colors_map.get(key, "grey")
    ax.scatter(row["true_rul"], row["predicted_rul"], color=color, s=60,
               edgecolors="white", linewidth=0.5, zorder=3)

max_v = max(results["true_rul"].max(), results["predicted_rul"].max())
ax.plot([0, max_v], [0, max_v], "k--", lw=1, alpha=0.5, label="Perfect prediction")
ax.axvline(30, color="#E74C3C", lw=1.5, linestyle=":", alpha=0.7, label="Failure window (30 cycles)")
ax.axhline(30, color="#E74C3C", lw=1.5, linestyle=":", alpha=0.7)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#27AE60", label=f"✔ Correctly flagged ({correctly_flagged})"),
    Patch(facecolor="#E74C3C", label=f"✗ Missed failure ({missed_failures})"),
    Patch(facecolor="#F39C12", label=f"⚠ False alarm ({false_alarms})"),
    Patch(facecolor="#95A5A6", label="✔ Correctly OK"),
]
ax.legend(handles=legend_elements, loc="upper left")
ax.set_xlabel("True RUL (cycles)", fontweight="bold")
ax.set_ylabel("Predicted RUL (cycles)", fontweight="bold")
ax.set_title("Maintenance Alert System — Test Engine Results\n"
             "Each dot = one engine at last observed cycle",
             fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "model_03_maintenance_schedule.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: model_03_maintenance_schedule.png")

print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY — SIEMENS AG DELIVERABLE")
print("=" * 65)
print(f"""
  REGRESSION MODEL (Predict exact RUL):
    Test RMSE        : {test_rmse:.1f} cycles
    Test MAE         : {test_mae:.1f} cycles
    Test R²          : {test_r2:.4f}

  CLASSIFICATION MODEL (Failure within 30 cycles):
    Precision        : {precision_score(y_val_cls, val_cls_preds):.4f}
    Recall           : {recall_score(y_val_cls, val_cls_preds):.4f}
    F1 Score         : {f1_score(y_val_cls, val_cls_preds):.4f}
    ROC-AUC          : {roc_auc_score(y_val_cls, val_cls_proba):.4f}

  BUSINESS IMPACT:
    Near-failure engines caught : {correctly_flagged}/{total_near_fail}
    Missed dangerous failures   : {missed_failures}
    False alarms                : {false_alarms}

  CONCLUSION:
    The model can predict engine failure with strong accuracy,
    enabling Siemens/BMW to replace scheduled maintenance with
    condition-based maintenance — reducing costs and downtime.
""")
print("  ✔ All outputs saved to /outputs/")
print("  ✔ MODELLING PIPELINE COMPLETE")
