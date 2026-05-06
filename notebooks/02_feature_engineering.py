# =============================================================================
# Predictive Maintenance — Feature Engineering
# Client: Siemens AG / BMW Group Industrial Engines Division
# =============================================================================
# Goal: Transform raw noisy sensor readings into meaningful ML features
#       that capture engine degradation trends clearly.
#
# Framework:
#   1. Remove constant / useless sensors
#   2. Smooth noisy signals with rolling averages
#   3. Add trend features (rolling min, max, std)
#   4. Create a synthetic Health Index (0=healthy → 1=about to fail)
#   5. Add binary classification target (failure within 30 cycles)
#   6. Normalise features for modelling
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130})

BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")

COLS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# ── Load Data ─────────────────────────────────────────────────────────────────
train = pd.read_csv(os.path.join(DATA, "train_FD001.txt"),
                    sep=r"\s+", header=None, names=COLS)
test  = pd.read_csv(os.path.join(DATA, "test_FD001.txt"),
                    sep=r"\s+", header=None, names=COLS)
rul_true = pd.read_csv(os.path.join(DATA, "RUL_FD001.txt"),
                       header=None, names=["RUL"])

# Add RUL to training data
max_cycles = train.groupby("engine_id")["cycle"].max().reset_index()
max_cycles.columns = ["engine_id", "max_cycle"]
train = train.merge(max_cycles, on="engine_id")
train["RUL"] = train["max_cycle"] - train["cycle"]
train.drop(columns=["max_cycle"], inplace=True)

print("=" * 65)
print("  FEATURE ENGINEERING — NASA C-MAPSS FD001")
print("=" * 65)
print(f"  Train: {train.shape}  |  Test: {test.shape}")

# ── Step 1 — Remove Constant Sensors ─────────────────────────────────────────
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
sensor_std  = train[sensor_cols].std()

# Sensors with near-zero variance carry no information
CONSTANT_SENSORS = sensor_std[sensor_std < 0.01].index.tolist()
USEFUL_SENSORS   = sensor_std[sensor_std >= 0.01].index.tolist()

train.drop(columns=CONSTANT_SENSORS, inplace=True)
test.drop(columns=CONSTANT_SENSORS,  inplace=True)

# Also drop near-constant operational settings (for FD001)
train.drop(columns=["op_setting_3"], inplace=True, errors="ignore")
test.drop(columns=["op_setting_3"],  inplace=True, errors="ignore")

print(f"\n  Step 1 — Removed {len(CONSTANT_SENSORS)} constant sensors: {CONSTANT_SENSORS}")
print(f"           Keeping {len(USEFUL_SENSORS)} useful sensors")

# ── Step 2 — Rolling Average (Smoothing Noisy Sensors) ────────────────────────
# Raw sensor readings are noisy — rolling mean smooths the signal
# Window of 5 cycles gives a good balance between smoothing and responsiveness
WINDOW = 5

def add_rolling_features(df, sensors, window=WINDOW):
    """Add rolling mean, std and min for each sensor per engine."""
    df = df.copy()
    df = df.sort_values(["engine_id", "cycle"])
    for sensor in sensors:
        grp = df.groupby("engine_id")[sensor]
        df[f"{sensor}_rmean"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{sensor}_rstd"]  = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{sensor}_rmin"]  = grp.transform(
            lambda x: x.rolling(window, min_periods=1).min())
        df[f"{sensor}_rmax"]  = grp.transform(
            lambda x: x.rolling(window, min_periods=1).max())
    return df

train = add_rolling_features(train, USEFUL_SENSORS)
test  = add_rolling_features(test,  USEFUL_SENSORS)
print(f"\n  Step 2 — Added rolling mean, std, min, max (window={WINDOW}) for {len(USEFUL_SENSORS)} sensors")
print(f"           New features added: {len(USEFUL_SENSORS) * 4}")

# ── Step 3 — Cycle-Based Features ─────────────────────────────────────────────
# Normalised cycle position within each engine's lifecycle
train["cycle_norm"] = train.groupby("engine_id")["cycle"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

# For test set we don't know max cycle — use cumulative position
test["cycle_norm"] = test.groupby("engine_id")["cycle"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8))

print("\n  Step 3 — Added normalised cycle position (0=new → 1=end of life)")

# ── Step 4 — Synthetic Health Index ──────────────────────────────────────────
# Combine the most correlated sensors into a single degradation score
# Using sensors with strongest correlation to RUL
# Positive correlation with RUL = decreases as engine ages (bad → increasing = deteriorating)
# Negative correlation with RUL = increases as engine ages

KEY_SENSORS_POS = ["sensor_2", "sensor_3", "sensor_4"]   # increase = degrading
KEY_SENSORS_NEG = ["sensor_11", "sensor_15", "sensor_17"] # decrease = degrading

def compute_health_index(df, sensors_pos, sensors_neg):
    """
    Synthetic Health Index:
    - 0 = perfectly healthy
    - 1 = about to fail
    Computed as normalised degradation score from key sensors.
    """
    scaler = MinMaxScaler()
    hi = pd.DataFrame()
    for s in sensors_pos:
        if s in df.columns:
            hi[s] = scaler.fit_transform(df[[s]])  # normalise 0-1
    for s in sensors_neg:
        if s in df.columns:
            hi[s] = 1 - scaler.fit_transform(df[[s]])  # invert
    return hi.mean(axis=1)

train["health_index"] = compute_health_index(train, KEY_SENSORS_POS, KEY_SENSORS_NEG)
test["health_index"]  = compute_health_index(test,  KEY_SENSORS_POS, KEY_SENSORS_NEG)

print("\n  Step 4 — Created synthetic Health Index (0=healthy, 1=critical)")
print(f"           Mean health index: {train['health_index'].mean():.3f}")

# Visualise health index
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sample_engines = [1, 5, 10, 15, 20]
for eng in sample_engines:
    eng_data = train[train["engine_id"] == eng]
    axes[0].plot(eng_data["cycle"], eng_data["health_index"],
                 alpha=0.8, linewidth=1.5, label=f"Engine {eng}")
axes[0].set_title("Health Index Over Time — Sample Engines", fontweight="bold")
axes[0].set_xlabel("Cycle")
axes[0].set_ylabel("Health Index (0=healthy, 1=critical)")
axes[0].legend(fontsize=8)
axes[0].axhline(0.7, color="red", linestyle="--", alpha=0.5, label="Alert threshold")

axes[1].scatter(train["health_index"], train["RUL"],
                alpha=0.1, s=5, color="#00B89F")
axes[1].set_title("Health Index vs RUL", fontweight="bold")
axes[1].set_xlabel("Health Index")
axes[1].set_ylabel("Remaining Useful Life")
axes[1].invert_xaxis()

plt.suptitle("Synthetic Health Index — Engine Degradation Score",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "fe_01_health_index.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: fe_01_health_index.png")

# ── Step 5 — RUL Clipping (Piecewise Linear Target) ──────────────────────────
# When RUL is very high (engine is new), precise RUL is less useful
# Industry standard: cap RUL at 125 cycles for training
# Rationale: We only need to warn operators when failure is approaching
RUL_CLIP = 125
train["RUL_clipped"] = train["RUL"].clip(upper=RUL_CLIP)

print(f"\n  Step 5 — RUL clipped at {RUL_CLIP} cycles (piecewise linear target)")
print(f"           This focuses the model on the critical degradation window")

# ── Step 6 — Binary Classification Target ────────────────────────────────────
# In addition to regression (predict exact RUL), create a binary target:
# "Will this engine fail within the next 30 cycles?" — YES=1, NO=0
# This is the most actionable prediction for maintenance scheduling
FAILURE_WINDOW = 30
train["will_fail_soon"] = (train["RUL"] <= FAILURE_WINDOW).astype(int)

churn_rate = train["will_fail_soon"].mean()
print(f"\n  Step 6 — Binary target: failure within {FAILURE_WINDOW} cycles")
print(f"           Failure rate in training data: {churn_rate:.1%}")

# ── Step 7 — Feature Selection ────────────────────────────────────────────────
# Drop original raw sensors (we have rolling features now)
# Keep only engineered features + targets
DROP_COLS  = USEFUL_SENSORS + CONSTANT_SENSORS + ["op_setting_1", "op_setting_2"]
KEEP_TRAIN = [c for c in train.columns if c not in DROP_COLS]
KEEP_TEST  = [c for c in test.columns  if c not in DROP_COLS]

train_final = train[KEEP_TRAIN].copy()
test_final  = test[KEEP_TEST].copy()

print(f"\n  Step 7 — Final feature count: {len(train_final.columns)} columns")
print(f"           Removed raw sensors, kept rolling + engineered features")

# ── Correlation heatmap of top features ───────────────────────────────────────
feature_cols = [c for c in train_final.columns
                if c not in ["engine_id", "cycle", "RUL", "RUL_clipped",
                              "will_fail_soon"]]
top_features = (train_final[feature_cols]
                .corrwith(train_final["RUL"])
                .abs()
                .nlargest(15)
                .index.tolist())

fig, ax = plt.subplots(figsize=(12, 5))
corr_vals = train_final[top_features].corrwith(train_final["RUL"]).sort_values()
colors = ["#e74c3c" if v < 0 else "#00B89F" for v in corr_vals.values]
ax.barh(corr_vals.index, corr_vals.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 15 Features — Correlation with RUL", fontweight="bold")
ax.set_xlabel("Pearson Correlation")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "fe_02_feature_correlation.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: fe_02_feature_correlation.png")

# ── Save ─────────────────────────────────────────────────────────────────────
train_final.to_csv(os.path.join(DATA, "train_engineered.csv"), index=False)
test_final.to_csv(os.path.join(DATA,  "test_engineered.csv"),  index=False)
rul_true.to_csv(os.path.join(DATA,    "rul_true.csv"),          index=False)

print("\n" + "=" * 65)
print("  FEATURE ENGINEERING COMPLETE")
print("=" * 65)
print(f"  Final train shape : {train_final.shape}")
print(f"  Final test shape  : {test_final.shape}")
print(f"  New features created:")
print(f"    Rolling mean/std/min/max : {len(USEFUL_SENSORS) * 4}")
print(f"    Cycle normalised         : 1")
print(f"    Health Index             : 1")
print(f"    RUL clipped target       : 1")
print(f"    Binary failure target    : 1")
print(f"  Saved: train_engineered.csv, test_engineered.csv")
