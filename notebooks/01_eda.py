# =============================================================================
# Predictive Maintenance — Turbofan Engine RUL Prediction
# Client: Siemens AG / BMW Group Industrial Engines Division
# Dataset: NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)
# =============================================================================
# Task 2: Exploratory Data Analysis
# Goal: Understand sensor behaviour, identify degradation patterns,
#       and determine which sensors carry useful predictive signal.
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titlesize": 13})

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(BASE)
DATA    = os.path.join(ROOT, "data")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

# ── Column Names ──────────────────────────────────────────────────────────────
# NASA C-MAPSS has no headers — we define them based on dataset documentation
COLS = (
    ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i}" for i in range(1, 22)]
)

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("=" * 65)
print("  LOADING NASA C-MAPSS DATASET — FD001")
print("=" * 65)

train = pd.read_csv(
    os.path.join(DATA, "train_FD001.txt"),
    sep=r"\s+", header=None, names=COLS
)
test = pd.read_csv(
    os.path.join(DATA, "test_FD001.txt"),
    sep=r"\s+", header=None, names=COLS
)
rul_true = pd.read_csv(
    os.path.join(DATA, "RUL_FD001.txt"),
    header=None, names=["RUL"]
)

print(f"  Train set  : {train.shape[0]:,} rows  |  {train['engine_id'].nunique()} engines")
print(f"  Test set   : {test.shape[0]:,} rows   |  {test['engine_id'].nunique()} engines")
print(f"  Sensors    : 21 sensor measurements per cycle")
print(f"  Settings   : 3 operational settings")

# ── 2. Add RUL to Training Data ───────────────────────────────────────────────
# RUL = max cycle for that engine - current cycle
# (engines run to failure in training set)
max_cycles = train.groupby("engine_id")["cycle"].max().reset_index()
max_cycles.columns = ["engine_id", "max_cycle"]
train = train.merge(max_cycles, on="engine_id")
train["RUL"] = train["max_cycle"] - train["cycle"]
train.drop(columns=["max_cycle"], inplace=True)

print(f"\n  RUL Statistics:")
print(f"  Min RUL  : {train['RUL'].min()}")
print(f"  Max RUL  : {train['RUL'].max()}")
print(f"  Mean RUL : {train['RUL'].mean():.1f} cycles")

# ── 3. RUL Distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(train["RUL"], bins=50, color="#00B89F", edgecolor="white", linewidth=0.5)
axes[0].set_title("RUL Distribution — Training Set", fontweight="bold")
axes[0].set_xlabel("Remaining Useful Life (cycles)")
axes[0].set_ylabel("Frequency")
axes[0].axvline(train["RUL"].mean(), color="#e74c3c", linestyle="--",
                label=f"Mean = {train['RUL'].mean():.0f}")
axes[0].legend()

# Engine lifecycle — show a few engines degrading over time
sample_engines = train["engine_id"].unique()[:8]
for eng in sample_engines:
    eng_data = train[train["engine_id"] == eng]
    axes[1].plot(eng_data["cycle"], eng_data["RUL"], alpha=0.7, linewidth=1)
axes[1].set_title("RUL Over Time — Sample Engines", fontweight="bold")
axes[1].set_xlabel("Operational Cycle")
axes[1].set_ylabel("Remaining Useful Life")
axes[1].invert_yaxis()

plt.suptitle("Target Variable — Remaining Useful Life (RUL)", 
             fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_01_rul_distribution.png"), bbox_inches="tight")
plt.show()
print("\n  ✔ Saved: eda_01_rul_distribution.png")

# ── 4. Sensor Analysis — Which Sensors Are Useful? ────────────────────────────
# Some sensors show no variation (constant readings) — useless for prediction
sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
sensor_std = train[sensor_cols].std()

print("\n  Sensor Standard Deviations (zero = constant = useless):")
print(sensor_std.sort_values().to_string())

# Identify constant sensors (std < 0.01)
constant_sensors = sensor_std[sensor_std < 0.01].index.tolist()
useful_sensors   = sensor_std[sensor_std >= 0.01].index.tolist()
print(f"\n  Constant sensors (removed): {constant_sensors}")
print(f"  Useful sensors  (kept)    : {useful_sensors}")

# ── 5. Sensor Degradation Over Time ───────────────────────────────────────────
# Show how sensor readings change as engine degrades
# We normalise by engine lifecycle % to compare engines of different lengths

fig, axes = plt.subplots(3, 3, figsize=(18, 14))
axes = axes.flatten()

# Use top 9 most variable sensors
top_sensors = sensor_std.nlargest(9).index.tolist()

for i, sensor in enumerate(top_sensors):
    # Average across all engines at each lifecycle % point
    train["life_pct"] = train["cycle"] / train.groupby("engine_id")["cycle"].transform("max")
    bins = pd.cut(train["life_pct"], bins=20, labels=False)
    avg = train.groupby(bins)[sensor].mean()

    axes[i].plot(avg.index, avg.values, color="#00B89F", linewidth=2)
    axes[i].fill_between(avg.index, avg.values, alpha=0.15, color="#00B89F")
    axes[i].set_title(sensor.replace("_", " ").title(), fontweight="bold")
    axes[i].set_xlabel("Engine Lifecycle %")
    axes[i].set_ylabel("Mean Reading")

plt.suptitle("Sensor Degradation Patterns — Averaged Across All Engines\n"
             "(x-axis = 0% new engine → 100% about to fail)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_02_sensor_degradation.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_02_sensor_degradation.png")

# ── 6. Single Engine Deep Dive ────────────────────────────────────────────────
# Track one engine across its full lifecycle
engine_1 = train[train["engine_id"] == 1].copy()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

key_sensors = ["sensor_2", "sensor_3", "sensor_4",
               "sensor_7", "sensor_11", "sensor_15"]

for i, sensor in enumerate(key_sensors):
    axes[i].scatter(engine_1["cycle"], engine_1[sensor],
                    c=engine_1["RUL"], cmap="RdYlGn",
                    s=10, alpha=0.8)
    axes[i].set_title(sensor.replace("_", " ").title(), fontweight="bold")
    axes[i].set_xlabel("Cycle")
    axes[i].set_ylabel("Sensor Reading")

plt.suptitle("Engine #1 — Sensor Readings Coloured by RUL\n"
             "(Green = healthy, Red = near failure)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_03_engine1_deep_dive.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_03_engine1_deep_dive.png")

# ── 7. Operational Settings Analysis ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, setting in enumerate(["op_setting_1", "op_setting_2", "op_setting_3"]):
    axes[i].hist(train[setting], bins=30, color="#1C1C1C", edgecolor="white")
    axes[i].set_title(setting.replace("_", " ").title(), fontweight="bold")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")

plt.suptitle("Operational Settings Distribution\n"
             "(FD001 = single operating condition — settings are near-constant)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_04_operational_settings.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_04_operational_settings.png")

# ── 8. Correlation Heatmap — Sensors vs RUL ──────────────────────────────────
corr_with_rul = train[sensor_cols + ["RUL"]].corr()["RUL"].drop("RUL").sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ["#e74c3c" if x < 0 else "#00B89F" for x in corr_with_rul.values]
ax.barh(corr_with_rul.index, corr_with_rul.values, color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Sensor Correlation with RUL\n"
             "(Positive = increases as engine ages, Negative = decreases)",
             fontweight="bold", fontsize=13)
ax.set_xlabel("Pearson Correlation with RUL")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS, "eda_05_sensor_rul_correlation.png"), bbox_inches="tight")
plt.show()
print("  ✔ Saved: eda_05_sensor_rul_correlation.png")

# ── 9. Key EDA Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  EDA SUMMARY — KEY FINDINGS")
print("=" * 65)
print(f"  1. Training set has {train['engine_id'].nunique()} engines, each run to failure")
print(f"  2. RUL ranges from 0 to {train['RUL'].max()} cycles (mean: {train['RUL'].mean():.0f})")
print(f"  3. {len(constant_sensors)} constant sensors identified — will be removed")
print(f"  4. {len(useful_sensors)} useful sensors carry real signal")
print(f"  5. Sensors 2, 3, 4, 7, 11, 15 show strongest degradation trends")
print(f"  6. Operational settings near-constant (FD001 = single condition)")
print(f"  7. Sensor readings are NOISY — rolling averages needed")
print(f"     → Feature engineering will smooth signals and add trend features")

# Save processed data for next step
train.to_csv(os.path.join(DATA, "train_with_rul.csv"), index=False)
print(f"\n  ✔ Saved: train_with_rul.csv")
print("  ✔ EDA complete — proceeding to Feature Engineering")
