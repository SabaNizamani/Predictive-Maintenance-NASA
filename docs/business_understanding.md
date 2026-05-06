# Task 1 — Business Understanding & Problem Framing
# Predictive Maintenance for Industrial Turbofan Engines
# Client: Siemens AG / BMW Group (Industrial Engines Division)

---

## The Client & Business Problem

**Client:** Siemens AG — Industrial Drives Division
**Sector:** Manufacturing & Industrial Automation
**Location:** Munich, Germany

### The Problem

Siemens operates thousands of industrial turbofan engines across its
manufacturing plants and energy generation facilities. These engines
run 24/7 and unexpected failures cause:

- 🔴 Unplanned downtime costing €50,000–€200,000 per hour
- 🔴 Safety risks to factory workers
- 🔴 Supply chain disruptions for downstream clients (BMW, Audi)
- 🔴 Emergency maintenance costs 3–5x higher than scheduled maintenance

Current approach — **Scheduled Maintenance** (fixed intervals):
- Replace parts every N cycles regardless of actual engine condition
- Results in replacing healthy parts (waste) OR missing degraded ones (risk)

**Our Solution — Predictive Maintenance:**
> Use real-time sensor data to predict exactly when each engine
> will fail — so maintenance is done at the right time, every time.

---

## The Business Question

> "Can we predict the Remaining Useful Life (RUL) of each engine
>  accurately enough to replace scheduled maintenance with
>  condition-based maintenance?"

**Remaining Useful Life (RUL)** = number of operational cycles
an engine can still run before it requires maintenance/replacement.

---

## Dataset: NASA C-MAPSS

**Source:** NASA Ames Prognostics Center of Excellence (PCoE)
**Full name:** Commercial Modular Aero-Propulsion System Simulation
**Download:** https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data

This is the industry-standard benchmark dataset for predictive
maintenance research. Used in real research by:
- Siemens Research
- Rolls-Royce
- GE Aviation
- BMW Group Research

### Dataset Structure:
- 4 sub-datasets (FD001–FD004) of increasing complexity
- We use FD001 (single fault mode, sea level conditions)
- 100 training engines run to failure
- 100 test engines (predict RUL before failure)
- 26 columns: engine ID, cycle, 3 operational settings, 21 sensors

---

## Our Analytical Approach (5 Steps)

1. **EDA** — Understand sensor behaviour, identify useful signals
2. **Feature Engineering** — Create rolling averages, health indices,
   degradation trend features from raw sensor readings
3. **Modelling** — Two approaches:
   - Regression: predict exact RUL (number of cycles)
   - Classification: predict failure within next 30 cycles (binary)
4. **Evaluation** — RMSE for regression, F1/Recall for classification
5. **Business Insight** — Translate predictions into maintenance schedule

---

## Expected Business Impact

| Metric | Current (Scheduled) | After Predictive ML |
|--------|--------------------|--------------------|
| Unplanned failures | ~15% of engines/year | < 3% |
| Maintenance cost | High (wasteful) | Reduced 25–40% |
| Engine downtime | Unpredictable | Planned in advance |
| Safety incidents | Reactive | Proactively prevented |

---

## Key Terminology for Interview

- **RUL** = Remaining Useful Life (cycles left before failure)
- **Degradation** = gradual worsening of engine performance over time
- **Health Index** = synthetic score (0–1) measuring engine condition
- **Rolling average** = smoothed sensor reading over last N cycles
- **Condition-based maintenance** = maintain only when data says so
