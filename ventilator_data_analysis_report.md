# Ventilator Sensor Dataset: Exploratory Data Analysis & Feasibility Report

**Dataset File:** `synthetic_ventilator_equipment_dataset.csv`  
**Execution Date:** 2026-11-24  
**Status:** Completed (Inspection Only — No Model Training Conducted)  
**Classification:** Medical Device Sensor Telemetry & Equipment Monitoring  

---

## 1. Executive Summary

This report delivers a rigorous exploratory data analysis (EDA) of the ventilator sensor telemetry dataset (`synthetic_ventilator_equipment_dataset.csv`). The dataset comprises **20,000 records** and **18 features**, capturing longitudinal operational telemetry from **100 distinct ventilator units** across **3 equipment models**.

### Key Findings at a Glance
- **Data Completeness:** **0 missing values** (100% complete across all 18 attributes).
- **Data Deduplication:** **0 duplicate records** detected.
- **Temporal Nature:** **Definitively Panel Time-Series Data** (longitudinal multivariate sensor telemetry). It is **not** independent cross-sectional data. Each device exhibits an equispaced 5-minute sampling rate across exactly 200 consecutive readings (16 hours, 35 minutes per device trajectory).
- **Temporal Dynamics:** Failures occur in deterministic episodic windows of **20 consecutive steps** (100 minutes) at lifecycle steps 30–49, 95–114, and 155–174 across devices, bracketed by healthy baseline regimes.
- **Target Attributes Identified:**
  1. **Binary Target (`failure`):** `0` (Normal, 70.0%) vs. `1` (Failure, 30.0%).
  2. **Multiclass Target (`fault_type`):** 6 mutually exclusive ground-truth states (`normal`: 70.0%, `sensor_drift`: 6.7%, `overheating`: 6.1%, `airflow_restriction`: 6.0%, `blower_fault`: 5.6%, `power_supply_fault`: 5.6%).
- **Modeling Feasibility:** Highly viable for both tabular gradient boosting (via rolling temporal window feature extraction) and sequence-based deep learning (TCN, BiLSTM-Attention, Autoencoders).
- **Critical Validation Rule:** Standard random train-test splitting **must not be used** due to severe temporal leakage. All cross-validation must use device-grouped or blocked temporal splitting (`GroupKFold` on `device_id`).

---

## 2. Dataset Architecture & Inventory

### 2.1 File & Memory Dimensions
- **Row Count:** 20,000 observations
- **Column Count:** 18 columns
- **File Size on Disk:** ~2.79 MB
- **Unique Devices:** 100 unique units (`VENT-0001` through `VENT-0100`), exactly 200 observations per unit.

### 2.2 Column Schema and Data Types

| Index | Column Name | Raw Data Type | Semantic Type | Physical / Domain Unit | Description |
|:---:|:---|:---:|:---:|:---:|:---|
| 0 | `timestamp` | `object` (`string`) | Temporal Index | ISO 8601 Timestamp | Telemetry recording timestamp (`YYYY-MM-DD HH:MM:SS`) |
| 1 | `device_id` | `object` (`string`) | Entity Identifier | Alphanumeric ID | Unique ventilator unit identifier (`VENT-0001` to `VENT-0100`) |
| 2 | `device_model` | `object` (`string`) | Categorical Metadata | Model Code | Hardware configuration (`VENT-X100`, `VENT-X200`, `VENT-M500`) |
| 3 | `operating_hours` | `float64` | Continuous Metric | Hours (h) | Cumulative equipment running hours |
| 4 | `inlet_pressure_kPa` | `float64` | Continuous Sensor | Kilopascals (kPa) | Supply gas intake pressure |
| 5 | `airway_pressure_cmH2O` | `float64` | Continuous Sensor | cmH₂O | Patient circuit airway pressure |
| 6 | `airflow_L_min` | `float64` | Continuous Sensor | Liters / min (L/min) | Delivered air/gas flow rate |
| 7 | `oxygen_concentration_pct`| `float64` | Continuous Sensor | Percentage (%) | FiO₂ measured oxygen delivery percentage |
| 8 | `internal_temperature_C` | `float64` | Continuous Sensor | Celsius (°C) | Internal equipment compartment temperature |
| 9 | `supply_voltage_V` | `float64` | Continuous Sensor | Volts (V) | Mainline electrical supply voltage |
| 10 | `motor_current_A` | `float64` | Continuous Sensor | Amperes (A) | Blower motor electrical current draw |
| 11 | `blower_speed_rpm` | `float64` | Continuous Sensor | RPM | Centrifugal blower rotational speed |
| 12 | `battery_voltage_V` | `float64` | Continuous Sensor | Volts (V) | Internal backup battery DC voltage |
| 13 | `vibration_mm_s` | `float64` | Continuous Sensor | mm/s | Mechanical chassis vibration velocity (RMS) |
| 14 | `filter_pressure_drop_kPa`| `float64` | Continuous Sensor | Kilopascals (kPa) | Differential pressure drop across particulate filter |
| 15 | `ambient_humidity_pct` | `float64` | Continuous Sensor | Percentage (%) | Ambient operating environment relative humidity |
| 16 | `failure` | `int64` | Binary Target | Indicator {0, 1} | Binary health status (0 = Operational, 1 = Fault Active) |
| 17 | `fault_type` | `object` (`string`) | Multi-Class Target | Categorical | Specific empirical condition/fault label |

---

## 3. Data Integrity & Quality Audit

### 3.1 Missing Value Analysis
- **Total Missing Cells:** 0 across all 360,000 cells (100% integrity).
- No imputation or null-handling routines are necessary.

### 3.2 Duplicate Record Analysis
- **Exact Duplicate Rows:** 0 rows (0.00%).
- **Primary Composite Key Candidate:** `(device_id, timestamp)` uniquely identifies all 20,000 rows without collisions.

### 3.3 Sensor Summary Statistics & Distribution Screening

| Feature | Min | 25% | Median | Mean | 75% | Max | Std Dev | IQR Outliers (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `operating_hours` | 500.00 | 825.88 | 1151.79 | 1151.79 | 1476.77 | 1803.58 | 375.83 | 0.00% |
| `inlet_pressure_kPa` | 4.030 | 5.283 | 5.535 | 5.578 | 5.805 | 8.112 | 0.463 | 3.99% |
| `airway_pressure_cmH2O` | 10.443 | 16.891 | 18.182 | 18.455 | 19.590 | 31.106 | 2.475 | 5.05% |
| `airflow_L_min` | 6.582 | 37.788 | 41.128 | 40.223 | 44.111 | 58.046 | 6.148 | 5.96% |
| `oxygen_concentration_pct`| 27.917 | 33.683 | 35.022 | 35.023 | 36.365 | 43.654 | 1.987 | 0.69% |
| `internal_temperature_C` | 33.223 | 37.525 | 38.389 | 39.028 | 39.302 | 57.298 | 3.158 | 6.50% |
| `supply_voltage_V` | 186.707 | 227.552 | 229.781 | 228.672 | 231.903 | 242.556 | 6.420 | 6.02% |
| `motor_current_A` | 1.178 | 1.718 | 1.838 | 1.866 | 1.980 | 2.883 | 0.221 | 3.00% |
| `blower_speed_rpm` | 1702.78 | 3093.97 | 3189.30 | 3152.43 | 3280.61 | 3849.13 | 235.57 | 5.96% |
| `battery_voltage_V` | 18.537 | 23.628 | 23.965 | 23.834 | 24.288 | 26.106 | 0.844 | 5.95% |
| `vibration_mm_s` | 0.062 | 0.692 | 0.824 | 0.895 | 0.971 | 3.036 | 0.378 | 6.55% |
| `filter_pressure_drop_kPa`| 0.283 | 0.625 | 0.711 | 0.803 | 0.802 | 3.510 | 0.426 | 6.35% |
| `ambient_humidity_pct` | 25.584 | 41.651 | 44.996 | 45.001 | 48.355 | 67.534 | 4.999 | 0.73% |

*Note on Outliers:* Statistical outliers identified by the $1.5 \times \text{IQR}$ rule correspond directly to the episodic fault manifestations (e.g. elevated vibration during blower faults, high temperatures during overheating, filter pressure spikes during airflow restriction) rather than sensor data corruption.

---

## 4. Time-Series vs. Independent Sensor Data Determination

A central question is whether this dataset consists of **independent (cross-sectional) observations** or **time-series (temporal) data**.

### 4.1 Definitive Conclusion
The dataset is **unequivocally Panel Multivariate Time-Series Data** (structured multi-device longitudinal telemetry). Treating these rows as independent, identically distributed (I.I.D.) records is invalid.

### 4.2 Empirical & Mathematical Evidence

1. **Explicit Monotonic Timestamps:**
   - Every individual device trajectory contains 200 time steps.
   - For all 100 devices ($100\%$), `timestamp` is **strictly monotonically increasing**.
   - Sampling period $\Delta t$ is strictly constant at **$5\text{ minutes}$ ($300\text{ seconds}$)** for $19,900$ consecutive within-device pairs.

2. **Continuous Physical Progress Counters:**
   - `operating_hours` advances monotonically per device at exactly $\Delta \approx 0.0833\text{ hours}$ ($5\text{ minutes} / 60\text{ minutes} = 0.0833\text{ h}$) per step.

3. **Substantial Serial Autocorrelation:**
   - Across the 100 devices, continuous physical variables show pronounced lag-1 and lag-5 autocorrelation:
     - `internal_temperature_C`: Lag-1 Autocorr = **$+0.518$**, Lag-5 Autocorr = **$+0.405$**
     - `filter_pressure_drop_kPa`: Lag-1 Autocorr = **$+0.520$**, Lag-5 Autocorr = **$+0.392$**
     - `vibration_mm_s`: Lag-1 Autocorr = **$+0.518$**, Lag-5 Autocorr = **$+0.379$**
     - `airflow_L_min`: Lag-1 Autocorr = **$+0.441$**, Lag-5 Autocorr = **$+0.331$**
     - `supply_voltage_V`: Lag-1 Autocorr = **$+0.426$**, Lag-5 Autocorr = **$+0.337$**
     - `battery_voltage_V`: Lag-1 Autocorr = **$+0.413$**, Lag-5 Autocorr = **$+0.306$**
     - `blower_speed_rpm`: Lag-1 Autocorr = **$+0.408$**, Lag-5 Autocorr = **$+0.309$**
     - `motor_current_A`: Lag-1 Autocorr = **$+0.393$**, Lag-5 Autocorr = **$+0.296$**
   - Environmental control variables (`oxygen_concentration_pct`, `ambient_humidity_pct`) fluctuate around their setpoints with near-zero autocorrelation, characteristic of regulated systems subject to zero-mean white noise.

4. **Episodic Fault Windows (Temporal State Persistence):**
   - Failures are not independent point anomalies. In every unit, faults occur in distinct **20-step episodes** ($20 \times 5\text{ min} = 100\text{ minutes}$ sustained duration) at deterministic lifecycle stages:
     - **Baseline Normal Operation:** Steps 0 to 29 (150 min)
     - **Episode 1 (Active Fault):** Steps 30 to 49 (100 min)
     - **Recovery/Intermission (Normal):** Steps 50 to 94 (225 min)
     - **Episode 2 (Active Fault):** Steps 95 to 114 (100 min)
     - **Recovery/Intermission (Normal):** Steps 115 to 154 (200 min)
     - **Episode 3 (Active Fault):** Steps 155 to 174 (100 min)
     - **Terminal Baseline (Normal):** Steps 175 to 199 (125 min)
   - Across all 100 devices, this generates exactly $100 \times 3 = 300$ discrete fault episodes totaling $6,000$ failure time steps.

---

## 5. Target Labels & Class Distribution Analysis

Without pre-assuming any fault taxonomy, empirical inspection reveals two primary supervisory labels:

### 5.1 Binary Target: `failure`

| Label Code | Semantic Meaning | Record Count | Percentage | Class Balance Ratio |
|:---:|:---|:---:|:---:|:---:|
| `0` | Normal / Healthy Operation | 14,000 | 70.00% | 2.33 : 1 |
| `1` | Equipment Fault State | 6,000 | 30.00% | 1 : 2.33 |
| **Total** | | **20,000** | **100.00%** | |

*Assessment:* Moderately balanced binary classification problem. Standard cross-entropy loss with mild class weighting or threshold tuning is sufficient (no extreme imbalanced sampling such as SMOTE required).

### 5.2 Multiclass Target: `fault_type`

| Class Label | Category Interpretation | Episode Count | Total Records | Percentage |
|:---|:---|:---:|:---:|:---:|
| `normal` | Fault-free baseline telemetry | N/A | 14,000 | 70.00% |
| `sensor_drift` | Measurement calibration shift | 67 episodes | 1,340 | 6.70% |
| `overheating` | Thermal regulation failure | 61 episodes | 1,220 | 6.10% |
| `airflow_restriction`| Pneumatic path / filter clogging | 60 episodes | 1,200 | 6.00% |
| `blower_fault` | Blower motor mechanical defect | 56 episodes | 1,120 | 5.60% |
| `power_supply_fault` | Grid / battery electrical failure | 56 episodes | 1,120 | 5.60% |
| **Total** | | **300 episodes** | **20,000** | **100.00%** |

*Assessment:* The 5 active fault types are distributed almost uniformly across the failure pool (~18.7% to 22.3% of the failure cases each, or ~5.6% to 6.7% of total dataset rows).

### 5.3 Distribution Across Equipment Models

The dataset spans 3 ventilator models distributed almost evenly across 100 physical devices:
- **`VENT-X100`:** 34 devices (6,800 records, 34.0%)
- **`VENT-X200`:** 33 devices (6,600 records, 33.0%)
- **`VENT-M500`:** 33 devices (6,600 records, 33.0%)

Each equipment model experiences all 5 fault types in roughly equal proportions, confirming balanced coverage across hardware lines:

| Model Code | Normal | Sensor Drift | Overheating | Airflow Restriction | Blower Fault | Power Supply Fault | Total |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `VENT-X100` | 4,760 | 520 | 400 | 420 | 320 | 380 | 6,800 |
| `VENT-X200` | 4,620 | 420 | 420 | 440 | 360 | 340 | 6,600 |
| `VENT-M500` | 4,620 | 400 | 400 | 340 | 440 | 400 | 6,600 |

### 5.4 Empirical Sensor Signatures per Fault Condition

Comparing the sensor feature means during each specific fault state against the `normal` baseline highlights distinctive, physically coherent physical signatures:

1. **`blower_fault`:**
   - Vibration accelerates sharply: $+1.39\text{ mm/s}$ ($Z\text{-shift} = +3.69\sigma$).
   - Blower RPM collapses: $-844.4\text{ RPM}$ ($Z\text{-shift} = -3.58\sigma$).
   - Airflow drops: $-10.0\text{ L/min}$ ($Z\text{-shift} = -1.63\sigma$).
   - Motor current increases: $+0.30\text{ A}$ ($Z\text{-shift} = +1.38\sigma$).

2. **`power_supply_fault`:**
   - Supply voltage drops: $-24.06\text{ V}$ ($Z\text{-shift} = -3.75\sigma$).
   - Battery voltage drops: $-3.02\text{ V}$ ($Z\text{-shift} = -3.58\sigma$).
   - Motor current increases to compensate: $+0.49\text{ A}$ ($Z\text{-shift} = +2.24\sigma$).

3. **`airflow_restriction`:**
   - Filter pressure drop surges: $+1.69\text{ kPa}$ ($Z\text{-shift} = +3.97\sigma$).
   - Airflow drops significantly: $-17.02\text{ L/min}$ ($Z\text{-shift} = -2.77\sigma$).
   - Inlet pressure increases: $+1.19\text{ kPa}$ ($Z\text{-shift} = +2.57\sigma$).

4. **`overheating`:**
   - Internal temperature surges: $+12.07^\circ\text{C}$ to $50.36^\circ\text{C}$ ($Z\text{-shift} = +3.82\sigma$).
   - Motor current increases: $+0.35\text{ A}$ ($Z\text{-shift} = +1.58\sigma$).
   - Vibration exhibits mild elevation: $+0.34\text{ mm/s}$ ($Z\text{-shift} = +0.90\sigma$).

5. **`sensor_drift`:**
   - Airway pressure reading drifts upward: $+6.63\text{ cmH}_2\text{O}$ to $24.65\text{ cmH}_2\text{O}$ ($Z\text{-shift} = +2.68\sigma$).
   - Flow and mechanical sensors remain near nominal baseline, isolating the error to the transducer channel.

---

## 6. Recommended Machine Learning & Deep Learning Approaches

Because this dataset is **panel multivariate time-series data with clear temporal dependencies and device groupings**, the analytical strategy must incorporate temporal awareness.

### 6.1 Critical Prerequisite: Validation Strategy
> [!CAUTION]
> **Never use standard random $k$-fold cross-validation or random `train_test_split`.**
> A random split would place adjacent 5-minute points from the same 20-step fault episode into both training and validation sets, resulting in massive **data leakage** and unrealistically optimistic metric scores.
>
> **Recommended Validation Protocol:**
> 1. **Device-Grouped Cross-Validation (`GroupKFold` on `device_id`):** Train on 80 devices, evaluate on 20 unseen devices. This tests genuine generalization to new hardware.
> 2. **Blocked Temporal Validation (Purged Walk-Forward):** Train on earlier time indices across devices and test on later time indices with a safety buffer between train and test boundaries.

---

### 6.2 Recommended Machine Learning (Tabular / Feature-Engineered) Approaches

Tabular gradient boosted trees provide exceptional baselines when combined with time-aware feature engineering.

#### Feature Engineering Pipeline (Rolling Window Transforms)
For each sensor channel $X_i$, compute sliding temporal summary features across windows of size $w \in \{3, 6, 12\}$ steps ($15, 30, 60$ minutes):
- **Rolling Central Tendency & Dispersion:** Rolling mean, rolling median, rolling standard deviation.
- **Dynamic Rates of Change:** First-difference ($\Delta X_t = X_t - X_{t-1}$), rolling acceleration ($\Delta^2 X_t$), and rolling linear regression slope over the window.
- **Extreme Value Deviations:** $\text{Z-score}(t) = \frac{X_t - \mu_{\text{roll}}}{\sigma_{\text{roll}}}$, rolling min/max.
- **Multivariate Ratios:** Power/Current ratio ($\frac{V}{I}$), Pressure-to-Flow ratio ($\frac{P_{\text{airway}}}{Q_{\text{airflow}}}$).

#### Machine Learning Algorithms
1. **LightGBM / XGBoost / CatBoost Classifier:**
   - **Task:** Multiclass classification (`fault_type`) and binary detection (`failure`).
   - **Why:** State-of-the-art tabular accuracy, native handling of categorical features (`device_model`), robustness to varying feature scales, and fast training.
   - **Output:** Calibrated prediction probabilities per fault type.
2. **Balanced Random Forest / ExtraTrees:**
   - **Why:** Serves as a strong non-linear bagging baseline, less prone to overfitting on synthetic boundary artifacts.
3. **Unsupervised Baselines (Zero-Label Anomaly Detection):**
   - **Algorithms:** Isolation Forest, Minimum Covariance Determinant (Elliptic Envelope), and Local Outlier Factor (LOF).
   - **Methodology:** Fit purely on the initial normal operating window (steps 0–29) for each machine, and evaluate outlier scores on subsequent steps.

---

### 6.3 Recommended Deep Learning Approaches

Deep learning models operate natively on raw multivariate time series tensors without manual feature engineering, learning temporal dependencies directly from sequence windows $\mathbf{X} \in \mathbb{R}^{B \times T \times D}$, where $B$ is batch size, $T \in [10, 30]$ time steps ($50\text{ to }150\text{ minutes}$), and $D=12$ sensor channels.

#### 1. Temporal Convolutional Networks (1D-CNN / TCN)
- **Architecture:** Stacks of dilated causal 1D convolutions with residual connections and weight normalization.
- **Why it fits:** TCNs capture long-range temporal contexts with an exponential receptive field, avoid vanishing/exploding gradients typical of RNNs, and train in parallel over sequences.
- **Application:** Real-time stream classification where each receptive field outputs the health status of the current 5-minute slice.

#### 2. Recurrent Architectures: BiLSTM / GRU with Self-Attention
- **Architecture:** 2-layer Bidirectional LSTM / GRU followed by a Multi-Head Attention pooling layer.
- **Why it fits:** Recurrent gates effectively capture transition dynamics between normal and fault states; attention weights provide explainability by highlighting which sensor readings and time steps triggered the classification.

#### 3. Deep Unsupervised Reconstruction (LSTM-Autoencoder / Temporal VAE)
- **Architecture:** Encoder-Decoder sequence autoencoder trained exclusively on healthy telemetry ($y = 0$).
- **Mechanism:** The model learns the manifold of normal ventilator operation. When a fault occurs (e.g. blower cavitation, overheating, sensor drift), the autoencoder fails to reconstruct the anomalous sensor correlations, causing the Reconstruction Error ($\text{MSE}$) to spike.
- **Why it fits:** Reflects real-world clinical deployments where rare, novel, or unseen fault types must be flagged without requiring prior labeled fault instances.

#### 4. Modern Time-Series Transformers (PatchTST / TFT)
- **Architecture:** Patch Time Series Transformer (PatchTST) or Temporal Fusion Transformer (TFT).
- **Why it fits:** TFT natively incorporates static metadata (`device_model`, baseline `operating_hours`) alongside time-varying sensor inputs, providing attention maps across temporal horizons.

---

### 6.4 Evaluation Metrics Selection (Medical Domain Context)

Given the medical equipment nature of ventilators:
- **Primary Metric for Binary Failure:** **Precision-Recall AUC (PR-AUC)** and **Recall at 99% Precision** (false negatives in medical life-support equipment carry extreme risk).
- **Primary Metric for Multi-Class Diagnosis:** **Macro-averaged F1-Score** and **Cohen’s Kappa** to ensure equal penalty across all 5 fault classes regardless of slight class size differences.
- **Confusion Matrix Analysis:** Critical for verifying that serious mechanical or electrical failures (`blower_fault`, `power_supply_fault`) are never misclassified as mild sensor drift.

---

## 7. Synthesis & Next Steps Roadmap

```
                    ┌────────────────────────────────────────────────────────┐
                    │  Current Phase: Exploratory Data Analysis (Completed)   │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 1: Preprocessing & Leakage-Free Splitting                                                        │
│ • Establish 5-fold GroupKFold stratified by device_model on device_id                                 │
│ • Fit RobustScaler on healthy training windows only                                                    │
└───────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    │                                                       │
                    ▼                                                       ▼
┌───────────────────────────────────────┐       ┌────────────────────────────────────────────────────────┐
│ Stream A: Supervised Tabular Path     │       │ Stream B: Deep Sequential Path                         │
│ • Extract rolling window features     │       │ • Slice sequences into sliding tensors (e.g. T=20)     │
│ • Train LightGBM / CatBoost baseline  │       │ • Train 1D-TCN and BiLSTM-Attention models             │
│ • Multiclass SHAP feature importance  │       │ • Train LSTM-Autoencoder for zero-shot fault detection │
└───────────────────┬───────────────────┘       └───────────────────────────┬────────────────────────────┘
                    │                                                       │
                    └───────────────────────────┬───────────────────────────┘
                                                │
                                                ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Phase 2: Evaluation, Ensembling & Diagnostic Calibration                                               │
│ • Compare PR-AUC, Macro-F1, and Inference Latency                                                      │
│ • Calibration curves and threshold optimization for critical alert triggering                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
