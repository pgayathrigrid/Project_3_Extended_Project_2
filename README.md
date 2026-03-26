# ⚡ Electricity Load Anomaly Detection Dashboard (Extended)

This project presents an extended electricity load anomaly detection system using a **Transformer Autoencoder**.  
It includes an interactive **Streamlit dashboard** for visualizing anomaly scores, saliency maps, seasonal anomaly analysis, and bottleneck ablation results.

---

## 📊 Project Overview

Electricity consumption data can contain unusual patterns caused by equipment faults, abnormal usage, sensor issues, or sudden behavioural changes. Detecting these anomalies manually is difficult because electricity data is large-scale, multivariate, and time-dependent.

This project uses a **Transformer Autoencoder** to learn normal electricity consumption behaviour and identify anomalies using **reconstruction error**.

The extended version of the project also includes:

- **Task 2:** Reconstruction error saliency analysis
- **Task 3:** Bottleneck dimensionality ablation
- **Task 4:** Seasonal decomposition and anomaly type analysis
- **Task 5:** Adversarial robustness discussion

All major outputs are visualized through a **Streamlit dashboard**.

---

## 🗂 Dataset

Dataset: **Electricity Load Diagrams (Portugal)**

- Original shape: `140256 × 370`
- Resampled to hourly data
- Selected **top 30 active smart meters**
- Window size: **168 hours (1 week)**

Final input shape: `(N, 168, 30)`

### Important Notice

Dataset: **Electricity Load Diagrams 2011–2014**

Due to GitHub file size limits, the full dataset is **not included in this repository**.

Direct download:  
https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip

Dataset source:  
https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014

---

## 🤖 Core Model

Main model used:

- **Transformer Autoencoder**

The model detects anomalies based on **high reconstruction error**.

### Best Performance

- **Best PR-AUC:** ~0.48
- **Target achieved:** PR-AUC ≥ 0.30 ✅

---

## 🔬 Extended Analysis Included

### Task 2 — Saliency Analysis
- Saliency map across time steps and meters
- Meter-wise saliency
- Hour-of-day saliency
- Time-step saliency within anomaly window

### Task 3 — Bottleneck Ablation
- Compared bottleneck sizes: `4, 8, 16, 32, 64`
- Best bottleneck: `64`
- Best trade-off: `16–32`

### Task 4 — Seasonal Decomposition
- Compared anomaly detection across:
  - seasonal component
  - full signal
  - residual component
- Seasonal component showed strongest anomaly detection performance

### Task 5 — Robustness Analysis
- Studied sensitivity to perturbations such as:
  - noise injection
  - sensor drift
  - calibration errors
  - adversarial conditions

---

## 📈 Dashboard Features

The extended Streamlit dashboard provides:

- Transformer AE anomaly score timeline
- Configurable threshold methods
  - p95 / p99
  - 2σ / 3σ
  - custom percentile
- Optional IsolationForest score overlay
- Top anomaly windows table
- CSV download of anomaly windows
- Saliency map visualization
- Meter importance visualization
- Hour-of-day saliency
- Time-step saliency
- Component PR-AUC comparison
- Anomaly type analysis
- Meter-level anomaly interpretation

---

## Run the Dashboard Locally

To run the updated dashboard locally:

```bash
streamlit run dashboard_1.py

## Live Dashboard

👉 https://electricity-anomaly-dashboard.streamlit.app
