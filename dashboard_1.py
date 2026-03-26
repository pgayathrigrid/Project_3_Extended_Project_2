import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Electricity Load Anomaly Detection Dashboard", layout="wide")

st.title("⚡ Electricity Load Anomaly Detection Dashboard (Extended)")
st.caption("Transformer AE anomaly timeline + saliency analysis + anomaly type analysis.")

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent
ART_DIR = BASE_DIR / "artifacts"

# ---------- Helper ----------
def require_file(path: Path, label: str):
    if not path.exists():
        st.error(f"Missing required file: `{label}` at `{path}`")
        st.stop()

# ---------- Required core artifacts ----------
require_file(ART_DIR / "val_times.pkl", "val_times.pkl")
require_file(ART_DIR / "val_err_transformer.npy", "val_err_transformer.npy")

times = pd.read_pickle(ART_DIR / "val_times.pkl")
times = pd.Series(pd.to_datetime(times))
scores_tf = np.load(ART_DIR / "val_err_transformer.npy")

# ---------- Optional older artifacts ----------
scores_if = None
has_if = (ART_DIR / "val_err_isoforest.npy").exists()
if has_if:
    try:
        scores_if = np.load(ART_DIR / "val_err_isoforest.npy")
    except Exception:
        scores_if = None
        has_if = False

meters = None
if (ART_DIR / "meters_k30.npy").exists():
    try:
        meters = np.load(ART_DIR / "meters_k30.npy", allow_pickle=True).tolist()
    except Exception:
        meters = None

# ---------- New extended artifacts ----------
saliency_map = None
meter_importance = None
hour_of_day_saliency = None
time_importance = None
top_meter_idx = None
component_results_df = None
meter_type_df = None
anomaly_type_summary = None

if (ART_DIR / "saliency_map.npy").exists():
    saliency_map = np.load(ART_DIR / "saliency_map.npy")

if (ART_DIR / "meter_importance.npy").exists():
    meter_importance = np.load(ART_DIR / "meter_importance.npy")

if (ART_DIR / "hour_of_day_saliency.npy").exists():
    hour_of_day_saliency = np.load(ART_DIR / "hour_of_day_saliency.npy")

if (ART_DIR / "time_importance.npy").exists():
    time_importance = np.load(ART_DIR / "time_importance.npy")

if (ART_DIR / "top_meter_idx.npy").exists():
    top_meter_idx = np.load(ART_DIR / "top_meter_idx.npy")

if (ART_DIR / "component_results.csv").exists():
    component_results_df = pd.read_csv(ART_DIR / "component_results.csv")

if (ART_DIR / "meter_type_df.csv").exists():
    meter_type_df = pd.read_csv(ART_DIR / "meter_type_df.csv")

if (ART_DIR / "anomaly_type_summary.csv").exists():
    anomaly_type_summary = pd.read_csv(ART_DIR / "anomaly_type_summary.csv")

# ---------- Sanity checks ----------
if len(times) != len(scores_tf):
    st.error(f"Length mismatch: times={len(times)} vs scores_tf={len(scores_tf)}")
    st.stop()

if has_if and (scores_if is not None) and (len(scores_if) != len(scores_tf)):
    st.error(f"Length mismatch: scores_if={len(scores_if)} vs scores_tf={len(scores_tf)}")
    st.stop()

# ---------- Sidebar ----------
st.sidebar.header("Controls")

threshold_mode = st.sidebar.selectbox(
    "Threshold method",
    ["p99", "p95", "2-sigma", "3-sigma", "Custom percentile"]
)

custom_p = None
if threshold_mode == "Custom percentile":
    custom_p = st.sidebar.slider("Percentile", 80, 99, 97)

show_if = st.sidebar.checkbox("Overlay IsolationForest score", value=has_if) if has_if else False
top_n = st.sidebar.slider("Top anomalies to show", 10, 200, 50)

# ---------- Threshold ----------
mu = float(scores_tf.mean())
sigma = float(scores_tf.std())

if threshold_mode == "p99":
    thr = float(np.percentile(scores_tf, 99))
elif threshold_mode == "p95":
    thr = float(np.percentile(scores_tf, 95))
elif threshold_mode == "2-sigma":
    thr = mu + 2 * sigma
elif threshold_mode == "3-sigma":
    thr = mu + 3 * sigma
else:
    thr = float(np.percentile(scores_tf, custom_p))

anom_mask = scores_tf >= thr
anom_indices = np.where(anom_mask)[0]

# ---------- Summary metrics ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Windows", len(scores_tf))
c2.metric("Threshold", f"{thr:.6f}")
c3.metric("Anomalies flagged", int(anom_mask.sum()))
c4.metric("Anomaly rate", f"{100*anom_mask.mean():.2f}%")

# ---------- Timeline ----------
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(times, scores_tf, label="Transformer AE score", linewidth=1.5)
ax.axhline(thr, linestyle="--", label=f"Threshold ({threshold_mode})")
ax.scatter(times[anom_mask], scores_tf[anom_mask], s=18, label="Flagged anomalies")

if show_if and scores_if is not None:
    ax.plot(times, scores_if, label="IsolationForest score", alpha=0.6)

ax.set_title("Anomaly Score Timeline")
ax.set_xlabel("Time")
ax.set_ylabel("Score (higher = more anomalous)")
ax.legend()
st.pyplot(fig)

# ---------- Top anomalies ----------
st.subheader(f"📌 Top {top_n} anomaly windows (by Transformer AE score)")
top_idx = np.argsort(scores_tf)[::-1][:top_n]

table = pd.DataFrame({
    "rank": np.arange(1, top_n + 1),
    "index": top_idx,
    "time": times.iloc[top_idx].values,
    "score_transformer": scores_tf[top_idx],
    "is_flagged": (scores_tf[top_idx] >= thr)
})

st.dataframe(table, use_container_width=True)

csv = table.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download anomalies table (CSV)",
    data=csv,
    file_name="anomaly_windows.csv",
    mime="text/csv"
)

# ---------- Task 2: Saliency ----------
st.subheader("🔍 Task 2 — Reconstruction Error Saliency")

if saliency_map is not None:
    default_idx = int(np.argmax(scores_tf))
    selected_idx = st.slider("Select anomaly index", 0, len(scores_tf)-1, default_idx)

    st.write("Selected time:", times.iloc[selected_idx])
    st.write("Selected anomaly score:", float(scores_tf[selected_idx]))

    st.info(
        "These saliency visuals are based on saved notebook artifacts. "
        "They show which time steps and meters contributed most to anomaly detection in the analyzed window."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(saliency_map.T, aspect="auto")
        ax.set_title("Saliency Map (Time vs Meters)")
        ax.set_xlabel("Time step (168 hours)")
        ax.set_ylabel("Meter index")
        st.pyplot(fig)

    with col2:
        if meter_importance is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(len(meter_importance)), meter_importance)
            ax.set_title("Meter-wise Saliency")
            ax.set_xlabel("Meter index")
            ax.set_ylabel("Importance")
            st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        if hour_of_day_saliency is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(range(24), hour_of_day_saliency)
            ax.set_title("Hour-of-Day Saliency")
            ax.set_xlabel("Hour of day")
            ax.set_ylabel("Aggregated saliency")
            st.pyplot(fig)

    with col4:
        if time_importance is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(time_importance)
            ax.set_title("Time-step Saliency in Window")
            ax.set_xlabel("Hour in window")
            ax.set_ylabel("Importance")
            ax.grid(True)
            st.pyplot(fig)

    if top_meter_idx is not None:
        st.write("### Top contributing meters")
        for i, m in enumerate(top_meter_idx.tolist()):
            meter_label = meters[m] if meters is not None and m < len(meters) else f"Meter {m}"
            st.write(f"{i+1}. {meter_label}")

    st.info("""
            Interpretation:
            - Bright regions in the saliency map show the time steps and meters that contributed most to reconstruction error.
            - Higher meter-wise saliency means that meter had stronger influence on anomaly detection.
            - Hour-of-day saliency helps show whether anomalies are concentrated at certain daily usage periods.
            """)    

else:
    st.warning("No saliency artifacts found. Save saliency_map.npy, meter_importance.npy, hour_of_day_saliency.npy, and time_importance.npy from Stage 9.")

# ---------- Task 4: Seasonal Decomposition + Anomaly Type ----------
st.subheader("📊 Task 4 — Anomaly Type Analysis (Trend vs Seasonal)")

if component_results_df is not None:
    col5, col6 = st.columns(2)

    with col5:
        st.write("### Component PR-AUC Comparison")
        st.dataframe(component_results_df, use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(component_results_df["component"], component_results_df["pr_auc"])
        ax.set_title("PR-AUC Across Components")
        ax.set_xlabel("Component")
        ax.set_ylabel("PR-AUC")
        st.pyplot(fig)

    with col6:
        if "recon_error_clean" in component_results_df.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(component_results_df["component"], component_results_df["recon_error_clean"])
            ax.set_title("Clean Reconstruction Error Across Components")
            ax.set_xlabel("Component")
            ax.set_ylabel("Mean Clean Reconstruction Error")
            st.pyplot(fig)

    best_component = component_results_df.sort_values("pr_auc", ascending=False).iloc[0]["component"]
    st.success(f"Best anomaly-capturing component: **{best_component}**")

    if best_component == "residual":
        st.info("Recommendation: anomalies appear more related to residual/noise-like deviations than regular seasonal patterns.")
    elif best_component == "seasonal":
        st.info("Recommendation: anomalies appear strongly connected to seasonal deviations.")
    else:
        st.info("Recommendation: full-series modeling currently captures anomalies best.")

else:
    st.warning("No component comparison file found. Save component_results.csv from Stage 10.")

# ---------- Anomaly type summary ----------
if anomaly_type_summary is not None and len(anomaly_type_summary) > 0:
    st.write("### Selected Anomaly Type")
    st.dataframe(anomaly_type_summary, use_container_width=True)

    predicted_type = anomaly_type_summary.iloc[0]["predicted_type"]
    st.write(f"Predicted anomaly type: **{predicted_type}**")

    st.write("Explanation:")
    if predicted_type == "trend_break":
        st.write("This anomaly looks like a sudden shift in the overall consumption trend.")
    elif predicted_type == "seasonal_violation":
        st.write("This anomaly looks like a deviation from the expected seasonal or repeating usage pattern.")
    else:
        st.write("This anomaly looks like a noise spike or irregular short-duration disturbance.")

# ---------- Meter-level anomaly type ----------
if meter_type_df is not None:
    st.write("### Meter-level Anomaly Type Breakdown")
    st.dataframe(meter_type_df, use_container_width=True)

    if "predicted_type" in meter_type_df.columns:
        counts = meter_type_df["predicted_type"].value_counts()

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(counts.index, counts.values)
        ax.set_title("Predicted Anomaly Type by Meter")
        ax.set_xlabel("Anomaly Type")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.caption("Note: if one anomaly type dominates this chart, it means the selected meters show a similar anomaly pattern in the current analysis window.")

else:
    st.warning("No meter_type_df.csv found. Save it from Stage 10.")

# ---------- Final insights ----------
st.subheader("📌 Final Insights")

st.success("""
- The Transformer Autoencoder identifies high-scoring anomalous windows from weekly electricity load patterns.
- Saliency analysis highlights which meters and time steps contribute most to reconstruction error.
- Seasonal decomposition helps separate anomalies into trend-related, seasonal, or residual/noise-driven behavior.
- Together, these views make the anomaly detector more interpretable and useful for investigation.
""")

# ---------- Notes ----------
with st.expander("What does this dashboard show?"):
    st.write(
        """
        - **Task 2**:
          - anomaly score timeline
          - saliency map for anomalous windows
          - meter-wise saliency
          - hour-of-day saliency
        - **Task 4**:
          - component comparison across full / seasonal / residual
          - anomaly type breakdown
          - meter-level anomaly categorization
        - Optional overlay:
          - IsolationForest baseline
        """
    )