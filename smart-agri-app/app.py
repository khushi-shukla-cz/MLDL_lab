import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# LOAD MODEL COMPONENTS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Crop_recommendation.csv"

FEATURE_COLUMNS = [
    "Nitrogen",
    "phosphorus",
    "potassium",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
]

FEATURE_LABELS = {
    "Nitrogen": "Nitrogen (N)",
    "phosphorus": "Phosphorus (P)",
    "potassium": "Potassium (K)",
    "temperature": "Temperature",
    "humidity": "Humidity",
    "ph": "Soil pH",
    "rainfall": "Rainfall",
}


def build_fallback_components():
    # Synthetic training set for demo use when artifacts are unavailable.
    x = np.array([
        [90, 40, 40, 25, 80, 6.5, 200],
        [100, 35, 35, 26, 78, 6.3, 180],
        [20, 20, 20, 28, 65, 6.8, 120],
        [18, 22, 25, 27, 60, 7.0, 110],
        [50, 60, 50, 22, 85, 6.0, 250],
        [55, 65, 48, 23, 88, 5.8, 260],
        [70, 55, 45, 24, 70, 6.2, 150],
        [75, 50, 50, 25, 72, 6.4, 160],
        [35, 25, 30, 30, 55, 7.2, 90],
        [32, 28, 28, 31, 52, 7.1, 85],
    ], dtype=float)

    y_labels = np.array([
        "rice", "rice", "millet", "millet", "sugarcane",
        "sugarcane", "maize", "maize", "cotton", "cotton",
    ])

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_labels)

    fitted_scaler = StandardScaler()
    x_scaled = fitted_scaler.fit_transform(x)

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(x_scaled, y_encoded)

    return clf, fitted_scaler, encoder


def safe_load_pickle(file_name):
    file_path = BASE_DIR / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {file_name}")
    if file_path.stat().st_size == 0:
        raise EOFError(f"Empty file: {file_name}")
    with open(file_path, "rb") as f:
        return pickle.load(f)


def load_crop_reference_data(path):
    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    expected = set(FEATURE_COLUMNS + ["label"])
    if not expected.issubset(df.columns):
        return None

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = df["label"].astype(str)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
    return df


load_error = None
try:
    model = safe_load_pickle("model.pkl")
    scaler = safe_load_pickle("scaler.pkl")
    le = safe_load_pickle("label_encoder.pkl")
except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
    model, scaler, le = build_fallback_components()
    load_error = str(e)

crop_reference_df = load_crop_reference_data(DATA_PATH)
crop_stats = {}
if crop_reference_df is not None and not crop_reference_df.empty:
    grouped = crop_reference_df.groupby("label")
    for crop_name, group in grouped:
        crop_stats[crop_name] = {
            "mean": group[FEATURE_COLUMNS].mean().to_dict(),
            "std": group[FEATURE_COLUMNS].std(ddof=0).replace(0, 1.0).fillna(1.0).to_dict(),
        }

# -----------------------------
# UI CONFIG
# -----------------------------
st.set_page_config(page_title="Smart Agriculture DSS", layout="centered")

st.title("🌾 Smart Agriculture Decision Support System")
st.write("Predict the best crop based on soil and weather conditions")

if load_error:
    st.warning(
        "Model artifacts are missing or invalid. Running with a built-in fallback model. "
        "Upload valid model.pkl, scaler.pkl, and label_encoder.pkl for production predictions."
    )
    st.caption(f"Details: {load_error}")

# -----------------------------
# INPUT SECTION
# -----------------------------
st.sidebar.header("Enter Soil & Weather Parameters")

N = st.sidebar.slider("Nitrogen (N)", 0, 150, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 40)
K = st.sidebar.slider("Potassium (K)", 0, 150, 40)
temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 25)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 80)
ph = st.sidebar.slider("pH Level", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0, 300, 200)

input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_crop(input_data):
    scaled = scaler.transform(input_data)
    
    probs = model.predict_proba(scaled)[0]
    classes = le.inverse_transform(model.classes_)
    
    top_indices = np.argsort(probs)[::-1][:3]
    
    return classes, probs, top_indices


def get_feature_importances():
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
        total = importances.sum()
        if total > 0:
            importances = importances / total
    else:
        importances = np.repeat(1.0 / len(FEATURE_COLUMNS), len(FEATURE_COLUMNS))

    return {feature: float(value) for feature, value in zip(FEATURE_COLUMNS, importances)}


def compute_local_support(input_row, top_crop, runner_up_crop, importances):
    if top_crop not in crop_stats or runner_up_crop not in crop_stats:
        return []

    values = dict(zip(FEATURE_COLUMNS, input_row[0]))
    support_rows = []

    for feature in FEATURE_COLUMNS:
        top_mean = crop_stats[top_crop]["mean"][feature]
        top_std = crop_stats[top_crop]["std"][feature] + 1e-6
        runner_mean = crop_stats[runner_up_crop]["mean"][feature]
        runner_std = crop_stats[runner_up_crop]["std"][feature] + 1e-6

        value = values[feature]
        dist_top = abs(value - top_mean) / top_std
        dist_runner = abs(value - runner_mean) / runner_std

        support_score = (dist_runner - dist_top) * importances[feature]
        support_rows.append(
            {
                "feature": FEATURE_LABELS[feature],
                "value": value,
                "support_score": support_score,
                "top_crop": top_crop,
                "runner_crop": runner_up_crop,
            }
        )

    support_rows.sort(key=lambda row: row["support_score"], reverse=True)
    return support_rows


def build_improvement_tips(input_row, top_crop, runner_up_crop):
    tips = []
    values = dict(zip(FEATURE_COLUMNS, input_row[0]))

    if top_crop not in crop_stats:
        return tips

    top_mean = crop_stats[top_crop]["mean"]

    if values["ph"] > 6.8:
        tips.append("Lower soil pH gradually toward ~6.5 using acidifying amendments and organic matter.")
    elif values["ph"] < 6.0:
        tips.append("Raise soil pH toward ~6.5 using agricultural lime after a soil test.")

    for feature, nutrient_name in [
        ("Nitrogen", "Nitrogen (N)"),
        ("phosphorus", "Phosphorus (P)"),
        ("potassium", "Potassium (K)"),
    ]:
        gap = float(top_mean[feature] - values[feature])
        if gap > 10:
            tips.append(f"{nutrient_name} appears low for {top_crop}; increase it in the next nutrient schedule.")
        elif gap < -15:
            tips.append(f"{nutrient_name} is already high for {top_crop}; avoid over-application to reduce waste and burn risk.")

    if runner_up_crop in crop_stats:
        top_h = crop_stats[top_crop]["mean"]["humidity"]
        runner_h = crop_stats[runner_up_crop]["mean"]["humidity"]
        val_h = values["humidity"]

        if abs(val_h - top_h) < abs(val_h - runner_h):
            tips.append(
                f"Current humidity supports {top_crop} more than {runner_up_crop}, which strengthens this recommendation."
            )
        else:
            tips.append(
                f"Current humidity is closer to {runner_up_crop} than {top_crop}; improved humidity control can increase confidence."
            )

    return tips


def build_fertilizer_plan(input_row, top_crop):
    plan = []
    values = dict(zip(FEATURE_COLUMNS, input_row[0]))

    if top_crop not in crop_stats:
        return plan

    target = crop_stats[top_crop]["mean"]
    nutrient_meta = [
        ("Nitrogen", "N", "Urea / N-rich source"),
        ("phosphorus", "P", "DAP / SSP (P source)"),
        ("potassium", "K", "MOP / SOP (K source)"),
    ]

    for feature, short_name, source in nutrient_meta:
        current_value = float(values[feature])
        target_value = float(target[feature])
        gap = target_value - current_value

        if gap > 5:
            plan.append(
                f"{short_name}: Increase by about {gap:.1f} soil-index points for {top_crop} profile. Suggested source: {source}."
            )
        elif gap < -5:
            plan.append(
                f"{short_name}: Currently above typical {top_crop} profile by {abs(gap):.1f}; reduce or pause {source}."
            )
        else:
            plan.append(f"{short_name}: Near the target range for {top_crop}; maintain current level.")

    return plan


def clamp_input_ranges(input_row):
    bounded = input_row.copy().astype(float)
    bounded[0, 0] = np.clip(bounded[0, 0], 0, 150)   # Nitrogen
    bounded[0, 1] = np.clip(bounded[0, 1], 0, 150)   # Phosphorus
    bounded[0, 2] = np.clip(bounded[0, 2], 0, 150)   # Potassium
    bounded[0, 3] = np.clip(bounded[0, 3], 0, 50)    # Temperature
    bounded[0, 4] = np.clip(bounded[0, 4], 0, 100)   # Humidity
    bounded[0, 5] = np.clip(bounded[0, 5], 0, 14)    # pH
    bounded[0, 6] = np.clip(bounded[0, 6], 0, 300)   # Rainfall
    return bounded


def run_monte_carlo_simulation(base_input, trials=300, noise_pct=0.08):
    if trials <= 0:
        return pd.DataFrame(columns=["Crop", "Simulation Share (%)"])

    base = base_input.astype(float)[0]
    floor_noise = np.array([4.0, 4.0, 4.0, 1.2, 2.5, 0.2, 10.0])
    scales = np.maximum(np.abs(base) * noise_pct, floor_noise)

    samples = np.random.normal(loc=base, scale=scales, size=(trials, len(FEATURE_COLUMNS)))

    # Clamp sampled values to realistic UI ranges.
    samples[:, 0] = np.clip(samples[:, 0], 0, 150)
    samples[:, 1] = np.clip(samples[:, 1], 0, 150)
    samples[:, 2] = np.clip(samples[:, 2], 0, 150)
    samples[:, 3] = np.clip(samples[:, 3], 0, 50)
    samples[:, 4] = np.clip(samples[:, 4], 0, 100)
    samples[:, 5] = np.clip(samples[:, 5], 0, 14)
    samples[:, 6] = np.clip(samples[:, 6], 0, 300)

    scaled = scaler.transform(samples)
    pred_encoded = model.predict(scaled)
    pred_labels = le.inverse_transform(pred_encoded.astype(int))

    dist = (
        pd.Series(pred_labels)
        .value_counts(normalize=True)
        .mul(100)
        .rename_axis("Crop")
        .reset_index(name="Simulation Share (%)")
        .sort_values("Simulation Share (%)", ascending=False)
    )
    return dist

# -----------------------------
# PREDICTION TRIGGER + STATE
# -----------------------------
if "prediction_payload" not in st.session_state:
    st.session_state.prediction_payload = None

if st.button("🌱 Predict Best Crop"):
    classes, probs, top_indices = predict_crop(input_data)
    st.session_state.prediction_payload = {
        "classes": classes.tolist(),
        "probs": probs.tolist(),
        "top_indices": top_indices.tolist(),
        "input_data": input_data.tolist(),
    }

payload = st.session_state.prediction_payload

if payload is not None:
    classes = np.array(payload["classes"])
    probs = np.array(payload["probs"], dtype=float)
    top_indices = np.array(payload["top_indices"], dtype=int)
    active_input = np.array(payload["input_data"], dtype=float)

    st.subheader("🌾 Prediction Result")

    best_crop = classes[top_indices[0]]
    confidence = probs[top_indices[0]] * 100

    st.success(f"Best Recommended Crop: {best_crop}")
    st.info(f"Confidence: {confidence:.2f}%")

    if confidence < 60:
        st.warning("Prediction confidence is moderate. Review top 3 crops and field conditions before final decision.")

    # Top 3 crops
    st.subheader("📊 Top 3 Crop Suggestions")

    for i in top_indices:
        st.write(f"🌿 {classes[i]} : {probs[i]*100:.2f}%")

    # Bar chart visualization
    st.subheader("📈 Probability Distribution")

    chart_data = {
        classes[i]: probs[i] for i in top_indices
    }

    st.bar_chart(chart_data)

    st.subheader("🔍 Why This Crop Was Predicted")

    importances = get_feature_importances()
    importance_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
            "Model Importance (%)": [importances[f] * 100 for f in FEATURE_COLUMNS],
        }
    ).sort_values("Model Importance (%)", ascending=False)
    st.write("Model-level influence of each feature:")
    st.dataframe(importance_df, use_container_width=True)

    if len(top_indices) > 1:
        runner_up = classes[top_indices[1]]
        support_rows = compute_local_support(active_input, best_crop, runner_up, importances)

        if support_rows:
            positive = [row for row in support_rows if row["support_score"] > 0][:3]
            negative = sorted([row for row in support_rows if row["support_score"] < 0], key=lambda x: x["support_score"])[:2]

            if positive:
                st.write(f"Top factors supporting {best_crop} over {runner_up}:")
                for row in positive:
                    st.write(
                        f"- {row['feature']} favored {best_crop} over {runner_up} (input: {row['value']:.2f})."
                    )

            if negative:
                st.write(f"Factors currently pulling toward {runner_up}:")
                for row in negative:
                    st.write(
                        f"- {row['feature']} is closer to {runner_up} conditions (input: {row['value']:.2f})."
                    )

    st.subheader("🛠️ What To Improve")
    if len(top_indices) > 1:
        runner_up = classes[top_indices[1]]
    else:
        runner_up = best_crop

    tips = build_improvement_tips(active_input, best_crop, runner_up)
    if tips:
        for tip in tips:
            st.write(f"- {tip}")
    else:
        st.write("- Your current profile is already close to the recommended crop baseline.")

    st.subheader("🧪 Fertilizer Recommendation (NPK)")
    st.caption("Guidance is data-driven from crop profiles. Validate final dose with local soil testing and agronomy advice.")
    fertilizer_plan = build_fertilizer_plan(active_input, best_crop)
    if fertilizer_plan:
        for item in fertilizer_plan:
            st.write(f"- {item}")
    else:
        st.write("- Fertilizer recommendation is unavailable because reference crop data could not be loaded.")

    st.subheader("📊 Visual Field Snapshot")
    current_values = dict(zip(FEATURE_COLUMNS, active_input[0]))
    if best_crop in crop_stats:
        visual_data = {
            "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
            "Current": [current_values[f] for f in FEATURE_COLUMNS],
            f"{best_crop} Baseline": [crop_stats[best_crop]["mean"][f] for f in FEATURE_COLUMNS],
        }
        if len(top_indices) > 1 and runner_up in crop_stats:
            visual_data[f"{runner_up} Baseline"] = [crop_stats[runner_up]["mean"][f] for f in FEATURE_COLUMNS]

        visual_df = pd.DataFrame(visual_data).set_index("Feature")
        st.bar_chart(visual_df)
        st.caption("Compare your field values with the recommended crop baseline and runner-up baseline.")

    st.subheader("🎛️ What-If Simulation")
    st.caption("Adjust field conditions and instantly see if the recommendation changes.")

    sim_col_1, sim_col_2, sim_col_3 = st.columns(3)
    with sim_col_1:
        d_n = st.slider("Delta N", -40, 40, 0, key="sim_d_n")
        d_temp = st.slider("Delta Temp", -10, 10, 0, key="sim_d_temp")
        d_ph = st.slider("Delta pH x0.1", -20, 20, 0, key="sim_d_ph")
    with sim_col_2:
        d_p = st.slider("Delta P", -40, 40, 0, key="sim_d_p")
        d_humidity = st.slider("Delta Humidity", -30, 30, 0, key="sim_d_humidity")
    with sim_col_3:
        d_k = st.slider("Delta K", -40, 40, 0, key="sim_d_k")
        d_rainfall = st.slider("Delta Rainfall", -100, 100, 0, key="sim_d_rainfall")

    sim_input = active_input.copy()
    sim_input[0, 0] += d_n
    sim_input[0, 1] += d_p
    sim_input[0, 2] += d_k
    sim_input[0, 3] += d_temp
    sim_input[0, 4] += d_humidity
    sim_input[0, 5] += d_ph / 10.0
    sim_input[0, 6] += d_rainfall
    sim_input = clamp_input_ranges(sim_input)

    sim_classes, sim_probs, sim_top_indices = predict_crop(sim_input)
    sim_best_crop = sim_classes[sim_top_indices[0]]
    sim_conf = sim_probs[sim_top_indices[0]] * 100

    if sim_best_crop != best_crop:
        st.warning(
            f"Simulation changed recommendation: {best_crop} → {sim_best_crop} (confidence: {sim_conf:.2f}%)."
        )
    else:
        st.info(f"Simulation keeps recommendation as {sim_best_crop} (confidence: {sim_conf:.2f}%).")

    compare_df = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
            "Original": [current_values[f] for f in FEATURE_COLUMNS],
            "Simulated": [dict(zip(FEATURE_COLUMNS, sim_input[0]))[f] for f in FEATURE_COLUMNS],
        }
    ).set_index("Feature")
    st.dataframe(compare_df, use_container_width=True)

    st.subheader("🧬 Robustness Simulation")
    st.caption("Monte Carlo simulation estimates how stable the recommendation is under noisy real-world conditions.")

    mc_col_1, mc_col_2 = st.columns(2)
    with mc_col_1:
        mc_trials = st.slider("Trials", 100, 1000, 300, step=100, key="mc_trials")
    with mc_col_2:
        mc_noise_pct = st.slider("Uncertainty (%)", 1, 20, 8, step=1, key="mc_noise") / 100.0

    mc_dist = run_monte_carlo_simulation(active_input, trials=mc_trials, noise_pct=mc_noise_pct)

    if not mc_dist.empty:
        st.bar_chart(mc_dist.set_index("Crop"))
        st.dataframe(mc_dist, use_container_width=True)

        best_row = mc_dist[mc_dist["Crop"] == best_crop]
        if not best_row.empty:
            stability = float(best_row["Simulation Share (%)"].iloc[0])
            st.write(f"Stability score for {best_crop}: {stability:.2f}% of simulated scenarios.")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ for Smart Agriculture DSS")