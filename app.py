import streamlit as st
import pandas as pd
import joblib
import json
import os

# Page Config - Set to "wide" to use screen space efficiently
st.set_page_config(page_title="German Rent Predictor", layout="wide")

# Custom CSS to reduce padding and make cards look better
st.markdown("""
    <style>
    .main > div { padding-top: 2rem; }
    .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    div[data-testid="stExpander"] { border: none; box-shadow: none; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_data():
    model_path = 'model/housing_model.pkl'
    metrics_path = 'model/metrics.json'
    if not os.path.exists(model_path):
        return None, None
    model = joblib.load(model_path)
    metrics = {"r2_score": 0}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    return model, metrics


model, metrics = load_data()

# --- Header ---
st.title("🏠 Germany Rent Predictor")

if model is None:
    st.error("Model file not found! Please run `python train_model.py` first.")
    st.stop()

# --- Sidebar for Inputs (Keeps the main page clean and small) ---
st.sidebar.header("📍 Apartment Features")
with st.sidebar:
    living_space = st.number_input("Living Space (m²)", 10.0, 500.0, 60.0)
    no_rooms = st.number_input("Rooms", 1, 10, 3, step=1)
    year_constructed = st.number_input("Year Built", 1900, 2025, 2015, step=1)

    selected_state = st.selectbox("State", [
        'Nordrhein_Westfalen', 'Sachsen', 'Bremen', 'Bayern', 'Berlin',
        'Hessen', 'Hamburg', 'Baden_Wuerttemberg', 'Thueringen', 'Sachsen_Anhalt', 'Other'
    ])

    selected_heating = st.selectbox("Heating", [
        'central_heating', 'floor_heating', 'district_heating',
        'gas_heating', 'oil_heating', 'self_contained_central_heating'
    ])

    c1, c2 = st.columns(2)
    has_balcony = c1.checkbox("Balcony", value=True)
    is_newly_built = c2.checkbox("New Built", value=False)

    st.markdown("---")
    actual_rent_input = st.number_input("Your Guess/Actual (€)", min_value=0.0, value=0.0)
    predict_btn = st.button("Predict Rent 💶", use_container_width=True, type="primary")

# --- Main Display Area ---
col_main, col_info = st.columns([2, 1])

with col_main:
    if predict_btn:
        input_data = pd.DataFrame({
            'livingSpace': [living_space], 'noRooms': [no_rooms],
            'heatingType': [selected_heating], 'balcony': [1.0 if has_balcony else 0.0],
            'newlyConst': [1.0 if is_newly_built else 0.0],
            'yearConstructed': [year_constructed], 'state': [selected_state]
        })

        try:
            prediction = model.predict(input_data)[0]

            # Big Result Card
            st.success(f"### Predicted Rent: **€{prediction:,.2f}**")

            # Comparison Metrics in a compact row
            if actual_rent_input > 0:
                diff = prediction - actual_rent_input
                m1, m2, m3 = st.columns(3)
                m1.metric("Model", f"€{prediction:,.0f}")
                m2.metric("Actual", f"€{actual_rent_input:,.0f}")
                m3.metric("Diff", f"€{diff:,.0f}", delta=-diff if diff != 0 else None)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("👈 Adjust features in the sidebar and click **Predict Rent**")

with col_info:
    # Small informational box
    with st.expander("📊 Model Info", expanded=True):
        if metrics:
            st.write(f"**Accuracy:** {metrics.get('r2_score', 0) * 100:.1f}%")
        st.caption("Data: ImmoScout24 historical prices.")

    st.warning("Tool for estimation only.")