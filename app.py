import streamlit as st
import pandas as pd
import joblib
import json
import os

# 1. Set Page Config to 'wide' to fill the iframe width
st.set_page_config(page_title="German Rent Predictor", layout="wide")

# 2. Custom CSS to remove top padding and hide Streamlit branding for a "fitted" look
st.markdown("""
    <style>
    /* Remove top padding of the main container */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    /* Hide the Streamlit header and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Style the metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.0rem;
    }
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

# --- App Header (Smaller) ---
st.subheader("🏠 Germany Rent Predictor")

if model is None:
    st.error("Model file not found!")
    st.stop()

# 3. Use Columns to fit everything on one screen height
col_input, col_result = st.columns([1.2, 1])

with col_input:
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            living_space = st.number_input("Living Space (m²)", 10.0, 500.0, 60.0)
            no_rooms = st.number_input("Rooms", 1, 10, 3, step=1)
            year_constructed = st.number_input("Year Built", 1900, 2025, 2015, step=1)
        with c2:
            selected_state = st.selectbox("State", [
                'Nordrhein_Westfalen', 'Sachsen', 'Bremen', 'Bayern', 'Berlin',
                'Hessen', 'Hamburg', 'Baden_Wuerttemberg', 'Thueringen', 'Sachsen_Anhalt', 'Other'
            ])
            selected_heating = st.selectbox("Heating", [
                'central_heating', 'floor_heating', 'district_heating',
                'gas_heating', 'oil_heating', 'self_contained_central_heating'
            ])
            has_balcony = st.checkbox("Balcony", value=True)
            is_newly_built = st.checkbox("Newly Built", value=False)
        st.subheader("🧠Can you guess closer than our model?")

        actual_rent_input = st.number_input("Enter Your prediction(€)", min_value=0.0, value=0.0, step=10.0)
        submit_button = st.button("Predict Rent 💶", use_container_width=True, type="primary")

with col_result:
    # --- Prediction Logic ---
    if submit_button:
        # Prepare input data
        input_data = pd.DataFrame({
            'livingSpace': [living_space],
            'noRooms': [no_rooms],
            'heatingType': [selected_heating],
            'balcony': [1.0 if has_balcony else 0.0],
            'newlyConst': [1.0 if is_newly_built else 0.0],
            'yearConstructed': [year_constructed],
            'state': [selected_state]
        })

        # MOVE TRY BLOCK INSIDE THE IF STATEMENT
        try:
            prediction = model.predict(input_data)[0]

            # Display Result
            st.success(f"### Estimated Rent: €{prediction:,.2f}")

            # Comparison Logic
            if actual_rent_input > 0:
                diff = prediction - actual_rent_input

                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Model Prediction", f"€{prediction:,.0f}")
                col_b.metric("Your Input", f"€{actual_rent_input:,.0f}")

                # Color logic for difference
                if abs(diff) < 50:
                    col_c.metric("Difference", f"€{diff:,.0f}", delta_color="off")
                    st.caption("✅ The prediction is very close!")
                else:
                    # If prediction is higher than actual, it shows red (overpriced estimate)
                    # If prediction is lower than actual, it shows green (good deal?)
                    col_c.metric("Difference", f"€{diff:,.0f}", delta=-diff)


        except Exception as e:
            st.error(f"Error: {e}")
    else:
        # This shows BEFORE the user clicks the button
        st.info("Enter details and click **Predict** to see the estimate.")
        st.caption("⚠️ Based on historical ImmoScout24 data.")