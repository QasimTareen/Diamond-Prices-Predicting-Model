import os

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="Diamond Price Predictor", page_icon="💎", layout="centered")

st.title("💎 Diamond Price Predictor")
st.caption("Machine Learning Fundamentals (AIC354) - Assignment 3. "
           "Predicts the price of a diamond in USD from its specifications.")

# ---------------------------------------------------------------
# Load the trained model (path works from any folder)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diamond_price_model.pkl")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not os.path.exists(MODEL_PATH):
    st.error("Model file `diamond_price_model.pkl` was not found. "
             "Upload it to the same folder as `app.py` and reboot the app.")
    st.stop()

model = load_model()

# ---------------------------------------------------------------
# Input form
# ---------------------------------------------------------------
st.subheader("Enter diamond details")

col1, col2 = st.columns(2)

with col1:
    carat = st.number_input("Carat (weight)", min_value=0.2, max_value=5.5, value=0.90, step=0.01)
    cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
    color = st.selectbox("Color (D = best, J = worst)", ["D", "E", "F", "G", "H", "I", "J"], index=3)
    clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=3)

with col2:
    depth = st.number_input("Depth %", min_value=40.0, max_value=80.0, value=61.5, step=0.1)
    table = st.number_input("Table %", min_value=40.0, max_value=100.0, value=57.0, step=0.1)
    x = st.number_input("Length x (mm)", min_value=1.0, max_value=12.0, value=6.10, step=0.01)
    y = st.number_input("Width y (mm)", min_value=1.0, max_value=60.0, value=6.15, step=0.01)
    z = st.number_input("Height z (mm)", min_value=1.0, max_value=35.0, value=3.77, step=0.01)

# ---------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------
if st.button("Predict Price", type="primary"):
    # Column names must match the training data exactly
    input_df = pd.DataFrame([{
        "carat": carat,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z,
        "cut": cut,
        "color": color,
        "clarity": clarity,
    }])

    try:
        price = float(model.predict(input_df)[0])
        price = max(price, 0.0)   # price can never be negative
        st.success(f"Estimated price: **${price:,.0f}**")
        st.caption("This is an estimate from a machine learning model trained on about 54,000 diamonds.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("About this model"):
    st.write(
        "Five algorithms (Ridge, Lasso, Extra Trees, Hist Gradient Boosting and MLP) were "
        "compared using 10-Fold Cross-Validation and RMSE. The best one was trained on the "
        "full dataset and deployed here."
    )