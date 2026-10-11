
import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained model from the deployment folder
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_model.joblib"
)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please place best_model.joblib in the deployment folder."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

# Streamlit UI
st.set_page_config(
    page_title="SuperKart Sales Forecasting",
    page_icon="📊",
    layout="centered"
)

st.title("SuperKart Sales Forecasting")
st.write(
    "Enter product and store details to estimate "
    "the expected sales revenue."
)

# Product inputs
st.subheader("Product Details")

Product_Weight = st.number_input(
    "Product Weight",
    min_value=0.0,
    value=12.0,
    step=0.1
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.01
)

Product_Type = st.selectbox(
    "Product Type",
    [
        "Meat", "Snack Foods", "Hard Drinks", "Dairy",
        "Canned", "Soft Drinks", "Health and Hygiene",
        "Baking Goods", "Bread", "Breakfast", "Frozen Foods",
        "Fruits and Vegetables", "Household", "Seafood",
        "Starchy Foods", "Others"
    ]
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=0.0,
    value=150.0,
    step=5.0
)

# Store inputs
st.subheader("Store Details")

Store_Establishment_Year = st.number_input(
    "Store Establishment Year",
    min_value=1980,
    max_value=2026,
    value=2000,
    step=1
)

Store_Size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Supermarket Type 1",
        "Supermarket Type 2",
        "Food Mart"
    ]
)

# Build input dataframe with the exact feature names
input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_Type": Product_Type,
    "Product_MRP": Product_MRP,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type
}])

# Prediction
if st.button("Predict Sales Revenue", type="primary"):
    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Predicted Sales Revenue")
        st.metric(
            label="Estimated Revenue",
            value=f"{prediction:,.2f}"
        )

        st.caption(
            "This estimate is based on the trained model and "
            "the product/store attributes entered above."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
