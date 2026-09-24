import streamlit as st
import pandas as pd
import joblib


# Load trained model
model = joblib.load("models/random_forest_model.pkl")

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)


# Streamlit page settings
st.set_page_config(
    page_title="Retail Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)


st.title("🛒 Retail Customer Purchase Prediction")

st.write(
    "Predict whether a customer is likely to purchase "
    "based on their website activity."
)

st.divider()


# Customer details
st.subheader("Customer Information")

col1, col2 = st.columns(2)


with col1:

    pages_visited = st.number_input(
        "Pages Visited",
        min_value=0.0,
        value=5.0
    )

    time_spent = st.number_input(
        "Time Spent",
        min_value=0.0,
        value=10.0
    )

    products_viewed = st.number_input(
        "Products Viewed",
        min_value=0.0,
        value=5.0
    )

    cart_activity = st.selectbox(
        "Cart Activity",
        ["No", "Yes"]
    )

    cart_items = st.number_input(
        "Cart Items",
        min_value=0.0,
        value=1.0
    )


with col2:

    previous_purchases = st.number_input(
        "Previous Purchases",
        min_value=0.0,
        value=2.0
    )

    discount_used = st.selectbox(
        "Discount Used",
        ["No", "Yes"]
    )

    device_type = st.selectbox(
        "Device Type",
        ["Desktop", "Mobile", "Tablet"]
    )

    traffic_source = st.selectbox(
        "Traffic Source",
        [
            "Direct",
            "Email",
            "Organic Search",
            "Paid Search",
            "Referral",
            "Social Media"
        ]
    )

    session_count = st.number_input(
        "Session Count",
        min_value=0.0,
        value=2.0
    )


st.divider()


# Predict button
if st.button("Predict Purchase", use_container_width=True):

    # Values which are not asked from the user
    # are given default values.
    data = {
        "Pages_Visited": pages_visited,
        "Time_Spent": time_spent,
        "Previous_Visits": 3.0,
        "Products_Viewed": products_viewed,
        "Cart_Items": cart_items,
        "Wishlist_Items": 1.0,
        "Searches_Performed": 3.0,
        "Product_Detail_Views": 4.0,
        "Previous_Purchases": previous_purchases,
        "Previous_Total_Spend": 5000.0,
        "Average_Order_Value": 2500.0,
        "Days_Since_Last_Visit": 5.0,
        "Days_Since_Last_Purchase": 20.0,
        "Session_Count": session_count
    }

    input_data = pd.DataFrame([data])


    # Convert categorical values into 0 and 1
    input_data["Cart_Activity_Yes"] = (
        cart_activity == "Yes"
    )

    input_data["Discount_Viewed_Yes"] = False

    input_data["Discount_Used_Yes"] = (
        discount_used == "Yes"
    )

    input_data["Device_Type_Mobile"] = (
        device_type == "Mobile"
    )

    input_data["Device_Type_Tablet"] = (
        device_type == "Tablet"
    )

    input_data["Traffic_Source_Email"] = (
        traffic_source == "Email"
    )

    input_data["Traffic_Source_Organic Search"] = (
        traffic_source == "Organic Search"
    )

    input_data["Traffic_Source_Paid Search"] = (
        traffic_source == "Paid Search"
    )

    input_data["Traffic_Source_Referral"] = (
        traffic_source == "Referral"
    )

    input_data["Traffic_Source_Social Media"] = (
        traffic_source == "Social Media"
    )


    # Keep the same columns used during training
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=False
    )


    # Make prediction
    prediction = model.predict(input_data)

    probability = model.predict_proba(input_data)[0][1]


    # Show result
    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.success(
            "🛒 Customer is likely to PURCHASE."
        )

    else:

        st.warning(
            "Customer is unlikely to PURCHASE."
        )

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )