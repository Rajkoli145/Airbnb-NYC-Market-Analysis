import streamlit as st

st.title("Airbnb NYC Revenue Predictor")

st.markdown("Enter listing details to predict whether it will generate high revenue.")

price = st.number_input("Price ($)", min_value=0.0)
minimum_nights = st.number_input("Minimum Nights", min_value=1)
availability = st.number_input("Availability (365 days)", min_value=0)

room_type = st.selectbox(
    "Room Type",
    ["Entire home/apt", "Private room", "Shared room"]
)

neighbourhood_group = st.selectbox(
    "Neighbourhood Group",
    ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
)

if st.button("Predict Revenue"):
    st.success("Prediction model will be integrated soon.")