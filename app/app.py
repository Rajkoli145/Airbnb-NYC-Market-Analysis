import streamlit as st
import joblib
import numpy as np
import pandas as pd
import altair as alt
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Airbnb NYC Revenue Predictor", page_icon="📈", layout="wide")

# --- MODEL LOADING ---
@st.cache_resource
def load_assets():
    scaler = joblib.load("models/scaler.joblib")
    kmeans = joblib.load("models/kmeans.joblib")
    model = joblib.load("models/logistic_model.joblib")
    
    # Load a small sample of data for the market positioning plot
    # We use airbnb_cleaned.csv and apply the same logic as the training script
    try:
        data_sample = pd.read_csv("data/processed/airbnb_cleaned.csv").sample(1000, random_state=42)
        X_sample = data_sample[["price", "number of reviews", "availability 365"]]
        X_sample_scaled = scaler.transform(X_sample)
        data_sample["cluster"] = kmeans.predict(X_sample_scaled)
    except:
        data_sample = pd.DataFrame() # Fallback if data is missing
        
    return scaler, kmeans, model, data_sample

try:
    scaler, kmeans, model, data_sample = load_assets()
except Exception as e:
    st.error(f"Error loading models or data: {e}")
    st.stop()

# --- HEADER ---
st.title(" Airbnb NYC Revenue Predictor")
st.markdown("""
    Transform your Airbnb listing into a high-revenue asset. This system uses **Logistic Regression** 
    and **KMeans Clustering** to analyze market positioning and predict performance.
""")

st.divider()

# --- INPUTS & PREDICTION ---
with st.sidebar:
    st.header("Listing Details")
    price = st.number_input(" Price per Night ($)", min_value=1.0, value=150.0, step=10.0)
    availability = st.slider(" Availability (Days/Year)", 0, 365, 180)
    reviews = st.number_input(" Number of Reviews", min_value=0, value=20, step=1)
    
    predict_btn = st.button(" Run Analysis", use_container_width=True)
    if st.button(" Clear & Start New", use_container_width=True):
        st.rerun()

# --- PREDICTION LOGIC ---
if predict_btn:
    # 1. Preprocess
    input_features = np.array([[price, reviews, availability]])
    input_scaled = scaler.transform(input_features)
    cluster_label = kmeans.predict(input_scaled)[0]
    final_features = np.append(input_scaled, [[cluster_label]], axis=1)
    
    # 2. Predict
    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0][1]
    
    # --- LAYOUT ---
    main_col, side_col = st.columns([2, 1])
    
    with main_col:
        # Result Card
        if prediction == 1:
            st.success("### Prediction: **HIGH REVENUE**")
        else:
            st.warning("### Prediction: **LOW REVENUE**")
            
        # Probability Gauge (Dark Mode Compatible)
        gauge_color = "#28a745" if prediction == 1 else "#ffc107"
        st.markdown(f"""
            <div style="background-color: #262730; padding: 20px; border-radius: 10px; border: 1px solid #464646; margin-bottom: 20px;">
                <h4 style="margin-top: 0; color: white;">Model Confidence</h4>
                <div style="background-color: #464646; border-radius: 10px; height: 25px; width: 100%;">
                    <div style="background-color: {gauge_color}; width: {probability*100}%; height: 100%; border-radius: 10px; transition: width 1s;"></div>
                </div>
                <p style="text-align: right; font-weight: bold; margin-bottom: 0; color: #dcdcdc; padding-top: 5px;">
                    {probability:.1%} Probability of High Revenue
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Feature Importance
        st.subheader("Why this prediction?")
        coefs = model.coef_[0]
        features = ["Price", "Reviews", "Availability", "Market Segment"]
        importance_df = pd.DataFrame({"Feature": features, "Impact": coefs})
        
        importance_chart = alt.Chart(importance_df).mark_bar().encode(
            x=alt.X("Impact:Q", title="Impact Strength (Coefficient)"),
            y=alt.Y("Feature:N", sort="-x", title=""),
            color=alt.condition(
                alt.datum.Impact > 0,
                alt.value("#28a745"), # Positive impact
                alt.value("#dc3545")  # Negative impact
            )
        ).properties(height=250)
        
        st.altair_chart(importance_chart, use_container_width=True)

    with side_col:
        st.subheader("Market Positioning")
        if not data_sample.empty:
            # Market Position Scatter Plot
            user_point = pd.DataFrame({
                "price": [price],
                "availability 365": [availability],
                "type": ["Your Listing"]
            })
            
            base_data = data_sample.copy()
            base_data["type"] = "Market Data"
            
            combined_data = pd.concat([base_data, user_point])
            
            scatter = alt.Chart(combined_data).mark_circle(size=60).encode(
                x=alt.X("price:Q", title="Price ($)", scale=alt.Scale(domain=[0, 1200])),
                y=alt.Y("availability 365:Q", title="Availability (Days)"),
                color=alt.Color("cluster:N", legend=alt.Legend(title="Segment"), scale=alt.Scale(scheme="viridis")),
                opacity=alt.condition(alt.datum.type == "Your Listing", alt.value(1), alt.value(0.4)),
                stroke=alt.condition(alt.datum.type == "Your Listing", alt.value("red"), alt.value(None)),
                strokeWidth=alt.condition(alt.datum.type == "Your Listing", alt.value(3), alt.value(0))
            ).properties(height=400)
            
            st.altair_chart(scatter, use_container_width=True)
            st.info(f"**Identified Segment:** Cluster {cluster_label}")
        else:
            st.info("Market position plot is unavailable as dataset sample couldn't be loaded.")

else:
    # Welcome state
    st.info("👈 Enter listing details in the sidebar and click 'Run Analysis' to see predictions and insights.")
    
    # Optionally show general market summary
    if not data_sample.empty:
        st.subheader("Current Market Overview (Sample)")
        st.write("Browse how thousands of NYC listings are distributed by Price and Availability.")
        overview_chart = alt.Chart(data_sample).mark_circle(size=40).encode(
            x=alt.X("price:Q", title="Price ($)", scale=alt.Scale(domain=[0, 1200])),
            y=alt.Y("availability 365:Q", title="Availability (Days)"),
            color="cluster:N",
            tooltip=["price", "availability 365", "number of reviews"]
        ).properties(height=400)
        st.altair_chart(overview_chart, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Developed for Airbnb NYC Market Analysis | Data-Driven Decisions")