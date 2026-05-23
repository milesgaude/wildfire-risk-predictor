import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Load the saved model and encoders
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'wildfire_model.pkl'))
le_cause = joblib.load(os.path.join(BASE_DIR, 'le_cause.pkl'))
le_state = joblib.load(os.path.join(BASE_DIR, 'le_state.pkl'))

# App title and description
st.set_page_config(page_title="Wildfire Risk Predictor", page_icon="🔥", layout="wide")
st.title("🔥 Wildfire Risk Predictor")
st.markdown("Predict wildfire risk level based on location, time of year, and conditions using a Random Forest model trained on 1.88 million US wildfire records (1992-2015).")
st.markdown("---")

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location")
    state = st.selectbox("State", sorted(le_state.classes_))
    latitude = st.slider("Latitude", 24.0, 50.0, 37.0, 0.1)
    longitude = st.slider("Longitude", -125.0, -66.0, -100.0, 0.1)

with col2:
    st.subheader("📅 Conditions")
    month = st.slider("Month", 1, 12, 7)
    day_of_year = st.slider("Day of Year", 1, 365, 180)
    cause = st.selectbox("Fire Cause", sorted(le_cause.classes_))
    fire_year = st.slider("Year", 1992, 2015, 2010)

# Engineer features from inputs
is_fire_season = 1 if month in [6, 7, 8, 9, 10] else 0
is_western = 1 if longitude < -100 else 0

if latitude < 30:
    lat_band = 0
elif latitude < 37:
    lat_band = 1
elif latitude < 42:
    lat_band = 2
elif latitude < 50:
    lat_band = 3
else:
    lat_band = 4

# Encode categorical inputs
state_code = le_state.transform([state])[0]
cause_code = le_cause.transform([cause])[0]

# Create feature array
features = pd.DataFrame([[fire_year, day_of_year, latitude, longitude, 
                           cause_code, state_code, month, is_fire_season, 
                           is_western, lat_band]],
                        columns=['FIRE_YEAR', 'DISCOVERY_DOY', 'LATITUDE', 'LONGITUDE',
                                 'CAUSE_CODE', 'STATE_CODE', 'MONTH', 'IS_FIRE_SEASON',
                                 'IS_WESTERN_US', 'LAT_BAND'])

# Make prediction
st.markdown("---")

if st.button("🔥 Predict Wildfire Risk", use_container_width=True):
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_
    
    # Color based on risk
    colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Extreme': '🔴'}
    
    st.markdown(f"## {colors.get(prediction, '')} Predicted Risk Level: **{prediction}**")
    
    # Show confidence for each level
    st.subheader("Confidence Breakdown")
    prob_df = pd.DataFrame({
        'Risk Level': classes,
        'Confidence': probabilities
    }).sort_values('Confidence', ascending=True)
    
    fig = px.bar(prob_df, x='Confidence', y='Risk Level', orientation='h',
                 color='Risk Level',
                 color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 
                                     'High': '#e67e22', 'Extreme': '#e74c3c'})
    fig.update_layout(showlegend=False, height=300,
                      xaxis_title="Confidence", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    
    # Show the input summary
    st.subheader("Input Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("State", state)
        st.metric("Latitude", f"{latitude}°")
    with summary_col2:
        st.metric("Longitude", f"{longitude}°")
        st.metric("Month", month)
    with summary_col3:
        st.metric("Fire Cause", cause)
        st.metric("Fire Season", "Yes" if is_fire_season else "No")

# Footer
st.markdown("---")
st.markdown("**Model Info:** Random Forest Classifier | 100 trees | Trained on 500K samples from 1.88M US wildfire records")
st.markdown("**Data Source:** US Forest Service FPA-FOD (1992-2015)")
