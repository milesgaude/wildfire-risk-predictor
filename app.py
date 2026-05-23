import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import json
from urllib.request import urlopen

# Load the saved model and encoders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'wildfire_model.pkl'))
le_cause = joblib.load(os.path.join(BASE_DIR, 'le_cause.pkl'))
le_state = joblib.load(os.path.join(BASE_DIR, 'le_state.pkl'))

# State coordinates for default values
state_coords = {
    'AL': (32.8, -86.8), 'AK': (64.0, -153.0), 'AZ': (34.3, -111.7),
    'AR': (34.8, -92.2), 'CA': (37.2, -119.5), 'CO': (39.0, -105.5),
    'CT': (41.6, -72.7), 'DE': (39.0, -75.5), 'FL': (28.6, -82.4),
    'GA': (32.7, -83.4), 'HI': (20.5, -157.4), 'ID': (44.4, -114.6),
    'IL': (40.0, -89.2), 'IN': (39.9, -86.3), 'IA': (42.0, -93.5),
    'KS': (38.5, -98.3), 'KY': (37.8, -85.7), 'LA': (31.0, -92.0),
    'ME': (45.4, -69.2), 'MD': (39.0, -76.7), 'MA': (42.3, -71.8),
    'MI': (44.3, -84.6), 'MN': (46.3, -94.3), 'MS': (32.7, -89.7),
    'MO': (38.4, -92.5), 'MT': (47.1, -109.6), 'NE': (41.5, -99.8),
    'NV': (39.3, -116.6), 'NH': (43.7, -71.6), 'NJ': (40.1, -74.7),
    'NM': (34.4, -106.1), 'NY': (42.9, -75.5), 'NC': (35.6, -79.4),
    'ND': (47.4, -100.5), 'OH': (40.4, -82.8), 'OK': (35.6, -97.5),
    'OR': (43.9, -120.6), 'PA': (40.9, -77.8), 'PR': (18.2, -66.5),
    'RI': (41.7, -71.5), 'SC': (33.9, -80.9), 'SD': (44.4, -100.2),
    'TN': (35.9, -86.4), 'TX': (31.5, -99.3), 'UT': (39.3, -111.7),
    'VT': (44.1, -72.6), 'VA': (37.5, -78.9), 'WA': (47.4, -120.7),
    'WV': (38.6, -80.6), 'WI': (44.6, -89.7), 'WY': (43.0, -107.6),
    'DC': (38.9, -77.0)
}

month_names = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}

# Function to look up coordinates from city/zip
def lookup_location(query):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={query}&countrycodes=us&format=json&limit=1"
        req = urlopen(url)
        data = json.loads(req.read().decode())
        if data:
            return float(data[0]['lat']), float(data[0]['lon']), data[0].get('display_name', '')
    except:
        pass
    return None, None, None

# App setup
st.set_page_config(page_title="Wildfire Risk Predictor", page_icon="🔥", layout="wide")
st.title("🔥 Wildfire Risk Predictor")
st.markdown("Predict wildfire risk based on location, time of year, and conditions. Powered by a Random Forest model trained on **1.88 million** US wildfire records.")
st.markdown("---")

# Input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Location")
    state = st.selectbox("State", sorted(le_state.classes_))
    default_lat, default_lon = state_coords.get(state, (37.0, -100.0))
    
    # City/zip lookup
    location_input = st.text_input("🔍 Search by city or zip code", placeholder="e.g. Atlanta, GA or 30301")
    
    if location_input:
        found_lat, found_lon, found_name = lookup_location(location_input)
        if found_lat:
            default_lat = found_lat
            default_lon = found_lon
            st.success(f"📍 Found: {found_name}")
        else:
            st.error("Location not found — try a different search")
    
    latitude = st.slider("Latitude", 24.0, 50.0, min(max(default_lat, 24.0), 50.0), 0.1)
    longitude = st.slider("Longitude", -125.0, -66.0, min(max(default_lon, -125.0), -66.0), 0.1)
    st.caption("Coordinates auto-fill from state or search — adjust sliders for precision")

with col2:
    st.subheader("📅 Conditions")
    month_name = st.selectbox("Month", list(month_names.keys()), index=6)
    month = month_names[month_name]
    
    cause = st.selectbox("Likely Fire Cause", sorted(le_cause.classes_))
    
    if month in [6, 7, 8, 9, 10]:
        st.warning("⚠️ This month falls within peak fire season (June–October)")
    else:
        st.info("ℹ️ This month is outside peak fire season")

# Calculate derived features
day_of_year = int((month - 1) * 30.44 + 15)
fire_year = 2015
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

state_code = le_state.transform([state])[0]
cause_code = le_cause.transform([cause])[0]

features = pd.DataFrame([[fire_year, day_of_year, latitude, longitude,
                           cause_code, state_code, month, is_fire_season,
                           is_western, lat_band]],
                        columns=['FIRE_YEAR', 'DISCOVERY_DOY', 'LATITUDE', 'LONGITUDE',
                                 'CAUSE_CODE', 'STATE_CODE', 'MONTH', 'IS_FIRE_SEASON',
                                 'IS_WESTERN_US', 'LAT_BAND'])

# Predict
st.markdown("---")

if st.button("🔥 Predict Wildfire Risk", use_container_width=True):
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    risk_colors = {
        'Low': ('#2ecc71', '🟢'),
        'Medium': ('#f39c12', '🟡'),
        'High': ('#e67e22', '🟠'),
        'Extreme': ('#e74c3c', '🔴')
    }

    color, icon = risk_colors.get(prediction, ('#95a5a6', '⚪'))

    result_col1, result_col2 = st.columns([1, 2])

    with result_col1:
        st.markdown(f"### {icon} {prediction} Risk")
        confidence = max(probabilities) * 100
        st.markdown(f"**Confidence: {confidence:.0f}%**")

    with result_col2:
        prob_df = pd.DataFrame({
            'Risk Level': classes,
            'Confidence': [round(p * 100, 1) for p in probabilities]
        }).sort_values('Confidence', ascending=True)

        fig = px.bar(prob_df, x='Confidence', y='Risk Level', orientation='h',
                     color='Risk Level',
                     color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12',
                                         'High': '#e67e22', 'Extreme': '#e74c3c'})
        fig.update_layout(showlegend=False, height=250,
                          xaxis_title="Confidence (%)", yaxis_title="",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ What does this mean?"):
        st.markdown("""
        **Risk Levels:**
        - 🟢 **Low** — Minimal fire risk. Fires in this category are typically under 0.25 acres.
        - 🟡 **Medium** — Moderate risk. Fires may reach up to 100 acres under these conditions.
        - 🟠 **High** — Elevated risk. Fires between 100–999 acres are more likely.
        - 🔴 **Extreme** — Severe risk. Conditions favor fires exceeding 1,000 acres.
        
        **Note:** This model uses historical patterns from location, seasonality, and cause data. 
        Real-time weather conditions (temperature, humidity, wind speed) are the strongest predictors 
        of wildfire risk but are not included in this version.
        """)

# Footer
st.markdown("---")
st.caption("Random Forest Classifier trained on 1.88M US wildfire records (1992–2015) from the US Forest Service FPA-FOD dataset.")
