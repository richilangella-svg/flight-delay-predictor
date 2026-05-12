import streamlit as st
import pickle
import pandas as pd
import numpy as np
import calendar

AIRLINE_NAMES = {
    '9E': 'Endeavor Air', 'AA': 'American Airlines', 'AS': 'Alaska Airlines',
    'B6': 'JetBlue Airways', 'CO': 'Continental Airlines', 'DL': 'Delta Air Lines',
    'EV': 'ExpressJet Airlines', 'F9': 'Frontier Airlines', 'FL': 'AirTran Airways',
    'G4': 'Allegiant Air', 'HA': 'Hawaiian Airlines', 'MQ': 'Envoy Air',
    'NK': 'Spirit Airlines', 'NW': 'Northwest Airlines', 'OH': 'PSA Airlines',
    'OO': 'SkyWest Airlines', 'UA': 'United Airlines', 'US': 'US Airways',
    'VX': 'Virgin America', 'WN': 'Southwest Airlines', 'XE': 'ExpressJet',
    'YV': 'Mesa Airlines', 'YX': 'Republic Airways',
    'Air Wisconsin Airlines Corp': 'Air Wisconsin Airlines',
    'Alaska Airlines Inc.': 'Alaska Airlines',
    'Allegiant Air': 'Allegiant Air',
    'American Airlines Inc.': 'American Airlines',
    'Capital Cargo International': 'Capital Cargo International',
    'Comair Inc.': 'Comair',
    'Commutair Aka Champlain Enterprises, Inc.': 'CommutAir',
    'Compass Airlines': 'Compass Airlines',
    'Delta Air Lines Inc.': 'Delta Air Lines',
    'Empire Airlines Inc.': 'Empire Airlines',
    'Endeavor Air Inc.': 'Endeavor Air',
    'Envoy Air': 'Envoy Air',
    'ExpressJet Airlines Inc.': 'ExpressJet Airlines',
    'Frontier Airlines Inc.': 'Frontier Airlines',
    'GoJet Airlines, LLC d/b/a United Express': 'GoJet Airlines',
    'Hawaiian Airlines Inc.': 'Hawaiian Airlines',
    'Horizon Air': 'Horizon Air',
    'JetBlue Airways': 'JetBlue Airways',
    'Mesa Airlines Inc.': 'Mesa Airlines',
    'Peninsula Airways Inc.': 'Peninsula Airways',
    'Republic Airlines': 'Republic Airlines',
    'SkyWest Airlines Inc.': 'SkyWest Airlines',
    'Southwest Airlines Co.': 'Southwest Airlines',
    'Spirit Air Lines': 'Spirit Airlines',
    'Trans States Airlines': 'Trans States Airlines',
    'United Air Lines Inc.': 'United Airlines',
}

with open('flight_delay_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
model_cancel = model_data['model_cancel']
features = model_data['features']
airline_map = model_data['airline_map']
origin_map = model_data['origin_map']
dest_map = model_data['dest_map']
route_map = model_data['route_map']
airline_delay = model_data['airline_delay']
route_delay = model_data['route_delay']
hour_delay = model_data['hour_delay']
month_delay = model_data['month_delay']

seen_names = {}
for code in sorted(airline_map.keys()):
    display = AIRLINE_NAMES.get(code, code)
    if display not in seen_names:
        seen_names[display] = code

airline_display = {code: name for name, code in seen_names.items()}
airline_options = sorted(airline_display.keys(), key=lambda x: airline_display[x])

st.title("Flight Delay & Cancellation Predictor")
st.markdown("Predict whether your flight will be delayed or cancelled based on historical US flight data (2009-2022)")

st.sidebar.header("Flight Details")
airline_code = st.sidebar.selectbox("Airline", airline_options, format_func=lambda x: airline_display[x])
origin = st.sidebar.selectbox("Origin Airport", sorted(origin_map.keys()))
dest = st.sidebar.selectbox("Destination Airport", sorted(dest_map.keys()))
month = st.sidebar.slider("Month", 1, 12, 7)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
hour = st.sidebar.slider("Departure Hour", 0, 23, 8)

month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

max_days = calendar.monthrange(2024, month)[1]
date_valid = day <= max_days

if st.sidebar.button("Predict", type="primary"):
    if origin == dest:
        st.error("Invalid: Origin and destination airports cannot be the same.")
    elif not date_valid:
        st.error(f"Invalid date: {month_names[month]} does not have {day} days.")
    else:
        route = f"{origin}_{dest}"
        route_known = route in route_map

        input_data = pd.DataFrame([{
            'Month': month, 'DayofMonth': day, 'DepHour': hour, 'Distance': 800,
            'Airline_code': airline_map.get(airline_code, 0),
            'Origin_code': origin_map.get(origin, 0),
            'Dest_code': dest_map.get(dest, 0),
            'Route_code': route_map.get(route, 0),
            'Airline_delay_rate': airline_delay.get(airline_code, 0.17),
            'Route_delay_rate': route_delay.get(route, 0.17),
            'Hour_delay_rate': hour_delay.get(hour, 0.17),
            'Month_delay_rate': month_delay.get(month, 0.17),
        }])

        delay_prob = model.predict_proba(input_data)[0][1]
        cancel_prob = model_cancel.predict_proba(input_data)[0][1]

        airline_name = airline_display[airline_code]
        st.subheader(f"Results for {airline_name}: {origin} to {dest}")
        st.write(f"{month_names[month]} {day}, departing at {hour}:00")

        if not route_known:
            st.warning("This route is not present in the historical data. Prediction is based on general averages and may be less accurate.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Delay Probability", f"{delay_prob*100:.1f}%")
            if delay_prob > 0.5:
                st.error("High risk of delay (>15 min)")
            elif delay_prob > 0.3:
                st.warning("Moderate risk of delay")
            else:
                st.success("Low risk of delay")
        with col2:
            st.metric("Cancellation Probability", f"{cancel_prob*100:.1f}%")
            if cancel_prob > 0.1:
                st.error("High risk of cancellation")
            elif cancel_prob > 0.05:
                st.warning("Moderate risk of cancellation")
            else:
                st.success("Low risk of cancellation")

        st.subheader("Historical Airlines' Statistics")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Airline avg delay rate", f"{airline_delay.get(airline_code, 0)*100:.1f}%")
        with col4:
            rate = route_delay.get(route, None)
            if rate is not None:
                st.metric("Route avg delay rate", f"{rate*100:.1f}%")
            else:
                st.metric("Route avg delay rate", "N/A")
        with col5:
            st.metric("Hour avg delay rate", f"{hour_delay.get(hour, 0)*100:.1f}%")
