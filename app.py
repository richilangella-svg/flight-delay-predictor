import streamlit as st
import pickle
import pandas as pd
import numpy as np
import calendar

# Maps every key that may appear in airline_map → canonical display name.

# For duplicates, all variants point to the same name.

# The key we keep in the selector is the FIRST one listed here (preferred key).

CANONICAL_AIRLINE = {
# Alaska Airlines
‘AS’:                       ‘Alaska Airlines’,
‘Alaska Airlines Inc.’:     ‘Alaska Airlines’,
# Allegiant Air
‘G4’:                       ‘Allegiant Air’,
‘Allegiant Air’:            ‘Allegiant Air’,
# American Airlines
‘AA’:                       ‘American Airlines’,
‘American Airlines Inc.’:   ‘American Airlines’,
# AirTran Airways
‘FL’:                       ‘AirTran Airways’,
# Capital Cargo International
‘Capital Cargo International’: ‘Capital Cargo International’,
# Comair
‘Comair Inc.’:              ‘Comair’,
# CommutAir
‘Commutair Aka Champlain Enterprises, Inc.’: ‘CommutAir’,
# Compass Airlines
‘Compass Airlines’:         ‘Compass Airlines’,
# Continental Airlines
‘CO’:                       ‘Continental Airlines’,
# Delta Air Lines
‘DL’:                       ‘Delta Air Lines’,
‘Delta Air Lines Inc.’:     ‘Delta Air Lines’,
# Empire Airlines
‘Empire Airlines Inc.’:     ‘Empire Airlines’,
# Endeavor Air
‘9E’:                       ‘Endeavor Air’,
‘Endeavor Air Inc.’:        ‘Endeavor Air’,
# Envoy Air
‘MQ’:                       ‘Envoy Air’,
‘Envoy Air’:                ‘Envoy Air’,
# ExpressJet Airlines
‘EV’:                       ‘ExpressJet Airlines’,
‘XE’:                       ‘ExpressJet Airlines’,
‘ExpressJet Airlines Inc.’: ‘ExpressJet Airlines’,
# Frontier Airlines
‘F9’:                       ‘Frontier Airlines’,
‘Frontier Airlines Inc.’:   ‘Frontier Airlines’,
# GoJet Airlines
‘GoJet Airlines, LLC d/b/a United Express’: ‘GoJet Airlines’,
# Hawaiian Airlines
‘HA’:                       ‘Hawaiian Airlines’,
‘Hawaiian Airlines Inc.’:   ‘Hawaiian Airlines’,
# Horizon Air
‘Horizon Air’:              ‘Horizon Air’,
# JetBlue Airways
‘B6’:                       ‘JetBlue Airways’,
‘JetBlue Airways’:          ‘JetBlue Airways’,
# Mesa Airlines
‘YV’:                       ‘Mesa Airlines’,
‘Mesa Airlines Inc.’:       ‘Mesa Airlines’,
# Northwest Airlines
‘NW’:                       ‘Northwest Airlines’,
# Peninsula Airways
‘Peninsula Airways Inc.’:   ‘Peninsula Airways’,
# PSA Airlines
‘OH’:                       ‘PSA Airlines’,
# Republic Airways
‘YX’:                       ‘Republic Airways’,
‘Republic Airlines’:        ‘Republic Airways’,
# SkyWest Airlines
‘OO’:                       ‘SkyWest Airlines’,
‘SkyWest Airlines Inc.’:    ‘SkyWest Airlines’,
# Southwest Airlines
‘WN’:                       ‘Southwest Airlines’,
‘Southwest Airlines Co.’:   ‘Southwest Airlines’,
# Spirit Airlines
‘NK’:                       ‘Spirit Airlines’,
‘Spirit Air Lines’:         ‘Spirit Airlines’,
# Trans States Airlines
‘Trans States Airlines’:    ‘Trans States Airlines’,
# United Airlines
‘UA’:                       ‘United Airlines’,
‘United Air Lines Inc.’:    ‘United Airlines’,
# US Airways
‘US’:                       ‘US Airways’,
# Virgin America
‘VX’:                       ‘Virgin America’,
# Air Wisconsin
‘Air Wisconsin Airlines Corp’: ‘Air Wisconsin Airlines’,
}

# Preferred key order: short IATA codes first, then full names

PREFERRED_KEY_ORDER = [
‘AS’, ‘G4’, ‘AA’, ‘FL’, ‘CO’, ‘DL’, ‘9E’, ‘MQ’, ‘EV’, ‘F9’,
‘HA’, ‘B6’, ‘YV’, ‘NW’, ‘OH’, ‘YX’, ‘OO’, ‘WN’, ‘NK’, ‘UA’,
‘US’, ‘VX’, ‘XE’,
‘Capital Cargo International’, ‘Comair Inc.’, ‘Commutair Aka Champlain Enterprises, Inc.’,
‘Compass Airlines’, ‘Empire Airlines Inc.’, ‘GoJet Airlines, LLC d/b/a United Express’,
‘Horizon Air’, ‘Peninsula Airways Inc.’, ‘Republic Airlines’,
‘Trans States Airlines’, ‘Air Wisconsin Airlines Corp’,
]

# Load model

with open(‘flight_delay_model.pkl’, ‘rb’) as f:
model_data = pickle.load(f)

model         = model_data[‘model’]
model_cancel  = model_data[‘model_cancel’]
features      = model_data[‘features’]
airline_map   = model_data[‘airline_map’]
origin_map    = model_data[‘origin_map’]
dest_map      = model_data[‘dest_map’]
route_map     = model_data[‘route_map’]
airline_delay = model_data[‘airline_delay’]
route_delay   = model_data[‘route_delay’]
hour_delay    = model_data[‘hour_delay’]
month_delay   = model_data[‘month_delay’]

# Build deduplicated selector:

# For each canonical name, pick the first key (from PREFERRED_KEY_ORDER) that

# actually exists in airline_map. That key is used for all model lookups.

seen_canonical = {}   # canonical_name → preferred key
for key in PREFERRED_KEY_ORDER:
if key not in airline_map:
continue
canonical = CANONICAL_AIRLINE.get(key, key)
if canonical not in seen_canonical:
seen_canonical[canonical] = key

# Also catch any keys in airline_map not covered above

for key in airline_map.keys():
canonical = CANONICAL_AIRLINE.get(key, key)
if canonical not in seen_canonical:
seen_canonical[canonical] = key

# Final selector: display name → model key

display_to_key = {name: key for name, key in sorted(seen_canonical.items())}
airline_options = sorted(display_to_key.keys())  # alphabetical by display name

# App title

st.title(“✈️ Flight Delay & Cancellation Predictor”)
st.markdown(“Predict whether your flight will be delayed or cancelled based on historical US flight data (2009-2022)”)

# Input form

st.sidebar.header(“Flight Details”)

airline_name_selected = st.sidebar.selectbox(
“Airline”,
airline_options,
)
airline_code = display_to_key[airline_name_selected]

origin = st.sidebar.selectbox(“Origin Airport”, sorted(origin_map.keys()))
dest   = st.sidebar.selectbox(“Destination Airport”, sorted(dest_map.keys()))

month = st.sidebar.slider(“Month”, 1, 12, 7)
day   = st.sidebar.slider(“Day of Month”, 1, 31, 15)
hour  = st.sidebar.slider(“Departure Hour”, 0, 23, 8)

month_names = {
1:“January”, 2:“February”, 3:“March”, 4:“April”,
5:“May”, 6:“June”, 7:“July”, 8:“August”,
9:“September”, 10:“October”, 11:“November”, 12:“December”
}

max_days   = calendar.monthrange(2024, month)[1]
date_valid = day <= max_days

if st.sidebar.button(“Predict”, type=“primary”):

```
# Fix 2: same origin and destination
if origin == dest:
    st.error("❌ Origin and destination airports cannot be the same. Please select different airports.")

# Fix: invalid date
elif not date_valid:
    st.error(f"❌ Invalid date: {month_names[month]} does not have {day} days. Please select a valid date.")

else:
    route = f"{origin}_{dest}"

    # Fix 5: route not in historical data
    route_exists = route in route_map

    # Use median distance from route if available, else a neutral fallback
    # (distance removed from user input but still needed as feature)
    # We pass 0 so the model uses it neutrally; real impact comes from route/airline encodings
    distance_fallback = 500

    input_data = pd.DataFrame([{
        'Month':             month,
        'DayofMonth':        day,
        'DepHour':           hour,
        'Distance':          distance_fallback,
        'Airline_code':      airline_map.get(airline_code, 0),
        'Origin_code':       origin_map.get(origin, 0),
        'Dest_code':         dest_map.get(dest, 0),
        'Route_code':        route_map.get(route, 0),
        'Airline_delay_rate': airline_delay.get(airline_code, 0.17),
        'Route_delay_rate':   route_delay.get(route, 0.17),
        'Hour_delay_rate':    hour_delay.get(hour, 0.17),
        'Month_delay_rate':   month_delay.get(month, 0.17),
    }])

    delay_prob  = model.predict_proba(input_data)[0][1]
    cancel_prob = model_cancel.predict_proba(input_data)[0][1]

    st.subheader(f"Results for {airline_name_selected}: {origin} → {dest}")
    st.write(f"📅 {month_names[month]} {day}, departing at {hour}:00")

    # Fix 5: warning if route not in historical data
    if not route_exists:
        st.warning("⚠️ This route is not present in the historical data. The prediction is based on general averages and may be less accurate.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Delay Probability", f"{delay_prob*100:.1f}%")
        if delay_prob > 0.5:
            st.error("⚠️ High risk of delay (>15 min)")
        elif delay_prob > 0.3:
            st.warning("🟡 Moderate risk of delay")
        else:
            st.success("✅ Low risk of delay")

    with col2:
        st.metric("Cancellation Probability", f"{cancel_prob*100:.1f}%")
        if cancel_prob > 0.1:
            st.error("⚠️ High risk of cancellation")
        elif cancel_prob > 0.05:
            st.warning("🟡 Moderate risk of cancellation")
        else:
            st.success("✅ Low risk of cancellation")

    # Fix 3 + title rename: "Historical Airlines' Statistics"
    st.subheader("📊 Historical Airlines' Statistics")

    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric("Airline avg delay rate", f"{airline_delay.get(airline_code, 0)*100:.1f}%")
    with col4:
        st.metric("Route avg delay rate", f"{route_delay.get(route, 0)*100:.1f}%")
    with col5:
        st.metric("Hour avg delay rate", f"{hour_delay.get(hour, 0)*100:.1f}%")
```
