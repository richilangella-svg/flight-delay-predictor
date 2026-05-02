
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
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

# App title
st.title("✈️ Flight Delay & Cancellation Predictor")
st.markdown("Predict whether your flight will be delayed or cancelled based on historical US flight data (2009-2022)")

# Input form
st.sidebar.header("Flight Details")

airline = st.sidebar.selectbox("Airline", sorted(airline_map.keys()))
origin = st.sidebar.selectbox("Origin Airport", sorted(origin_map.keys()))
dest = st.sidebar.selectbox("Destination Airport", sorted(dest_map.keys()))
month = st.sidebar.slider("Month", 1, 12, 7)
day = st.sidebar.slider("Day of Month", 1, 31, 15)
hour = st.sidebar.slider("Departure Hour", 0, 23, 8)
distance = st.sidebar.number_input("Distance (miles)", min_value=50, max_value=5000, value=500)

month_names = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}

if st.sidebar.button("Predict", type="primary"):
    route = f"{origin}_{dest}"

    input_data = pd.DataFrame([{
        'Month': month,
        'DayofMonth': day,
        'DepHour': hour,
        'Distance': distance,
        'Airline_code': airline_map.get(airline, 0),
        'Origin_code': origin_map.get(origin, 0),
        'Dest_code': dest_map.get(dest, 0),
        'Route_code': route_map.get(route, 0),
        'Airline_delay_rate': airline_delay.get(airline, 0.17),
        'Route_delay_rate': route_delay.get(route, 0.17),
        'Hour_delay_rate': hour_delay.get(hour, 0.17),
        'Month_delay_rate': month_delay.get(month, 0.17),
    }])

    delay_prob = model.predict_proba(input_data)[0][1]
    cancel_prob = model_cancel.predict_proba(input_data)[0][1]

    st.subheader(f"Results for {airline}: {origin} → {dest}")
    st.write(f"📅 {month_names[month]} {day}, departing at {hour}:00")

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

    # Historical stats
    st.subheader("📊 Historical Statistics")
    col3, col4, col5 = st.columns(3)
    with col3:
        st.metric("Airline avg delay rate", f"{airline_delay.get(airline, 0)*100:.1f}%")
    with col4:
        st.metric("Route avg delay rate", f"{route_delay.get(route, 0)*100:.1f}%")
    with col5:
        st.metric("Hour avg delay rate", f"{hour_delay.get(hour, 0)*100:.1f}%")
