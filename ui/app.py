"""Evaluator UI, runs on Laptop 2 (Owner: D).
Run:  set PATHSNAP_API=http://<laptop1-ip>:8000   then   streamlit run ui/app.py
"""
import os

import requests
import streamlit as st

API = os.environ.get("PATHSNAP_API", "http://localhost:8000")

st.title("PathSnap")
st.caption(f"API: {API}")

try:
    trips = requests.get(f"{API}/trips", timeout=5).json()["trips"]
except Exception as e:
    st.error(f"Cannot reach API: {e}")
    st.stop()

trip = st.selectbox("Evaluation trip", trips)
if st.button("Run") and trip:
    r = requests.post(f"{API}/predict/{trip}", timeout=120)
    st.write(r.json())
    # TODO: folium map (streamlit_folium) with GPS + predicted route, metrics panel
