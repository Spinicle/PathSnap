"""Streamlit evaluator interface for the PathSnap FastAPI backend."""

from __future__ import annotations

import os
from typing import Any

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium


API_BASE = os.environ.get("PATHSNAP_API", "http://127.0.0.1:8000").rstrip("/")
TRIPS_TIMEOUT_SECONDS = 5
PREDICT_TIMEOUT_SECONDS = 120


@st.cache_data(ttl=300)
def fetch_trips() -> list[str]:
    """Fetch available evaluator trip IDs from the contract-defined endpoint."""
    response = requests.get(
        f"{API_BASE}/trips",
        timeout=TRIPS_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not isinstance(payload.get("trips"), list):
        raise ValueError("The /trips response is missing a valid 'trips' list.")
    trips = payload["trips"]
    if not all(isinstance(trip_id, str) and trip_id for trip_id in trips):
        raise ValueError("The /trips response contains an invalid trip ID.")
    return trips


def fetch_prediction(trip_id: str) -> dict[str, Any]:
    """Fetch and validate the contract-defined prediction response."""
    response = requests.post(
        f"{API_BASE}/predict/{trip_id}",
        timeout=PREDICT_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("The prediction response is not a JSON object.")

    required_keys = {"trip_id", "gps", "predicted_arcs", "predicted_path", "metrics", "inference_ms"}
    missing = sorted(required_keys.difference(payload))
    if missing:
        raise ValueError(f"The prediction response is missing fields: {', '.join(missing)}.")

    gps = payload["gps"]
    predicted_path = payload["predicted_path"]
    metrics = payload["metrics"]
    if not isinstance(gps, list) or not isinstance(predicted_path, list):
        raise ValueError("gps and predicted_path must be coordinate lists.")
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be a JSON object.")

    expected_metrics = {
        "top1_accuracy",
        "route_continuity",
        "classifier_only_top1",
    }
    missing_metrics = sorted(expected_metrics.difference(metrics))
    if missing_metrics:
        raise ValueError(
            f"The prediction metrics are missing fields: {', '.join(missing_metrics)}."
        )
    return payload


def validate_coordinates(
    coordinates: list[Any],
    field_name: str,
) -> list[list[float]]:
    """Return finite, two-component [latitude, longitude] coordinates."""
    valid: list[list[float]] = []
    for coordinate in coordinates:
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
            raise ValueError(f"{field_name} contains a coordinate that is not [lat, lon].")
        try:
            latitude = float(coordinate[0])
            longitude = float(coordinate[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} contains a non-numeric coordinate.") from exc
        if (
            not (-90.0 <= latitude <= 90.0)
            or not (-180.0 <= longitude <= 180.0)
        ):
            raise ValueError(f"{field_name} contains coordinates outside geographic bounds.")
        valid.append([latitude, longitude])
    return valid


def percentage(value: Any, label: str) -> str:
    """Format an API percentage value without recomputing or replacing it."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not numeric.") from None
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{label} must be a fraction between 0 and 1.")
    return f"{numeric_value * 100:.2f}%"


def build_map(
    gps_coordinates: list[list[float]],
    predicted_path: list[list[float]],
) -> folium.Map:
    """Build a GPS/predicted-route Folium map with toggleable layers."""
    all_coordinates = gps_coordinates + predicted_path
    if all_coordinates:
        latitude = sum(point[0] for point in all_coordinates) / len(all_coordinates)
        longitude = sum(point[1] for point in all_coordinates) / len(all_coordinates)
        map_object = folium.Map(location=[latitude, longitude], zoom_start=14)
    else:
        map_object = folium.Map(location=[0.0, 0.0], zoom_start=2)

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for index, coordinate in enumerate(gps_coordinates, start=1):
        folium.CircleMarker(
            location=coordinate,
            radius=3,
            color="#1E88E5",
            fill=True,
            fill_color="#1E88E5",
            fill_opacity=0.85,
            tooltip=f"GPS observation {index}",
        ).add_to(gps_layer)
    if gps_coordinates:
        folium.PolyLine(
            gps_coordinates,
            color="#1E88E5",
            weight=3,
            opacity=0.75,
        ).add_to(gps_layer)
    gps_layer.add_to(map_object)

    predicted_layer = folium.FeatureGroup(name="Predicted route", show=True)
    if predicted_path:
        folium.PolyLine(
            predicted_path,
            color="#D32F2F",
            weight=5,
            opacity=0.90,
        ).add_to(predicted_layer)
    predicted_layer.add_to(map_object)

    legend = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: white; padding: 8px 12px; border-radius: 6px;
                border: 1px solid #bbb; box-shadow: 0 1px 4px rgba(0,0,0,.2);">
      <div><span style="color:#1E88E5;">&#9679;</span> GPS observations</div>
      <div><span style="color:#D32F2F;">&#8212;</span> Predicted route</div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(map_object)
    return map_object


def display_api_error(error: Exception) -> None:
    """Display concise, safe API failures for connection and HTTP errors."""
    if isinstance(error, requests.exceptions.Timeout):
        st.error("The API request timed out. Please check that the backend is running.")
    elif isinstance(error, requests.exceptions.ConnectionError):
        st.error("Cannot connect to the PathSnap API. Please verify the backend and PATHSNAP_API.")
    elif isinstance(error, requests.exceptions.HTTPError):
        response = getattr(error, "response", None)
        status = response.status_code if response is not None else "unknown"
        st.error(f"The API returned an HTTP error (status {status}).")
    elif isinstance(error, (ValueError, requests.exceptions.JSONDecodeError)):
        st.error(f"The API returned a malformed response: {error}")
    else:
        st.error(f"The API request failed: {error}")


def main() -> None:
    """Render the Streamlit page and handle API interactions."""
    st.set_page_config(
        page_title="PathSnap – GPS Map Matching",
        page_icon="🗺️",
        layout="wide",
    )

    st.title("PathSnap – GPS Map Matching")
    st.write(
        "Select an evaluation trip and run the existing map-matching API to visualize "
        "GPS observations and the predicted route."
    )

    try:
        with st.spinner("Loading available trips..."):
            trips = fetch_trips()
    except Exception as error:
        display_api_error(error)
        return

    if not trips:
        st.warning("No evaluator trips are currently available.")
        return

    selected_trip = st.selectbox("Available trip", trips)
    st.caption(f"Selected trip ID: `{selected_trip}`")

    if st.button("Run Map Matching", type="primary", use_container_width=True):
        try:
            with st.spinner("Running map matching..."):
                prediction = fetch_prediction(selected_trip)

            gps = validate_coordinates(prediction["gps"], "gps")
            predicted_path = validate_coordinates(
                prediction["predicted_path"],
                "predicted_path",
            )

            metrics = prediction["metrics"]
            top1_accuracy = percentage(
                metrics["top1_accuracy"],
                "metrics.top1_accuracy",
            )
            classifier_top1 = percentage(
                metrics["classifier_only_top1"],
                "metrics.classifier_only_top1",
            )
            route_continuity = percentage(
                metrics["route_continuity"],
                "metrics.route_continuity",
            )

            try:
                inference_ms = float(prediction["inference_ms"])
            except (TypeError, ValueError):
                raise ValueError("inference_ms must be numeric.") from None

            st.subheader("Prediction Metrics")
            metric_columns = st.columns(4)
            metric_columns[0].metric("HMM Top-1 Accuracy", top1_accuracy)
            metric_columns[1].metric("Classifier-only Top-1 Accuracy", classifier_top1)
            metric_columns[2].metric("Route Continuity", route_continuity)
            metric_columns[3].metric("Inference Time", f"{inference_ms:.2f} ms")

            st.subheader("Map")
            map_object = build_map(gps, predicted_path)
            st_folium(map_object, width=None, height=520, returned_objects=[])

            if not predicted_path:
                st.info("No predicted route coordinates were returned by the API.")
            if not gps:
                st.info("No GPS observation coordinates were returned by the API.")
        except Exception as error:
            display_api_error(error)


if __name__ == "__main__":
    main()
