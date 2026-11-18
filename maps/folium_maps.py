"""Folium map generation for observed GPS, road network, and routes."""
from pathlib import Path

import folium
import numpy as np
from scipy.spatial import cKDTree

from data_processing.loader import load_trip


def make_trip_map(trip_id: str, split: str = "train", predicted_arcs: list[int] | None = None) -> Path:
    trip = load_trip(trip_id, split)
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]

    center = [float(track["lat"].mean()), float(track["lon"].mean())]
    m = folium.Map(location=center, zoom_start=14)

    road_group = folium.FeatureGroup(name="Road network", show=False)
    longitude_scale = np.cos(np.radians(center[0]))
    gps_points = track[["lat", "lon"]].to_numpy(dtype=float)
    gps_points[:, 1] *= longitude_scale
    node_points = nodes[["lat", "lon"]].to_numpy(dtype=float)
    node_points[:, 1] *= longitude_scale
    nearby = cKDTree(gps_points).query(node_points, distance_upper_bound=0.0015)[0]
    arc_nodes = nodes.index.get_indexer(arcs[["u", "v"]].to_numpy().ravel()).reshape(-1, 2)
    valid_arcs = (arc_nodes >= 0).all(axis=1)
    local_arc_positions = np.flatnonzero(
        valid_arcs & (np.isfinite(nearby[arc_nodes.clip(min=0)]).any(axis=1))
    )
    if len(local_arc_positions) > 5000:
        local_arc_positions = local_arc_positions[
            np.linspace(0, len(local_arc_positions) - 1, 5000, dtype=int)
        ]
    for arc_position in local_arc_positions:
        start_id, end_id = arc_nodes[arc_position]
        start, end = nodes.iloc[start_id], nodes.iloc[end_id]
        folium.PolyLine(
            [[start["lat"], start["lon"]], [end["lat"], end["lon"]]],
            color="#9E9E9E",
            weight=2,
            opacity=0.45,
        ).add_to(road_group)
    road_group.add_to(m)

    gps_group = folium.FeatureGroup(name="GPS observations")
    for index, row in track.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=2.5,
            color="#1E88E5",
            fill=True,
            fill_color="#1E88E5",
            fill_opacity=0.8,
            tooltip=f"Observation {index}",
        ).add_to(gps_group)
    gps_group.add_to(m)

    def add_arc_route(name: str, arc_ids: list[int], color: str) -> None:
        group = folium.FeatureGroup(name=name)
        for arc_id in arc_ids:
            if 0 <= arc_id < len(arcs):
                u, v = arcs.iloc[arc_id]
                folium.PolyLine(
                    [[nodes.loc[u, "lat"], nodes.loc[u, "lon"]],
                     [nodes.loc[v, "lat"], nodes.loc[v, "lon"]]],
                    color=color,
                    weight=5,
                    opacity=0.85,
                ).add_to(group)
        group.add_to(m)

    add_arc_route("Reference route", route, "#2E7D32")
    if predicted_arcs is not None:
        add_arc_route("Predicted route", predicted_arcs, "#D81B60")

    folium.LayerControl(collapsed=False).add_to(m)

    out_dir = Path(__file__).resolve().parents[1] / "maps" / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{trip_id}.html"
    m.save(out)
    return out


if __name__ == "__main__":
    from data_processing.loader import list_trips

    trip_id = list_trips("train")[0]
    print(make_trip_map(trip_id))
