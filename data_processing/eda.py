"""Dataset inventory and exploratory visualisations."""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data_processing.loader import list_trips, load_trip


def haversine_m(lon1: pd.Series, lat1: pd.Series, lon2: pd.Series, lat2: pd.Series) -> pd.Series:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000 * 2 * np.arcsin(np.sqrt(a))


def summarize_trip(trip_id: str, split: str) -> dict:
    trip = load_trip(trip_id, split)
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]

    lon = pd.to_numeric(track["lon"], errors="coerce")
    lat = pd.to_numeric(track["lat"], errors="coerce")
    timestamps = pd.to_numeric(track["t"], errors="coerce")
    dt = timestamps.sort_values().diff()

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes.index)
    graph.add_edges_from(arcs.itertuples(index=False, name=None))
    weak = nx.DiGraph.to_undirected(graph)
    isolated = list(nx.isolates(graph))

    route_length_m = np.nan
    if "route" in trip and len(trip["route"]) and len(arcs):
        route_arcs = arcs.iloc[[i for i in trip["route"] if 0 <= i < len(arcs)]]
        if not route_arcs.empty:
            distances = []
            for u, v in route_arcs.itertuples(index=False, name=None):
                if u in nodes.index and v in nodes.index:
                    distances.append(
                        haversine_m(
                            pd.Series(nodes.loc[u, "lon"]),
                            pd.Series(nodes.loc[u, "lat"]),
                            pd.Series(nodes.loc[v, "lon"]),
                            pd.Series(nodes.loc[v, "lat"]),
                        ).iloc[0]
                    )
            route_length_m = float(np.sum(distances))

    valid_gps = lon.between(-180, 180) & lat.between(-90, 90)
    return {
        "partition": split,
        "trip_id": trip_id,
        "observations": len(track),
        "missing_lon": int(lon.isna().sum()),
        "missing_lat": int(lat.isna().sum()),
        "missing_t": int(timestamps.isna().sum()),
        "invalid_gps": int((~valid_gps).sum()),
        "lon_min": float(lon.min()) if lon.notna().any() else np.nan,
        "lon_max": float(lon.max()) if lon.notna().any() else np.nan,
        "lat_min": float(lat.min()) if lat.notna().any() else np.nan,
        "lat_max": float(lat.max()) if lat.notna().any() else np.nan,
        "median_dt_s": float(dt.median()) if dt.notna().any() else np.nan,
        "gap_gt_30s": int((dt > 30).sum()),
        "nodes": len(nodes),
        "arcs": len(arcs),
        "weak_components": nx.number_weakly_connected_components(graph),
        "isolated_nodes": len(isolated),
        "route_length_m": route_length_m,
    }


def run_eda(splits=("train", "student_test")) -> tuple[pd.DataFrame, pd.DataFrame]:
    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inventory_rows = []
    gap_rows = []
    for split in splits:
        for trip_id in list_trips(split):
            trip = load_trip(trip_id, split)
            inventory_rows.append(summarize_trip(trip_id, split))
            dt = pd.to_numeric(trip["track"]["t"], errors="coerce").sort_values().diff()
            gap_rows.extend(
                {"partition": split, "trip_id": trip_id, "dt_s": value}
                for value in dt.dropna().tolist()
            )

    inventory = pd.DataFrame(inventory_rows)
    gaps = pd.DataFrame(gap_rows)
    inventory.to_csv(results_dir / "eda_inventory.csv", index=False)
    gaps.to_csv(results_dir / "eda_time_gaps.csv", index=False)

    if not gaps.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for split, frame in gaps.groupby("partition"):
            ax.hist(frame["dt_s"].clip(upper=300), bins=50, alpha=0.55, label=split)
        ax.set_xlabel("Time gap (seconds, clipped at 300)")
        ax.set_ylabel("Number of GPS intervals")
        ax.set_title("Distribution of GPS time gaps")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "eda_time_gaps.png", dpi=150)
        plt.close(fig)

    if not inventory.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for split, frame in inventory.groupby("partition"):
            ax.hist(frame["observations"], bins=20, alpha=0.55, label=split)
        ax.set_xlabel("GPS observations per trip")
        ax.set_ylabel("Number of trips")
        ax.set_title("Trip length inventory")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures_dir / "eda_trip_lengths.png", dpi=150)
        plt.close(fig)

    return inventory, gaps


if __name__ == "__main__":
    inventory, gaps = run_eda()
    print(inventory.groupby("partition").agg(
        trips=("trip_id", "count"),
        observations=("observations", "sum"),
        median_trip_length=("observations", "median"),
        missing_gps=("invalid_gps", "sum"),
    ).to_string())
