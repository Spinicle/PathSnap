"""Raw file readers. Formats (from Dataset Exploration.pdf, verify yourselves):
  .track  lon lat t        (whitespace, no header)
  .nodes  lon lat          (row index = node_id)
  .arcs   from_node to_node (row index = arc_id)
  .route  arc_id           (chronological ground-truth arcs)
"""
from pathlib import Path

import pandas as pd

from common.config import split_dir


def list_trips(split: str) -> list[str]:
    return sorted(p.name for p in split_dir(split).iterdir() if p.is_dir())


def _read(path: Path, names: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=names)
    return pd.read_csv(path, sep=r"\s+", header=None, names=names)


def load_trip(trip_id: str, split: str, with_route: bool = True) -> dict:
    d = split_dir(split) / trip_id
    out = {
        "trip_id": trip_id,
        "track": _read(d / f"{trip_id}.track", ["lon", "lat", "t"]),
        "nodes": _read(d / f"{trip_id}.nodes", ["lon", "lat"]),
        "arcs": _read(d / f"{trip_id}.arcs", ["u", "v"]).astype("int64"),
    }
    out["nodes"].index.name = "node_id"
    out["arcs"].index.name = "arc_id"
    # Ground truth: never pass with_route=True inside the inference path.
    if with_route:
        out["route"] = _read(d / f"{trip_id}.route", ["arc_id"])["arc_id"].astype("int64").tolist()
    return out


if __name__ == "__main__":
    for split in ("train", "student_test"):
        ids = list_trips(split)
        t = load_trip(ids[0], split)
        print(split, len(ids), "trips | first:", ids[0],
              {k: len(v) for k, v in t.items() if k != "trip_id"})
