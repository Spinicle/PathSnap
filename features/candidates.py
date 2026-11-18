"""Trips span lon -100..137, so a fixed UTM 36N is wrong for most; each trip uses the UTM zone of its track centroid."""
from functools import lru_cache

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from common.config import CFG

PREFILTER_PAD_M = 2000.0


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project_trip(trip: dict) -> dict:
    """Per-trip UTM projection of track and nodes, cached on the trip dict so alignment and features share it."""
    if "_proj" in trip:
        return trip["_proj"]
    tr, nd = trip["track"], trip["nodes"]
    epsg = utm_epsg(float(tr["lon"].mean()), float(tr["lat"].mean()))
    tf = _transformer(epsg)
    px, py = tf.transform(tr["lon"].to_numpy(float), tr["lat"].to_numpy(float))
    nx, ny = tf.transform(nd["lon"].to_numpy(float), nd["lat"].to_numpy(float))
    trip["_proj"] = dict(epsg=epsg, px=np.asarray(px), py=np.asarray(py),
                         nx=np.asarray(nx), ny=np.asarray(ny))
    return trip["_proj"]


def point_seg(px, py, ax, ay, bx, by):
    """Point-to-segment distance and 0..1 projection fraction (broadcasting); zero-length segment gives frac 0."""
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        f = ((px - ax) * dx + (py - ay) * dy) / L2
    f = np.where(L2 > 0, np.clip(f, 0.0, 1.0), 0.0)
    qx, qy = ax + f * dx, ay + f * dy
    return np.hypot(px - qx, py - qy), f


def get_candidates(trip: dict, radius_m: float | None = None,
                   max_k: int | None = None) -> pd.DataFrame:
    radius_m = CFG["candidates"]["radius_m"] if radius_m is None else radius_m
    max_k = CFG["candidates"]["max_k"] if max_k is None else max_k
    cols = ["trip_id", "obs_idx", "arc_id", "dist_m", "proj_frac"]
    P = project_trip(trip)
    px, py, nx, ny = P["px"], P["py"], P["nx"], P["ny"]
    arcs = trip["arcs"]
    if len(px) == 0 or len(arcs) == 0:
        return pd.DataFrame(columns=cols)

    u = arcs["u"].to_numpy(np.int64)
    v = arcs["v"].to_numpy(np.int64)
    ax, ay, bx, by = nx[u], ny[u], nx[v], ny[v]

    # prefilter: arcs whose bbox overlaps the padded track bbox
    pad = radius_m + PREFILTER_PAD_M
    x0, x1 = px.min() - pad, px.max() + pad
    y0, y1 = py.min() - pad, py.max() + pad
    keep = ((np.maximum(ax, bx) >= x0) & (np.minimum(ax, bx) <= x1) &
            (np.maximum(ay, by) >= y0) & (np.minimum(ay, by) <= y1))
    arc_ids = np.flatnonzero(keep)
    if len(arc_ids) == 0:
        return pd.DataFrame(columns=cols)

    coords = np.stack([np.stack([ax[keep], ay[keep]], 1),
                       np.stack([bx[keep], by[keep]], 1)], 1)  # (M, 2, 2)
    tree = shapely.STRtree(shapely.linestrings(coords))
    pts = shapely.points(px, py)
    pi, ai = tree.query(pts, predicate="dwithin", distance=radius_m)
    if len(pi) == 0:
        return pd.DataFrame(columns=cols)

    gid = arc_ids[ai]
    d, f = point_seg(px[pi], py[pi], ax[gid], ay[gid], bx[gid], by[gid])
    df = pd.DataFrame({"obs_idx": pi.astype(np.int64), "arc_id": gid.astype(np.int64),
                       "dist_m": d, "proj_frac": f})
    df = df.sort_values(["obs_idx", "dist_m", "arc_id"], kind="mergesort")
    df = df[df.groupby("obs_idx").cumcount() < max_k].reset_index(drop=True)
    df.insert(0, "trip_id", trip["trip_id"])
    return df[cols]


if __name__ == "__main__":
    import sys
    from data_processing.loader import load_trip, list_trips
    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    t = load_trip(list_trips(split)[0], split, with_route=False)
    c = get_candidates(t)
    print(c.head(), "\nrows", len(c), "points with cands", c.obs_idx.nunique(), "/", len(t["track"]))
