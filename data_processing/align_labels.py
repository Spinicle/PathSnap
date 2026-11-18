"""Per-point labels from .route.
.route is an ordered arc list, NOT one label per GPS point. Monotonic alignment:
choose route positions r_0 <= r_1 <= ... <= r_{N-1} minimising sum_i dist(p_i, route[r_i]).

DP: C[i, j] = d[i, j] + min_{k <= j} C[i-1, k]   (prefix minimum over route positions)
Loop over points, vectorised over route positions; int32 back-pointers; backtrack.

Limitation: every point gets some route arc, even points recorded before/after the
route or inside long gaps; those show up as large align_dist_m.
Output: DataFrame[trip_id, obs_idx, true_arc_id, route_pos, align_dist_m]
"""
import numpy as np
import pandas as pd

from features.candidates import point_seg, project_trip

CHUNK = 2000


def _dist_matrix(px, py, ax, ay, bx, by) -> np.ndarray:
    N, R = len(px), len(ax)
    D = np.empty((N, R), dtype=np.float32)
    for s in range(0, N, CHUNK):
        e = min(N, s + CHUNK)
        d, _ = point_seg(px[s:e, None], py[s:e, None], ax[None], ay[None], bx[None], by[None])
        D[s:e] = d
    return D


def align_trip(trip: dict) -> pd.DataFrame:
    cols = ["trip_id", "obs_idx", "true_arc_id", "route_pos", "align_dist_m"]
    route = np.asarray(trip["route"], dtype=np.int64)
    P = project_trip(trip)
    px, py, nx, ny = P["px"], P["py"], P["nx"], P["ny"]
    N, R = len(px), len(route)
    if N == 0 or R == 0:
        return pd.DataFrame(columns=cols)
    arcs = trip["arcs"]
    u = arcs["u"].to_numpy(np.int64)[route]
    v = arcs["v"].to_numpy(np.int64)[route]
    D = _dist_matrix(px, py, nx[u], ny[u], nx[v], ny[v])

    back = np.empty((N, R), dtype=np.int32)
    back[0] = np.arange(R)
    C = D[0].astype(np.float64)
    idx = np.arange(R)
    for i in range(1, N):
        pm = np.minimum.accumulate(C)
        # argmin of the prefix: last position where the prefix min was set
        is_new = C <= pm
        arg = np.maximum.accumulate(np.where(is_new, idx, 0))
        back[i] = arg
        C = pm + D[i]

    path = np.empty(N, dtype=np.int64)
    path[-1] = int(np.argmin(C))
    for i in range(N - 1, 0, -1):
        path[i - 1] = back[i, path[i]]

    return pd.DataFrame({
        "trip_id": trip["trip_id"],
        "obs_idx": np.arange(N, dtype=np.int64),
        "true_arc_id": route[path],
        "route_pos": path,
        "align_dist_m": D[np.arange(N), path].astype(np.float32),
    })[cols]


def align_stats(df: pd.DataFrame, name: str = "") -> None:
    d = df["align_dist_m"].to_numpy()
    if len(d) == 0:
        print(f"[align {name}] no points")
        return
    print(f"[align {name}] points={len(d)} mean={d.mean():.2f} median={np.median(d):.2f} "
          f"p90={np.percentile(d, 90):.2f} p99={np.percentile(d, 99):.2f} max={d.max():.1f} "
          f">20m={(d > 20).mean():.3%} >50m={(d > 50).mean():.3%}")


def align_split(split: str, limit: int | None = None) -> pd.DataFrame:
    from data_processing.loader import list_trips, load_trip
    ids = list_trips(split)[:limit]
    df = pd.concat([align_trip(load_trip(t, split)) for t in ids], ignore_index=True)
    align_stats(df, split)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    align_split(a.split, a.limit)
