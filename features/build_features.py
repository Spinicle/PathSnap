"""Must not read .route except to attach the label column during training."""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from common.config import REPO_ROOT
from features.candidates import get_candidates, project_trip

CACHE = REPO_ROOT / "cache"
MIN_SPEED_FOR_HEADING = 1.0  # m/s

FEATURE_COLS = [
    "f_dist_m", "f_dist_rank", "f_dist_minus_min", "f_n_candidates",
    "f_gps_speed", "f_heading_diff", "f_arc_length_m", "f_proj_frac",
    "f_connected_prev",
]


def _neighbour_deltas(px, py, t):
    """Central differences i-1 -> i+1, one-sided at the ends."""
    n = len(px)
    lo = np.maximum(np.arange(n) - 1, 0)
    hi = np.minimum(np.arange(n) + 1, n - 1)
    return px[hi] - px[lo], py[hi] - py[lo], t[hi] - t[lo]


def _speed(px, py, t):
    dx, dy, dt = _neighbour_deltas(px, py, t)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.hypot(dx, dy) / dt
    return np.where(dt > 0, s, 0.0)


def _bearing(dx, dy):
    """Compass bearing in degrees (0 = north, clockwise) from projected deltas."""
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def _angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def build_trip_features(trip: dict) -> pd.DataFrame:
    """One row per (point, candidate arc). Never reads trip['route']."""
    cand = get_candidates(trip)
    if len(cand) == 0:
        return pd.DataFrame(columns=["trip_id", "obs_idx", "arc_id"] + FEATURE_COLS)
    P = project_trip(trip)
    px, py, nx, ny = P["px"], P["py"], P["nx"], P["ny"]
    t = trip["track"]["t"].to_numpy(float)
    arcs = trip["arcs"]
    u_all = arcs["u"].to_numpy(np.int64)
    v_all = arcs["v"].to_numpy(np.int64)

    obs = cand["obs_idx"].to_numpy()
    aid = cand["arc_id"].to_numpy()
    df = cand[["trip_id", "obs_idx", "arc_id"]].copy()

    df["f_dist_m"] = cand["dist_m"].to_numpy()
    g = df.groupby("obs_idx", sort=False)["f_dist_m"]
    df["f_dist_rank"] = g.cumcount().to_numpy()
    df["f_dist_minus_min"] = df["f_dist_m"] - g.transform("min")
    df["f_n_candidates"] = g.transform("size")

    speed = _speed(px, py, t)
    dx, dy, _ = _neighbour_deltas(px, py, t)
    gps_bear = _bearing(dx, dy)
    df["f_gps_speed"] = speed[obs]

    u, v = u_all[aid], v_all[aid]
    adx, ady = nx[v] - nx[u], ny[v] - ny[u]
    alen = np.hypot(adx, ady)
    hd = _angle_diff(gps_bear[obs], _bearing(adx, ady))
    df["f_heading_diff"] = np.where((speed[obs] < MIN_SPEED_FOR_HEADING) | (alen == 0), 90.0, hd)
    df["f_arc_length_m"] = alen
    df["f_proj_frac"] = cand["proj_frac"].to_numpy()

    # connectivity to previous point's nearest candidate (no labels used)
    nearest = cand.groupby("obs_idx", sort=False)["arc_id"].first()
    n_pts = len(px)
    prev_arc = np.full(n_pts, -1, dtype=np.int64)
    nearest_arr = np.full(n_pts, -1, dtype=np.int64)
    nearest_arr[nearest.index.to_numpy()] = nearest.to_numpy()
    prev_arc[1:] = nearest_arr[:-1]
    pa = prev_arc[obs]
    has_prev = pa >= 0
    pa_c = np.where(has_prev, pa, 0)
    pu, pv = u_all[pa_c], v_all[pa_c]
    conn = (aid == pa) | (u == pu) | (u == pv) | (v == pu) | (v == pv)
    df["f_connected_prev"] = (has_prev & conn).astype(np.float32)

    for c in FEATURE_COLS:
        df[c] = df[c].astype(np.float32)
    return df.reset_index(drop=True)


def _process_trip(args):
    trip_id, split = args
    from data_processing.align_labels import align_trip
    from data_processing.loader import load_trip
    trip = load_trip(trip_id, split, with_route=True)
    feats = build_trip_features(trip)
    n_pts = len(trip["track"])
    if trip.get("route"):
        lab = align_trip(trip)
        true_arc = lab["true_arc_id"].to_numpy()
        feats["label"] = (feats["arc_id"].to_numpy() == true_arc[feats["obs_idx"].to_numpy()]).astype(np.int8)
        hit_pts = feats.loc[feats["label"] == 1, "obs_idx"].nunique()
        misses = n_pts - hit_pts
    else:
        lab = pd.DataFrame()
        feats["label"] = np.int8(-1)
        misses = np.nan
    return feats, lab, dict(trip_id=trip_id, n_pts=n_pts, misses=misses,
                            pts_with_cands=feats["obs_idx"].nunique())


def build_split(split: str, limit: int | None = None, workers: int | None = None) -> pd.DataFrame:
    from data_processing.align_labels import align_stats
    from data_processing.loader import list_trips
    t0 = time.time()
    ids = list_trips(split)[:limit]
    workers = workers or max(1, min(6, (os.cpu_count() or 2) - 1))
    tasks = [(t, split) for t in ids]
    if workers == 1:
        results = [_process_trip(a) for a in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_process_trip, tasks))
    feats = pd.concat([r[0] for r in results], ignore_index=True)
    labs = pd.concat([r[1] for r in results], ignore_index=True)
    st = pd.DataFrame([r[2] for r in results])

    CACHE.mkdir(exist_ok=True)
    feats.to_parquet(CACHE / f"features_{split}.parquet", index=False)
    if len(labs):
        labs.to_parquet(CACHE / f"labels_{split}.parquet", index=False)

    n_pts = st["n_pts"].sum()
    print(f"\n=== {split}: {len(ids)} trips, {n_pts} points, {len(feats)} rows "
          f"({len(feats) / max(n_pts, 1):.2f} cands/point)")
    print(f"positive rate        {feats['label'].mean():.4f}")
    print(f"candidate miss rate  {st['misses'].sum() / n_pts:.4%}  "
          f"(no-candidate points {1 - st['pts_with_cands'].sum() / n_pts:.4%})")
    print(f"NaNs in features     {int(feats[FEATURE_COLS].isna().sum().sum())}")
    if len(labs):
        align_stats(labs, split)
    worst = st.assign(miss_rate=st["misses"] / st["n_pts"]).nlargest(3, "miss_rate")
    print("worst trips by miss rate:\n", worst[["trip_id", "n_pts", "miss_rate"]].to_string(index=False))
    print(f"runtime              {time.time() - t0:.1f}s  -> {CACHE / f'features_{split}.parquet'}")
    return feats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "student_test", "all"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    for s in (["train", "student_test"] if a.split == "all" else [a.split]):
        build_split(s, a.limit, a.workers)
