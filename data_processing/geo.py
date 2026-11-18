from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

_GEOD = Geod(ellps="WGS84")


def utm_epsg(lon: float, lat: float) -> int:
    """Return the UTM EPSG code for a longitude and hemisphere."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=64)
def _transformer(epsg: int) -> Transformer:
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)


def project(lon: Any, lat: Any) -> tuple[Any, Any]:
    """Project a set of points using the UTM zone of their mean location.

    Pass all points from one trip together. To project a trip's GPS points and
    road nodes with exactly the same CRS, use features.candidates.project_trip.
    """
    lons, lats = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    valid = np.isfinite(lons) & np.isfinite(lats)
    if not np.any(valid):
        raise ValueError("cannot select a UTM zone without finite coordinates")
    epsg = utm_epsg(float(np.mean(lons[valid])), float(np.mean(lats[valid])))
    return _transformer(epsg).transform(lon, lat)


def _point(lon: Any, lat: Any) -> tuple[float, float] | None:
    try:
        x, y = float(lon), float(lat)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y) or not (-180 <= x <= 180 and -90 <= y <= 90):
        return None
    return x, y


def distance_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    """WGS84 geodesic distance in metres; invalid coordinates return NaN."""
    a, b = _point(lon1, lat1), _point(lon2, lat2)
    if a is None or b is None:
        return math.nan
    return float(_GEOD.inv(a[0], a[1], b[0], b[1])[2])


def bearing_deg(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    """Compass bearing from point 1 to point 2 (north=0, clockwise)."""
    a, b = _point(lon1, lat1), _point(lon2, lat2)
    if a is None or b is None:
        return math.nan
    azimuth, _, dist = _GEOD.inv(a[0], a[1], b[0], b[1])
    return float(azimuth % 360) if math.isfinite(dist) and dist > 1e-9 else math.nan


def heading_deg(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    """Compass heading of travel from one GPS point to the next."""
    return bearing_deg(lon1, lat1, lon2, lat2)


def angular_difference_deg(start_deg: Any, end_deg: Any) -> float:
    """Signed shortest clockwise turn from start to end, in [-180, 180)."""
    try:
        a, b = float(start_deg), float(end_deg)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    return float((b - a + 180) % 360 - 180)


def speed_mps(lon1: Any, lat1: Any, t1: Any, lon2: Any, lat2: Any, t2: Any, *, max_gap_s: float | None = None) -> float:
    """Average geodesic speed in m/s; invalid or non-increasing times return NaN."""
    try:
        dt = float(t2) - float(t1)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(dt) or dt <= 0 or (max_gap_s is not None and dt > max_gap_s):
        return math.nan
    d = distance_m(lon1, lat1, lon2, lat2)
    return d / dt if math.isfinite(d) else math.nan


def track_motion(track: pd.DataFrame, *, max_gap_s: float | None = None) -> pd.DataFrame:
    """Per-row incoming-step time, distance, speed, and heading features."""
    missing = {"lon", "lat", "t"} - set(track.columns)
    if missing:
        raise ValueError(f"track is missing required columns: {sorted(missing)}")
    n = len(track)
    out = pd.DataFrame(np.nan, index=track.index, columns=["time_delta_s", "step_distance_m", "speed_mps", "heading_deg"])
    if n < 2:
        return out
    lon = pd.to_numeric(track["lon"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    lat = pd.to_numeric(track["lat"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    ts = pd.to_numeric(track["t"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    for i in range(1, n):
        dt = ts[i] - ts[i - 1]
        if not math.isfinite(dt) or dt <= 0 or (max_gap_s is not None and dt > max_gap_s):
            continue
        d = distance_m(lon[i - 1], lat[i - 1], lon[i], lat[i])
        if not math.isfinite(d):
            continue
        out.iloc[i] = [dt, d, d / dt, bearing_deg(lon[i - 1], lat[i - 1], lon[i], lat[i])]
    return out
