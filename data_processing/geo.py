"""Geographic and motion helpers used by the PathSnap feature pipeline.

Coordinate arguments are always passed as ``(longitude, latitude)`` in degrees.
Bearings/headings use compass degrees: 0 is north and angles increase clockwise.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

from common.config import CFG

_TO_M = Transformer.from_crs(
    "EPSG:4326", CFG["geo"]["crs_projected"], always_xy=True
)
_GEOD = Geod(ellps="WGS84")


def project(lon: Any, lat: Any) -> tuple[Any, Any]:
    """Project longitude/latitude to the configured metric CRS.

    Scalars, lists, and NumPy arrays are accepted. ``always_xy=True`` keeps the
    input order as longitude then latitude.
    """
    return _TO_M.transform(lon, lat)


def _coordinate_pair(lon: Any, lat: Any) -> tuple[float, float] | None:
    """Return a finite, geographically valid lon/lat pair, or ``None``."""
    try:
        lon_value = float(lon)
        lat_value = float(lat)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lon_value) and math.isfinite(lat_value)):
        return None
    if not -180.0 <= lon_value <= 180.0 or not -90.0 <= lat_value <= 90.0:
        return None
    return lon_value, lat_value


def distance_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    """Return WGS84 geodesic distance in metres between two lon/lat points.

    Invalid coordinates produce ``NaN`` so a bad observation can be handled as
    a missing feature instead of stopping the rest of a trip.
    """
    start = _coordinate_pair(lon1, lat1)
    end = _coordinate_pair(lon2, lat2)
    if start is None or end is None:
        return math.nan
    _, _, distance = _GEOD.inv(start[0], start[1], end[0], end[1])
    return float(distance)


def bearing_deg(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    """Return the compass bearing from the first point to the second.

    The result is in ``[0, 360)`` degrees (north is 0, east is 90). A bearing
    is undefined for coincident or invalid points, in which case ``NaN`` is
    returned.
    """
    start = _coordinate_pair(lon1, lat1)
    end = _coordinate_pair(lon2, lat2)
    if start is None or end is None:
        return math.nan
    azimuth, _, distance = _GEOD.inv(start[0], start[1], end[0], end[1])
    if not math.isfinite(distance) or distance <= 1e-9:
        return math.nan
    return float(azimuth % 360.0)


def heading_deg(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> float:
    """Return the direction of travel from one GPS point to the next.

    This uses the same compass convention as :func:`bearing_deg`; the separate
    name makes call sites clear when the points come from a GPS trajectory.
    """
    return bearing_deg(lon1, lat1, lon2, lat2)


def angular_difference_deg(start_deg: Any, end_deg: Any) -> float:
    """Return the signed shortest turn from ``start_deg`` to ``end_deg``.

    The result is in ``[-180, 180)``; positive values are clockwise turns.
    """
    try:
        start = float(start_deg)
        end = float(end_deg)
    except (TypeError, ValueError):
        return math.nan
    if not (math.isfinite(start) and math.isfinite(end)):
        return math.nan
    return float((end - start + 180.0) % 360.0 - 180.0)


def speed_mps(
    lon1: Any,
    lat1: Any,
    t1: Any,
    lon2: Any,
    lat2: Any,
    t2: Any,
    *,
    max_gap_s: float | None = None,
) -> float:
    """Return average speed between two timed GPS observations in m/s.

    Non-finite coordinates/timestamps, non-increasing timestamps, and gaps
    longer than ``max_gap_s`` produce ``NaN``. No gap limit is applied when
    ``max_gap_s`` is omitted.
    """
    try:
        elapsed_s = float(t2) - float(t1)
    except (TypeError, ValueError):
        return math.nan
    if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
        return math.nan
    if max_gap_s is not None:
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("max_gap_s must be a positive finite number")
        if elapsed_s > max_gap_s:
            return math.nan
    distance = distance_m(lon1, lat1, lon2, lat2)
    if not math.isfinite(distance):
        return math.nan
    return distance / elapsed_s


def track_motion(
    track: pd.DataFrame, *, max_gap_s: float | None = None
) -> pd.DataFrame:
    """Compute incoming-step motion features for each row of a GPS track.

    ``track`` must contain ``lon``, ``lat``, and ``t`` columns. The returned
    frame preserves its index and row order and contains:

    * ``time_delta_s``: elapsed seconds from the preceding observation;
    * ``step_distance_m``: geodesic distance from the preceding observation;
    * ``speed_mps``: distance divided by elapsed time;
    * ``heading_deg``: compass direction of that incoming step.

    The first row has ``NaN`` motion values. Invalid or non-increasing times,
    invalid coordinates, and (when supplied) overlong gaps also produce
    ``NaN`` values for that step.
    """
    missing = {"lon", "lat", "t"} - set(track.columns)
    if missing:
        raise ValueError(f"track is missing required columns: {sorted(missing)}")
    if max_gap_s is not None and (
        not math.isfinite(max_gap_s) or max_gap_s <= 0
    ):
        raise ValueError("max_gap_s must be a positive finite number")

    names = ["time_delta_s", "step_distance_m", "speed_mps", "heading_deg"]
    values = {name: np.full(len(track), np.nan, dtype=float) for name in names}
    if len(track) < 2:
        return pd.DataFrame(values, index=track.index)

    lon = pd.to_numeric(track["lon"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    lat = pd.to_numeric(track["lat"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    timestamp = pd.to_numeric(track["t"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )

    for idx in range(1, len(track)):
        elapsed_s = timestamp[idx] - timestamp[idx - 1]
        if not math.isfinite(elapsed_s) or elapsed_s <= 0.0:
            continue
        if max_gap_s is not None and elapsed_s > max_gap_s:
            continue

        start = _coordinate_pair(lon[idx - 1], lat[idx - 1])
        end = _coordinate_pair(lon[idx], lat[idx])
        if start is None or end is None:
            continue

        distance = distance_m(start[0], start[1], end[0], end[1])
        if not math.isfinite(distance):
            continue
        values["time_delta_s"][idx] = elapsed_s
        values["step_distance_m"][idx] = distance
        values["speed_mps"][idx] = distance / elapsed_s
        values["heading_deg"][idx] = bearing_deg(
            start[0], start[1], end[0], end[1]
        )

    return pd.DataFrame(values, index=track.index)
