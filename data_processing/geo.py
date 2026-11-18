"""Projection to metres. Owner: A."""
from pyproj import Transformer

from common.config import CFG

_TO_M = Transformer.from_crs("EPSG:4326", CFG["geo"]["crs_projected"], always_xy=True)


def project(lon, lat):
    """lon/lat (deg) -> x/y (m) in UTM 36N."""
    return _TO_M.transform(lon, lat)
