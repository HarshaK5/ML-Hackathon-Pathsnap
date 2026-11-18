"""Raw-data loading. File formats (no headers, whitespace separated):
.track  lon lat t(s)   | .nodes lon lat (row = node id) | .arcs from to (row = arc id)
.route  arc_id per row (reference route, chronological)."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from pyproj import Transformer

from .config import DATA_ROOT


@dataclass
class Trip:
    trip_id: str
    split: str
    track: np.ndarray            # (N,3) lon, lat, t
    nodes: np.ndarray            # (M,2) lon, lat
    arcs: np.ndarray             # (A,2) int from,to
    route: Optional[np.ndarray]  # (R,) int arc ids or None (never given to the model)
    path: Path = None


def trip_dir(split_dir: Path, trip_id: str) -> Path:
    return Path(split_dir) / trip_id


def load_trip(split_dir, trip_id: str, with_route: bool = True, split: str = "") -> Trip:
    base = trip_dir(split_dir, trip_id) / trip_id
    track = np.loadtxt(str(base) + ".track", ndmin=2)
    nodes = np.loadtxt(str(base) + ".nodes", ndmin=2)
    arcs = np.loadtxt(str(base) + ".arcs", dtype=np.int64, ndmin=2)
    route = None
    rp = Path(str(base) + ".route")
    if with_route and rp.exists():
        route = np.loadtxt(rp, dtype=np.int64, ndmin=1)
    return Trip(trip_id, split or Path(split_dir).name, track, nodes, arcs, route, Path(split_dir))


def list_trips(split_dir) -> list:
    return sorted(p.name for p in Path(split_dir).iterdir() if p.is_dir() and not p.name.startswith("."))


def make_projector(lon0: float, lat0: float) -> Transformer:
    """Local azimuthal-equidistant projection centred on the trip: metres, accurate
    (distance error <<1 %) over the tens of km a trip spans. Euclidean distance in
    degrees would be wrong (a degree of longitude shrinks with cos(lat))."""
    return Transformer.from_crs("EPSG:4326",
                                f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m",
                                always_xy=True)
