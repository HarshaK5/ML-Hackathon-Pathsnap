"""Road network: projected geometry, spatial index, adjacency graph for routing."""
import numpy as np
import shapely
from shapely import STRtree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .data import Trip, make_projector


def seg_dist(P, A, B):
    """Distance from points P to segments A-B (all (n,2)); returns dist, frac u in [0,1], signed lateral."""
    ab = B - A
    L2 = (ab ** 2).sum(1)
    L2s = np.where(L2 < 1e-9, 1.0, L2)
    u = np.clip(((P - A) * ab).sum(1) / L2s, 0.0, 1.0)
    u = np.where(L2 < 1e-9, 0.0, u)
    Q = A + u[:, None] * ab
    d = np.hypot(*(P - Q).T)
    cross = ab[:, 0] * (P[:, 1] - A[:, 1]) - ab[:, 1] * (P[:, 0] - A[:, 0])
    lat = np.sign(cross) * d
    return d, u, lat


class RoadNetwork:
    def __init__(self, trip: Trip):
        lon0, lat0 = trip.track[:, 0].mean(), trip.track[:, 1].mean()
        self.tr = make_projector(lon0, lat0)
        self.px, self.py = self.tr.transform(trip.track[:, 0], trip.track[:, 1])
        nx, ny = self.tr.transform(trip.nodes[:, 0], trip.nodes[:, 1])
        self.nx, self.ny = np.asarray(nx), np.asarray(ny)
        self.arcs = trip.arcs
        self.n_nodes = len(self.nx)
        self.n_arcs = len(self.arcs)
        f, t = self.arcs[:, 0], self.arcs[:, 1]
        self.A = np.c_[self.nx[f], self.ny[f]]
        self.B = np.c_[self.nx[t], self.ny[t]]
        d = self.B - self.A
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 1], d[:, 0])
        self.out_deg = np.bincount(f, minlength=self.n_nodes)
        self.in_deg = np.bincount(t, minlength=self.n_nodes)
        # reverse-arc lookup (two-way street indicator)
        key = f.astype(np.int64) * self.n_nodes + t
        rkey = t.astype(np.int64) * self.n_nodes + f
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[order][pos] == rkey
        self.reverse = np.where(hit, order[pos], -1)
        self._tree = None
        self._graph = None
        self._arc_of = None

    # ---- spatial index (lazy; only needed for candidate generation) ----
    @property
    def tree(self):
        if self._tree is None:
            coords = np.stack([self.A, self.B], axis=1)           # (A,2,2)
            self._lines = shapely.linestrings(coords)
            self._tree = STRtree(self._lines)
        return self._tree

    # ---- routing graph (node graph, weight = arc length) ----
    @property
    def graph(self):
        if self._graph is None:
            f, t = self.arcs[:, 0], self.arcs[:, 1]
            # parallel arcs: keep the shortest for routing weight
            order = np.lexsort((self.length, t, f))
            f2, t2, w2 = f[order], t[order], self.length[order]
            first = np.r_[True, (f2[1:] != f2[:-1]) | (t2[1:] != t2[:-1])]
            self._graph = csr_matrix((np.maximum(w2[first], 1e-3), (f2[first], t2[first])),
                                     shape=(self.n_nodes, self.n_nodes))
            self._arc_of = {(int(a), int(b)): int(i) for i, a, b in zip(order[first], f2[first], t2[first])}
        return self._graph

    def arc_between(self, a, b):
        self.graph
        return self._arc_of.get((int(a), int(b)), -1)

    def sp_arcs(self, a_node, b_node, limit=np.inf):
        """Arc ids along shortest path node a -> node b ([] if unreachable)."""
        dist, pred = dijkstra(self.graph, directed=True, indices=a_node, return_predecessors=True, limit=limit)
        if not np.isfinite(dist[b_node]):
            return None
        path = [b_node]
        while path[-1] != a_node:
            path.append(int(pred[path[-1]]))
        path = path[::-1]
        return [self.arc_between(u, v) for u, v in zip(path[:-1], path[1:])]
