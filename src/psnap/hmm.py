"""HMM + Viterbi route reconstruction.

Hidden states   : at GPS step t, the candidate road arcs C_t (same candidate set the classifier scored, so the
                  candidate -> arc correspondence is preserved end to end).
Emission        : log P(arc | observation) = log of the classifier's probability that the candidate is the true arc.
Transition      : Newson-Krumm style. For candidates a (step t-1) and b (step t) we compute the ROAD-NETWORK
                  travel distance d_route(a->b) (position on a -> end of a -> shortest path -> start of b -> position
                  on b) and compare with the straight GPS displacement d_gps:  log T = -|d_route - d_gps| / beta.
                  Unreachable pairs get a fixed penalty (so decoding can always continue = 'route break'),
                  U-turns get an extra penalty, and after long GPS gaps only reachability matters.
Decoding        : Viterbi dynamic programme (max-sum in log space).
The expensive part (route distances) does not depend on HMM parameters, so it is precomputed once per trip
(`Lattice`) and parameter tuning is then fast."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from .network import RoadNetwork, seg_dist

CACHE_LIMIT = 500.0      # metres explored per cached Dijkstra source
VMAX = 45.0              # m/s, used to bound search after long gaps
CORRIDOR_M = 700.0       # routing sub-graph = nodes within this distance of any GPS point
MAX_GAP_SEARCH = 15000.0


@dataclass
class HMMParams:
    beta: float = 12.0          # metres: scale of |d_route - d_gps|
    lam: float = 1.0            # weight of classifier emission
    unreach_pen: float = 12.0   # log-penalty for an unreachable transition (route break)
    uturn_pen: float = 6.0      # extra log-penalty for reversing onto the opposite arc
    switch_pen: float = 0.0     # log-penalty for any arc change (smoothness)
    gap_thresh: float = 20.0    # seconds: above this only reachability is used

    def as_dict(self):
        return asdict(self)


class Lattice:
    """Candidate lattice with precomputed route distances (independent of HMM params)."""

    def __init__(self, net: RoadNetwork, t, pi, ai):
        self.net = net
        N = len(net.px)
        self.N = N
        self.pi, self.ai = pi, ai
        P = np.c_[net.px, net.py]
        _, u, _ = seg_dist(P[pi], net.A[ai], net.B[ai])
        self.u = u
        self.start = np.r_[0, np.flatnonzero(pi[1:] != pi[:-1]) + 1]
        self.end = np.r_[self.start[1:], len(pi)]
        assert len(self.start) == N, "every GPS point needs >=1 candidate"
        self.t = np.asarray(t, dtype=float)
        self.dgps = np.r_[0.0, np.hypot(np.diff(net.px), np.diff(net.py))]
        self.dt = np.r_[0.0, np.diff(self.t)]
        self._sub()
        self._route_dists()

    # ---- routing sub-graph (compact node ids) ----
    def _sub(self):
        net = self.net
        kd = cKDTree(np.c_[net.nx, net.ny])
        near = kd.query_ball_point(np.c_[net.px[::3], net.py[::3]], CORRIDOR_M)
        keep = np.unique(np.concatenate([np.asarray(n, dtype=np.int64) for n in near])) if len(near) else np.array([], int)
        lmap = -np.ones(net.n_nodes, dtype=np.int64)
        lmap[keep] = np.arange(len(keep))
        f, t = net.arcs[:, 0], net.arcs[:, 1]
        m = (lmap[f] >= 0) & (lmap[t] >= 0)
        self.lmap = lmap
        self.g = csr_matrix((np.maximum(net.length[m], 1e-3), (lmap[f[m]], lmap[t[m]])), shape=(len(keep), len(keep)))
        self._cache = {}

    def _src(self, node):
        """Sparse (sorted node ids, dists) reachable within CACHE_LIMIT from `node`."""
        c = self._cache.get(node)
        if c is None:
            ln = self.lmap[node]
            if ln < 0:
                c = (np.array([], dtype=np.int64), np.array([]))
            else:
                d = dijkstra(self.g, directed=True, indices=ln, limit=CACHE_LIMIT)
                idx = np.flatnonzero(np.isfinite(d)); c = (idx, d[idx])
            self._cache[node] = c
        return c

    def _lookup(self, node, targets_local):
        idx, d = self._src(node)
        if len(idx) == 0:
            return np.full(len(targets_local), np.inf)
        pos = np.searchsorted(idx, targets_local); pos = np.clip(pos, 0, len(idx) - 1)
        return np.where(idx[pos] == targets_local, d[pos], np.inf)

    def _route_dists(self):
        net = self.net
        self.D = [None]                    # D[s]: (K_{s-1}, K_s) route distances, inf = unreachable
        for s in range(1, self.N):
            a = self.ai[self.start[s - 1]:self.end[s - 1]]; ua = self.u[self.start[s - 1]:self.end[s - 1]]
            b = self.ai[self.start[s]:self.end[s]]; ub = self.u[self.start[s]:self.end[s]]
            dgps, dt = self.dgps[s], self.dt[s]
            big = dt > 20.0 or dgps > 0.6 * CACHE_LIMIT
            ends = net.arcs[a, 1]
            starts_local = self.lmap[net.arcs[b, 0]]
            sp = np.full((len(a), len(b)), np.inf)
            if not big:
                for i, e in enumerate(ends):
                    sp[i] = self._lookup(int(e), starts_local)
            else:
                lim = min(max(2 * dgps + 300, VMAX * max(dt, 1.0)), MAX_GAP_SEARCH)
                for i, e in enumerate(ends):
                    le = self.lmap[e]
                    if le < 0:
                        continue
                    d = dijkstra(self.g, directed=True, indices=le, limit=lim)
                    ok = starts_local >= 0
                    sp[i, ok] = d[starts_local[ok]]
            D = (net.length[a] * (1 - ua))[:, None] + sp + (net.length[b] * ub)[None, :]
            same = a[:, None] == b[None, :]
            D = np.where(same, np.abs(ub[None, :] - ua[:, None]) * net.length[b][None, :], D)
            self.D.append(D.astype(np.float32))

    def cand_slice(self, s):
        return slice(self.start[s], self.end[s])


def viterbi(lat: Lattice, p, params: HMMParams):
    """p: classifier probability per candidate row (aligned with lat.pi/lat.ai). Returns chosen candidate row per step."""
    net = lat.net
    N = lat.N
    emis = params.lam * np.log(np.clip(p, 1e-6, 1.0))
    score = emis[lat.cand_slice(0)].copy()
    back = [None]
    rev = net.reverse
    for s in range(1, N):
        sl = lat.cand_slice(s); sp_ = lat.cand_slice(s - 1)
        D = lat.D[s]
        a, b = lat.ai[sp_], lat.ai[sl]
        if lat.dt[s] <= params.gap_thresh:
            lt = -np.abs(D - lat.dgps[s]) / params.beta
        else:
            lt = -0.001 * D
        lt = np.where(np.isfinite(D), lt, -params.unreach_pen)
        diff = a[:, None] != b[None, :]
        if params.switch_pen:
            lt = lt - params.switch_pen * diff
        if params.uturn_pen:
            lt = lt - params.uturn_pen * (rev[a][:, None] == b[None, :])
        tot = score[:, None] + lt
        arg = np.argmax(tot, axis=0)
        score = tot[arg, np.arange(tot.shape[1])] + emis[sl]
        back.append(arg)
    j = int(np.argmax(score))
    path = np.empty(N, dtype=np.int64)
    path[-1] = lat.start[N - 1] + j
    for s in range(N - 1, 0, -1):
        j = int(back[s][j])
        path[s - 1] = lat.start[s - 1] + j
    return path                                     # candidate-row index per GPS step


def complete_route(net: RoadNetwork, arc_per_point, limit=3000.0):
    """Collapse per-point arcs to a route; bridge consecutive disconnected arcs with the shortest path.
    Returns (route arcs, n_bridged, n_unbridged_breaks)."""
    seq = np.asarray(arc_per_point)
    keep = np.r_[True, seq[1:] != seq[:-1]]
    s = seq[keep]
    route, bridged, broken = [int(s[0])], 0, 0
    for a, b in zip(s[:-1], s[1:]):
        ea, sb = net.arcs[a, 1], net.arcs[b, 0]
        if ea != sb:
            path = net.sp_arcs(int(ea), int(sb), limit=limit)
            if path is None:
                broken += 1
            else:
                route += [x for x in path if x >= 0]; bridged += 1
        route.append(int(b))
    return route, bridged, broken
